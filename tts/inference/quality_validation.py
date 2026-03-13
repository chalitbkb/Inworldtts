import abc
import os

import torch
import torchaudio
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.core.codec import decoding, encoding
from tts.data import data_utils, text_normalization
from tts.inference import inferencing

_TEST_COMBINATION = tuple[str, str, str]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwraps the model to use for quality validation."""
    result = model

    if hasattr(result, "_forward_module"):
        # FSDP requires the model to be wrapped in a module.
        if isinstance(
            result._forward_module, torch.distributed.fsdp.FullyShardedDataParallel
        ):
            return result
        result = result._forward_module

    if hasattr(result, "_orig_mod"):
        if hasattr(result._orig_mod, "module"):
            result = result._orig_mod.module
        else:
            result = result._orig_mod

    # Unwrap DDP.
    if isinstance(result, torch.nn.parallel.DistributedDataParallel):
        result = result.module

    return result


def _get_test_combinations(
    prompt_wav_path: str, prompt_text: str, phrases: list[str]
) -> list[_TEST_COMBINATION]:
    """Returns all test combinations of prompt wavs and phrases."""
    result = []
    for phrase in phrases:
        result.append((prompt_wav_path, prompt_text, phrase))
    return result


class QualityValidator(metaclass=abc.ABCMeta):
    """Abstract base class for computing quality validation artifacts/metrics."""

    @abc.abstractmethod
    def validate(self, model: torch.nn.Module, step: int):
        raise NotImplementedError("|validate| must be implemented by subclasses.")


class NoOpQualityValidator(QualityValidator):
    """Quality validator that does nothing."""

    def validate(self, model: torch.nn.Module, step: int):
        del model, step  # Unused.


# TODO: improve and cover more use cases (eg voice description).
class RandomPhrasesSynthesizer(QualityValidator):
    """Quality validator that synthesizes random phrases with the codec on CPU."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        checkpointing_dir: str,
        global_rank: int,
        world_size: int,
        device: torch.device,
        prompt_compiler: prompting.PromptCompiler,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        phrases: list[str] | None,
        codec_encoder_checkpoint_path: str | None,
        codec_decoder_checkpoint_path: str | None,
        enable_text_normalization: bool = True,
    ):
        self._tokenizer = tokenizer
        self._audio_encoder = encoding.CachingAudioEncoder(
            codec_encoder_checkpoint_path, device
        )
        self._audio_decoder = decoding.create(codec_decoder_checkpoint_path, device)
        self._checkpointing_dir = checkpointing_dir
        self._global_rank = global_rank
        self._world_size = world_size
        self._device = device
        self._prompt_compiler = prompt_compiler
        self._prompt_wav_path = prompt_wav_path
        self._prompt_text = prompt_text
        self._phrases = phrases
        # By default, we enable text normalization.
        self._text_normalizer = text_normalization.create_text_normalizer(
            enable_text_normalization
        )

    def _load_prompt_wav(self, prompt_wav_path: str) -> torch.Tensor:
        prompt_wav, _ = data_utils.load_wav(
            prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE
        )
        return prompt_wav

    def _select_test_combinations(
        self, test_combinations: list[_TEST_COMBINATION]
    ) -> list[_TEST_COMBINATION]:
        if self._world_size == 1:
            return test_combinations

        n_items = len(test_combinations)
        left = (self._global_rank * n_items) // self._world_size
        right = ((self._global_rank + 1) * n_items) // self._world_size
        right = min(right, n_items)
        return test_combinations[left:right]

    def _get_model(self, model: torch.nn.Module) -> inferencing.LocalTtsModel:
        return inferencing.LocalTtsModel(
            model=_unwrap_model(model),
            device=self._device,
            tokenizer=self._tokenizer,
            audio_encoder=self._audio_encoder,
            audio_decoder=self._audio_decoder,
            prompt_compiler=self._prompt_compiler,
        )

    # TODO: consider reusing more local inferencing code.
    def validate(self, model: torch.nn.Module, step: int) -> None:
        generation_dir = os.path.join(self._checkpointing_dir, f"generations/{step}/")
        os.makedirs(generation_dir, exist_ok=True)
        logging.info(
            "Starting to synthesize test phrases for step %d. They will be saved in %s",
            step,
            generation_dir,
        )

        if not self._prompt_wav_path or not self._prompt_text or not self._phrases:
            test_combinations = []
        else:
            test_combinations = self._select_test_combinations(
                _get_test_combinations(str(self._prompt_wav_path), str(self._prompt_text), self._phrases)
            )

        if not test_combinations:
            logging.warning("No test combinations available for Quality Validation.")
            return

        logging.info("Synthesizing %d phrases...", len(test_combinations))

        # TODO: support batch inference.
        for idx, (prompt_wav_path, prompt_text, phrase) in enumerate(test_combinations):
            prompt_wav = self._load_prompt_wav(prompt_wav_path)
            phrase = self._text_normalizer.normalize(phrase)
            inference_result = self._get_model(model).synthesize_speech(
                inference_settings=inferencing.DEFAULT_INFERENCE_SETTINGS,
                text_to_synthesize=phrase,
                prompt_id=prompt_wav_path,
                prompt_wav=prompt_wav,
                audio_prompt_transcription=prompt_text,
                voice_description="",
            )
            wav_path = os.path.join(
                generation_dir, f"rank_{self._global_rank}_{idx}.wav"
            )
            torchaudio.save(
                wav_path, inference_result.wav, self._audio_decoder.sample_rate
            )
            logging.info(
                "Synthesized %d/%d phrases...", idx + 1, len(test_combinations)
            )


class PromptContinuationValidator(QualityValidator):
    """Quality validator that continues audio prompts."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        checkpointing_dir: str,
        global_rank: int,
        world_size: int,
        device: torch.device,
        prompt_wav_path: str | None,
        codec_encoder_checkpoint_path: str | None,
        codec_decoder_checkpoint_path: str | None,
    ):
        """Initializes the prompt continuation validator."""
        self._tokenizer = tokenizer

        # Note: This class instantiates a copy of the encoder/decoder, so if someone
        # accidentally uses it with the random phrases for TTS - it'll own a copy of it.
        self._audio_encoder = encoding.create(codec_encoder_checkpoint_path, device)
        self._audio_decoder = decoding.create(codec_decoder_checkpoint_path, device)

        self._checkpointing_dir = checkpointing_dir
        self._global_rank = global_rank
        self._world_size = world_size
        self._device = device

        self._prompt_wav_paths = [prompt_wav_path] if prompt_wav_path else []

    def validate(self, model: torch.nn.Module, step: int):
        """Validates the model by continuing audio prompts."""
        if self._global_rank != 0:
            return

        generation_dir = os.path.join(
            self._checkpointing_dir, f"prompt_continuations/{step}/"
        )
        os.makedirs(generation_dir, exist_ok=True)
        logging.info(
            "Starting prompt continuation validation for step %d. "
            "Results will be saved in %s",
            step,
            generation_dir,
        )

        logging.info("Processing %d prompt(s)...", len(self._prompt_wav_paths))
        for idx, prompt_wav_path in enumerate(self._prompt_wav_paths):
            prompt_wav, _ = data_utils.load_wav(
                prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE
            )
            logging.info(
                "Processing prompt %d/%d: %.2fs long audio from %s",
                idx + 1,
                len(self._prompt_wav_paths),
                prompt_wav.shape[1] / constants.CODEC_SAMPLE_RATE,
                prompt_wav_path,
            )

            gen_wav = inferencing.complete_prompt(
                model=_unwrap_model(model),
                encoder=self._audio_encoder,
                tokenizer=self._tokenizer,
                decoder=self._audio_decoder,
                prompt_wav=prompt_wav,
                model_device=self._device,
                inference_settings=inferencing.DEFAULT_INFERENCE_SETTINGS,
            )

            # Save the continuation.
            base_name = f"prompt_{idx}"
            torchaudio.save(
                os.path.join(generation_dir, f"{base_name}_continuation.wav"),
                gen_wav,
                self._audio_decoder.sample_rate,
            )
            logging.info(
                "Completed validation for prompt %d/%d",
                idx + 1,
                len(self._prompt_wav_paths),
            )


def create_quality_validator(
    tokenizer: transformers.PreTrainedTokenizer,
    checkpointing_dir: str,
    save_intermediate_generations: bool,
    global_rank: int,
    world_size: int,
    device: torch.device,
    validation_type: str,
    checkpointing_config=None,
) -> QualityValidator:
    """Creates a quality validator for master process based on the provided settings."""
    if not save_intermediate_generations:
        return NoOpQualityValidator()

    if not checkpointing_config or not checkpointing_config.codec_encoder_checkpoint_path or not checkpointing_config.codec_decoder_checkpoint_path:
        logging.warning("Codec checkpoint path is not provided. Quality validation is disabled.")
        return NoOpQualityValidator()

    if validation_type == "prompt_continuation":
        return PromptContinuationValidator(
            tokenizer=tokenizer,
            checkpointing_dir=checkpointing_dir,
            global_rank=global_rank,
            world_size=world_size,
            device=device,
            prompt_wav_path=checkpointing_config.validation_prompt_wav,
            codec_encoder_checkpoint_path=checkpointing_config.codec_encoder_checkpoint_path,
            codec_decoder_checkpoint_path=checkpointing_config.codec_decoder_checkpoint_path,
        )
    prompt_compiler = prompting.InferencePromptCompiler()
    return RandomPhrasesSynthesizer(
        tokenizer=tokenizer,
        checkpointing_dir=checkpointing_dir,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
        prompt_compiler=prompt_compiler,
        prompt_wav_path=checkpointing_config.validation_prompt_wav,
        prompt_text=checkpointing_config.validation_prompt_text,
        phrases=checkpointing_config.validation_test_phrases,
        codec_encoder_checkpoint_path=checkpointing_config.codec_encoder_checkpoint_path,
        codec_decoder_checkpoint_path=checkpointing_config.codec_decoder_checkpoint_path,
    )

