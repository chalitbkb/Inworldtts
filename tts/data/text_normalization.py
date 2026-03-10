"""Input text normalization."""

import abc
import logging as base_logging
import re

import lingua
import unidecode
from nemo_text_processing.text_normalization import normalize

base_logging.getLogger("NeMo-text-processing").setLevel(base_logging.CRITICAL)

# Mapping from ISO 639-1 language codes to lingua Language enum.
# Add new languages here to automatically support them across the pipeline.
_LINGUA_LANG_MAP: dict[str, lingua.Language] = {
    "en": lingua.Language.ENGLISH,
    "ja": lingua.Language.JAPANESE,
    "zh": lingua.Language.CHINESE,
    "es": lingua.Language.SPANISH,
    "fr": lingua.Language.FRENCH,
    "de": lingua.Language.GERMAN,
    "ko": lingua.Language.KOREAN,
    "th": lingua.Language.THAI,
    "pt": lingua.Language.PORTUGUESE,
    "ru": lingua.Language.RUSSIAN,
    "it": lingua.Language.ITALIAN,
    "nl": lingua.Language.DUTCH,
    "pl": lingua.Language.POLISH,
    "ar": lingua.Language.ARABIC,
    "hi": lingua.Language.HINDI,
    "vi": lingua.Language.VIETNAMESE,
    "id": lingua.Language.INDONESIAN,
    "tr": lingua.Language.TURKISH,
    "sv": lingua.Language.SWEDISH,
}

# Reverse mapping: lingua Language -> language code.
_LINGUA_LANG_REVERSE_MAP: dict[lingua.Language, str] = {
    v: k for k, v in _LINGUA_LANG_MAP.items()
}

# Languages that NeMo text normalizer supports natively.
# Other languages will pass through without normalization.
_NEMO_SUPPORTED_LANGS = {"en", "ja", "zh", "es", "fr", "de"}

# Languages where ASCII conversion should be applied before normalization.
_ASCII_CONVERT_LANGS = {"en"}

# Common Thai abbreviations and their expanded forms for TTS.
# Languages that have custom normalization via PyThaiNLP.
_THAI_NORMALIZE_LANGS = {"th"}

_THAI_ABBREVIATIONS: dict[str, str] = {
    r"\bกทม\.": "กรุงเทพมหานคร",
    r"\bรพ\.": "โรงพยาบาล",
    r"\bชม\.": "ชั่วโมง",
    r"\bกก\.": "กิโลกรัม",
    r"\bกม\.": "กิโลเมตร",
    r"\bม\.": "เมตร",
    r"\bซม\.": "เซนติเมตร",
    r"\bมม\.": "มิลลิเมตร",
    r"\bล\.": "ลิตร",
    r"\bมล\.": "มิลลิลิตร",
    r"\bพ\.ศ\.": "พุทธศักราช",
    r"\bค\.ศ\.": "คริสต์ศักราช",
    r"\bน\.": "นาฬิกา",
    r"\bบ\.": "บาท",
    r"\bจ\.": "จังหวัด",
    r"\bอ\.": "อำเภอ",
    r"\bต\.": "ตำบล",
    r"\bถ\.": "ถนน",
    r"\bซ\.": "ซอย",
}

# Thai digit names for reading numbers one-by-one (phone numbers, IDs, etc.)
_THAI_DIGIT_NAMES: dict[str, str] = {
    "0": "ศูนย์",
    "1": "หนึ่ง",
    "2": "สอง",
    "3": "สาม",
    "4": "สี่",
    "5": "ห้า",
    "6": "หก",
    "7": "เจ็ด",
    "8": "แปด",
    "9": "เก้า",
}

# Thai numeral characters → Arabic digit mapping.
_THAI_NUMERAL_MAP: dict[str, str] = {
    "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
    "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9",
}

# Thai month names (1-indexed, index 0 is empty placeholder).
_THAI_MONTHS: list[str] = [
    "",
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน",
    "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม",
    "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]

# Common units and symbols → Thai spoken form.
_THAI_UNITS: list[tuple[str, str]] = [
    (r"°C\b", "องศาเซลเซียส"),
    (r"°F\b", "องศาฟาเรนไฮต์"),
    (r"km/h\b", "กิโลเมตรต่อชั่วโมง"),
    (r"m/s\b", "เมตรต่อวินาที"),
    (r"kg\b", "กิโลกรัม"),
    (r"km\b", "กิโลเมตร"),
    (r"cm\b", "เซนติเมตร"),
    (r"mm\b", "มิลลิเมตร"),
    (r"ml\b", "มิลลิลิตร"),
    (r"mg\b", "มิลลิกรัม"),
    (r"%", "เปอร์เซ็นต์"),
    (r"฿", "บาท"),
    (r"\$", "ดอลลาร์"),
    (r"€", "ยูโร"),
    (r"¥", "เยน"),
]


def _convert_thai_numerals(text: str) -> str:
    """Convert Thai numeral characters (๐-๙) to Arabic digits (0-9)."""
    for thai_digit, arabic_digit in _THAI_NUMERAL_MAP.items():
        text = text.replace(thai_digit, arabic_digit)
    return text


def _normalize_thai_text(text: str) -> str:
    """Normalize Thai text for TTS using a multi-stage pipeline.

    Pipeline stages (in order):
        1. Thai numerals — convert "๑๒๓" to "123" (so later stages can process)
        2. Character cleaning — fix floating vowels and duplicate tone marks
        3. Time normalization — convert "14:30 น." to "สิบสี่นาฬิกาสามสิบนาที"
        4. Date normalization — convert "1/1/2568" to "วันที่หนึ่งเดือนมกราคม..."
        5. Phone number handling — read digits one-by-one ("0812345678")
        6. Abbreviation expansion — convert "กทม." to "กรุงเทพมหานคร"
        7. Currency position — move symbol after number ("฿100" → "100 บาท")
        8. Unit/symbol expansion — convert "°C" to "องศาเซลเซียส"
        9. Mai Yamok (ๆ) — duplicate preceding word ("เด็กๆ" → "เด็กเด็ก")
        10. Long digit sequences — read 7+ digits one-by-one (ID cards, etc.)
        11. Number-to-word — convert remaining numbers to Thai words

    All pythainlp imports are lazy so the pipeline won't break
    on systems where pythainlp is not installed.

    Args:
        text: Input Thai text.

    Returns:
        Normalized text suitable for TTS training/inference.
    """
    try:
        from pythainlp.util import normalize as thai_normalize
        from pythainlp.util import num_to_thaiword
    except ImportError:
        # If pythainlp is not installed, return text as-is.
        return text

    # --- Stage 1: Thai numeral conversion ---
    # Convert ๐๑๒๓๔๕๖๗๘๙ → 0123456789 so all later stages work uniformly.
    text = _convert_thai_numerals(text)

    # --- Stage 2: Character cleaning ---
    # Fix floating vowels, duplicate tone marks, etc.
    text = thai_normalize(text)

    # --- Stage 3: Time normalization ---
    # Patterns like "14:30", "14.30 น.", "8:00" → Thai spoken form.
    def _replace_time(match: re.Match) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        try:
            hour_text = num_to_thaiword(hours) + "นาฬิกา"
            minute_text = num_to_thaiword(minutes) + "นาที" if minutes > 0 else ""
            return hour_text + minute_text
        except (ValueError, TypeError):
            return match.group()

    text = re.sub(r"(\d{1,2})[:\.](\d{2})(?:\s?น\.)?", _replace_time, text)

    # --- Stage 4: Date normalization ---
    # Patterns: "1/1/2568", "15/03/2568", "2025-03-06"
    def _replace_date(match: re.Match) -> str:
        try:
            day = int(match.group("day"))
            month = int(match.group("month"))
            year = int(match.group("year"))
            if 1 <= month <= 12:
                day_text = "วันที่" + num_to_thaiword(day)
                month_text = "เดือน" + _THAI_MONTHS[month]
                # Determine era: years > 2400 are likely พ.ศ., otherwise ค.ศ.
                if year > 2400:
                    year_text = "พุทธศักราช" + num_to_thaiword(year)
                else:
                    year_text = "คริสต์ศักราช" + num_to_thaiword(year)
                return day_text + month_text + year_text
        except (ValueError, TypeError, IndexError):
            pass
        return match.group()

    # Match DD/MM/YYYY or DD-MM-YYYY (Thai date format: day first).
    text = re.sub(
        r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})",
        _replace_date, text,
    )
    # Match YYYY-MM-DD (ISO format).
    text = re.sub(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})",
        _replace_date, text,
    )

    # --- Stage 5: Phone number handling ---
    # Thai phone numbers (mobile: 0x-xxxx-xxxx, landline: 0x-xxx-xxxx).
    def _replace_phone(match: re.Match) -> str:
        digits = match.group().replace("-", "").replace(" ", "")
        return "".join(_THAI_DIGIT_NAMES.get(d, d) for d in digits)

    text = re.sub(
        r"0\d[\s-]?\d{3,4}[\s-]?\d{4}", _replace_phone, text
    )

    # --- Stage 6: Abbreviation expansion ---
    for abbr_pattern, full_word in _THAI_ABBREVIATIONS.items():
        text = re.sub(abbr_pattern, full_word, text)

    # --- Stage 7: Currency position normalization ---
    # Move currency symbols from before to after the number for Thai word order.
    # Supports commas in numbers, e.g., "฿1,500" → "1,500 บาท"
    text = re.sub(r"฿\s?([\d,]+(?:\.\d+)?)", r"\1 บาท", text)
    text = re.sub(r"\$\s?([\d,]+(?:\.\d+)?)", r"\1 ดอลลาร์", text)
    text = re.sub(r"€\s?([\d,]+(?:\.\d+)?)", r"\1 ยูโร", text)
    text = re.sub(r"¥\s?([\d,]+(?:\.\d+)?)", r"\1 เยน", text)

    # --- Stage 8: Unit/symbol expansion ---
    for unit_pattern, thai_word in _THAI_UNITS:
        text = re.sub(unit_pattern, thai_word, text)

    # --- Stage 9: Mai Yamok (ๆ) handling ---
    # "เด็กๆ" → "เด็กเด็ก", "คนๆ" → "คนคน"
    if "ๆ" in text:
        try:
            from pythainlp.tokenize import word_tokenize

            words = word_tokenize(text, engine="newmm")
            normalized_words = []
            for i, word in enumerate(words):
                if word.strip() == "ๆ" and i > 0:
                    # Find the last non-space word to duplicate.
                    prev_word = ""
                    for j in range(i - 1, -1, -1):
                        if words[j].strip():
                            prev_word = words[j].strip()
                            break
                    normalized_words.append(prev_word)
                else:
                    normalized_words.append(word)
            text = "".join(normalized_words)
        except ImportError:
            pass

    # --- Stage 10: Long digit sequence handling ---
    # Sequences of 7+ digits (ID cards, bank accounts, etc.) should be
    # read digit-by-digit, not as mathematical numbers.
    def _read_digits_one_by_one(match: re.Match) -> str:
        return "".join(_THAI_DIGIT_NAMES.get(d, d) for d in match.group())

    text = re.sub(r"\d{7,}", _read_digits_one_by_one, text)

    # --- Stage 11: Number-to-word conversion ---
    # Convert remaining standalone numbers (less than 7 digits) to Thai words.
    # Supports commas in numbers, e.g., "1,500" → "หนึ่งพันห้าร้อย"
    def _replace_number(match: re.Match) -> str:
        number_str = match.group().replace(",", "")
        try:
            if "." in number_str:
                return num_to_thaiword(float(number_str))
            return num_to_thaiword(int(number_str))
        except (ValueError, TypeError):
            return match.group()

    text = re.sub(r"-?[\d,]+\.?\d*", _replace_number, text)

    return text


class TextNormalizer(metaclass=abc.ABCMeta):
    """Text normalization class for normalizers to implement."""

    @abc.abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize"""
        raise NotImplementedError("|normalize| is not implemented.")

    @abc.abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        raise NotImplementedError("|get_supported_languages| is not implemented.")

    @abc.abstractmethod
    def normalize_with_language(self, text: str, language: str) -> str:
        """Normalize text with a specific language."""
        raise NotImplementedError("|normalize_with_language| is not implemented.")


class NoOpTextNormalizer(TextNormalizer):
    """No-op text normalizer."""

    def normalize(self, text: str) -> str:
        return text

    def get_supported_languages(self) -> list[str]:
        return []

    def normalize_with_language(self, text: str, language: str) -> str:
        return text


class NemoTextNormalizer(TextNormalizer):
    """Text normalizer for different languages using Nvidias NeMo text normalization
    library. Supports extensible language registration via _LINGUA_LANG_MAP."""

    def __init__(self):
        super().__init__()
        # All languages registered in _LINGUA_LANG_MAP are considered "supported".
        self._supported_languages = list(_LINGUA_LANG_MAP.keys())

        # Only create NeMo normalizers for languages that NeMo actually supports in this environment.
        # Use try-except to safely skip languages that cause NotImplementedError (like 'ja' in certain versions).
        self._normalize_text = {}
        for lang in _NEMO_SUPPORTED_LANGS:
            try:
                self._normalize_text[lang] = normalize.Normalizer(input_case="cased", lang=lang)
            except NotImplementedError:
                # Silently skip unsupported languages so the rest of the application can run.
                pass
        self.lang_detector = None

    def init_lang_detector(self):
        self.lang_detector = lingua.LanguageDetectorBuilder.from_languages(
            *_LINGUA_LANG_MAP.values()
        ).build()

    def convert_to_ascii(self, text: str) -> str:
        return unidecode.unidecode(text)

    def get_supported_languages(self) -> list[str]:
        return self._supported_languages

    def normalize(self, text: str) -> str:
        # detect language and normalize text
        try:
            # Only initialize the language detector if it's not already initialized
            # (dynamic language detection).
            if self.lang_detector is None:
                self.init_lang_detector()
            detected_lang = self.lang_detector.detect_language_of(text)

            # Look up the ISO language code from the detected lingua Language.
            lang_code = _LINGUA_LANG_REVERSE_MAP.get(detected_lang)
            if lang_code:
                return self.normalize_with_language(text, lang_code)
            else:
                # Language detected but not in our supported map — return as-is.
                return text
        except Exception:
            return text

    def normalize_with_language(self, text: str, language: str) -> str:
        # If the language isn't in our supported set, pass through.
        if language not in _LINGUA_LANG_MAP:
            return text

        # Apply ASCII conversion for languages that need it (e.g., English).
        if language in _ASCII_CONVERT_LANGS:
            text = self.convert_to_ascii(text)

        # Apply Thai-specific normalization (numbers → Thai words).
        if language in _THAI_NORMALIZE_LANGS:
            text = _normalize_thai_text(text)

        # Use NeMo normalizer if available for this language,
        # otherwise return text as-is (pass-through for unsupported NeMo languages).
        if language in self._normalize_text:
            try:
                text = self._normalize_text[language].normalize(text)
            except Exception:
                # return the unnormalized text if error
                return text

        return text


def create_text_normalizer(enable_text_normalization: bool) -> TextNormalizer:
    """Create text normalizer for NemoNormalizer."""
    if enable_text_normalization:
        return NemoTextNormalizer()
    else:
        return NoOpTextNormalizer()
