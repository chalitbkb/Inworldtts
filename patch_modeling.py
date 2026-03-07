import os
import torch
from tts.core import modeling

def _patched_load_model_from_checkpoint(*args, **kwargs):
    pass
