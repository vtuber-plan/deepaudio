# coding=utf-8
"""VALL-E model for zero-shot TTS."""

from soniq.models.tts.valle.configuration_valle import VALLEConfig
from soniq.models.tts.valle.modeling_valle import VALLE

__all__ = [
    "VALLEConfig",
    "VALLE",
]
