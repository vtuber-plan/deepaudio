# coding=utf-8
"""TTS models for Soniq."""

from soniq.models.tts.base import BaseTTSModel
from soniq.models.tts.vits import VITS, VITSConfig
from soniq.models.tts.fastspeech2 import FastSpeech2, FastSpeech2Config

__all__ = [
    "BaseTTSModel",
    "VITS",
    "VITSConfig",
    "FastSpeech2",
    "FastSpeech2Config",
]