# coding=utf-8
"""Vocoder task module."""

from .system import VocoderTaskSystem, VocoderConfig
from .datasets import VocoderDataset
from .collators import VocoderCollator
from .vocos_system import VocosTaskSystem, VocosConfig

__all__ = [
    "VocoderTaskSystem",
    "VocoderConfig",
    "VocoderDataset",
    "VocoderCollator",
    "VocosTaskSystem",
    "VocosConfig",
]