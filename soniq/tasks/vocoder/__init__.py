# coding=utf-8
"""Vocoder task module."""

from soniq.tasks.vocoder.system import VocoderTaskSystem, VocoderConfig
from soniq.tasks.vocoder.datasets import VocoderDataset
from soniq.tasks.vocoder.collators import VocoderCollator

__all__ = [
    "VocoderTaskSystem",
    "VocoderConfig",
    "VocoderDataset",
    "VocoderCollator",
]
