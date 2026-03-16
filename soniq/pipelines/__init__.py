# coding=utf-8
"""Pipelines module."""

from soniq.pipelines.base import BasePipeline
from soniq.pipelines.vocoder import VocoderPipeline

# Keep legacy imports for backward compatibility
from soniq.pipelines.audio_pipeline import AudioPipeline
from soniq.pipelines.mel_pipeline import MelPipeline

__all__ = [
    "BasePipeline",
    "VocoderPipeline",
    "AudioPipeline",
    "MelPipeline",
]
