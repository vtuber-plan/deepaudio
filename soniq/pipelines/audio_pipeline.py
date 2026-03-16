# coding=utf-8
"""
Audio Pipeline for Soniq.

This module provides the AudioPipeline class for loading and preprocessing audio files.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Union
import numpy as np


class AudioPipeline:
    """
    Audio pipeline for loading and preprocessing audio files.

    This pipeline handles:
        - Audio loading from file paths or tensors
        - Resampling to target sample rate
        - Normalization
        - Mono/Stereo conversion

    Example:
        ```python
        pipeline = AudioPipeline(sample_rate=24000)
        audio, sr = pipeline("path/to/audio.wav")
        ```
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        normalize: bool = True,
        mono: bool = True,
    ):
        """
        Initialize AudioPipeline.

        Args:
            sample_rate: Target sample rate for resampling.
            normalize: Whether to normalize audio to [-1, 1].
            mono: Whether to convert to mono.
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.mono = mono
        self.resampler = None

    def __call__(
        self,
        audio: Union[str, torch.Tensor],
        src_sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process audio input.

        Args:
            audio: Audio file path or audio tensor.
            src_sample_rate: Source sample rate (required if audio is a tensor).

        Returns:
            Processed audio tensor of shape (channels, time).
        """
        if isinstance(audio, str):
            # Load from file
            waveform, sr = torchaudio.load(audio)
        elif isinstance(audio, torch.Tensor):
            if src_sample_rate is None:
                raise ValueError("src_sample_rate must be provided when audio is a tensor")
            waveform = audio
            sr = src_sample_rate
        else:
            raise TypeError(f"Expected str or torch.Tensor, got {type(audio)}")

        # Convert to mono if needed
        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resampler(waveform)

        # Normalize if needed
        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform
