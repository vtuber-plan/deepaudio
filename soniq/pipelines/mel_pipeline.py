# coding=utf-8
"""
Mel Spectrogram Pipeline for Soniq.

This module provides the MelPipeline class for extracting mel spectrograms from audio.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Union
import numpy as np


class MelPipeline:
    """
    Mel spectrogram pipeline for extracting mel features from audio.

    This pipeline handles:
        - Mel spectrogram extraction
        - Normalization
        - Dynamic range compression

    Example:
        ```python
        pipeline = MelPipeline(
            sample_rate=24000,
            n_fft=1024,
            n_mel=80,
            hop_length=256,
            win_length=1024,
        )
        mel = pipeline(audio)
        ```
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        n_mel: int = 80,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalized: bool = False,
        log_scale: bool = True,
        dB_scale: bool = True,
        top_db: float = 80.0,
    ):
        """
        Initialize MelPipeline.

        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            n_mel: Number of mel filterbanks.
            hop_length: Hop length for STFT.
            win_length: Window length for STFT (default: n_fft).
            f_min: Minimum frequency for mel filters.
            f_max: Maximum frequency for mel filters.
            normalized: Whether to normalize STFT.
            log_scale: Whether to apply log scaling.
            dB_scale: Whether to convert to dB scale.
            top_db: Maximum dB for dynamic range compression.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mel = n_mel
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.normalized = normalized
        self.log_scale = log_scale
        self.dB_scale = dB_scale
        self.top_db = top_db

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            n_mels=n_mel,
            f_min=f_min,
            f_max=f_max,
            normalized=normalized,
        )

        # Amplitude to dB transform
        if dB_scale:
            self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
        else:
            self.amplitude_to_db = None

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio tensor of shape (channels, time).

        Returns:
            Mel spectrogram tensor of shape (n_mel, time).
        """
        # Ensure mono
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to dB scale if needed
        if self.dB_scale and self.amplitude_to_db is not None:
            mel_spec = self.amplitude_to_db(mel_spec)

        return mel_spec.squeeze(0) if audio.shape[0] == 1 else mel_spec
