# coding=utf-8
"""
Mel spectrogram features for Soniq.
"""

import torch
from typing import Optional, Dict, Any
import numpy as np


class MelFeatures:
    """
    Configuration and utilities for mel spectrogram features.

    This class provides default configurations and utility functions
    for mel spectrogram extraction.

    Example:
        ```python
        config = MelFeatures.get_default_config()
        mel_spec = MelFeatures.extract(audio, **config)
        ```
    """

    # Default configurations for common sample rates
    CONFIG_16K = {
        "sample_rate": 16000,
        "n_fft": 1024,
        "n_mel": 80,
        "hop_length": 256,
        "win_length": 1024,
        "f_min": 0.0,
        "f_max": 8000.0,
    }

    CONFIG_22K = {
        "sample_rate": 22050,
        "n_fft": 1024,
        "n_mel": 80,
        "hop_length": 256,
        "win_length": 1024,
        "f_min": 0.0,
        "f_max": 11025.0,
    }

    CONFIG_24K = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "n_mel": 80,
        "hop_length": 256,
        "win_length": 1024,
        "f_min": 0.0,
        "f_max": 12000.0,
    }

    CONFIG_44K = {
        "sample_rate": 44100,
        "n_fft": 2048,
        "n_mel": 128,
        "hop_length": 512,
        "win_length": 2048,
        "f_min": 0.0,
        "f_max": 22050.0,
    }

    CONFIG_48K = {
        "sample_rate": 48000,
        "n_fft": 2048,
        "n_mel": 128,
        "hop_length": 512,
        "win_length": 2048,
        "f_min": 0.0,
        "f_max": 24000.0,
    }

    @classmethod
    def get_default_config(cls, sample_rate: int = 24000) -> Dict[str, Any]:
        """
        Get default configuration for a given sample rate.

        Args:
            sample_rate: Target sample rate.

        Returns:
            Configuration dictionary.
        """
        config_map = {
            16000: cls.CONFIG_16K,
            22050: cls.CONFIG_22K,
            24000: cls.CONFIG_24K,
            44100: cls.CONFIG_44K,
            48000: cls.CONFIG_48K,
        }

        if sample_rate not in config_map:
            # Return closest match or default
            return cls.CONFIG_24K

        return config_map[sample_rate]

    @classmethod
    def compute_num_frames(
        cls,
        audio_length: int,
        hop_length: int,
        win_length: int,
    ) -> int:
        """
        Compute the number of mel frames for given audio length.

        Args:
            audio_length: Length of audio in samples.
            hop_length: Hop length.
            win_length: Window length.

        Returns:
            Number of mel frames.
        """
        return (audio_length - win_length) // hop_length + 1

    @classmethod
    def compute_audio_length(
        cls,
        num_frames: int,
        hop_length: int,
        win_length: int,
    ) -> int:
        """
        Compute the audio length from number of mel frames.

        Args:
            num_frames: Number of mel frames.
            hop_length: Hop length.
            win_length: Window length.

        Returns:
            Audio length in samples.
        """
        return num_frames * hop_length + win_length - hop_length

    @classmethod
    def mel_to_audio_time(
        cls,
        mel_frame_idx: int,
        hop_length: int,
        win_length: int,
        sample_rate: int,
    ) -> float:
        """
        Convert mel frame index to audio time in seconds.

        Args:
            mel_frame_idx: Mel frame index.
            hop_length: Hop length.
            win_length: Window length.
            sample_rate: Sample rate.

        Returns:
            Time in seconds.
        """
        sample_idx = mel_frame_idx * hop_length
        return sample_idx / sample_rate

    @classmethod
    def audio_time_to_mel(
        cls,
        time_seconds: float,
        hop_length: int,
        sample_rate: int,
    ) -> int:
        """
        Convert audio time in seconds to mel frame index.

        Args:
            time_seconds: Time in seconds.
            hop_length: Hop length.
            sample_rate: Sample rate.

        Returns:
            Mel frame index.
        """
        sample_idx = int(time_seconds * sample_rate)
        return sample_idx // hop_length
