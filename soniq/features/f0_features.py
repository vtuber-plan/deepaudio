# coding=utf-8
"""
F0 (fundamental frequency) features for Soniq.
"""

import torch
from typing import Optional, Dict, Any, Tuple
import numpy as np


class F0Features:
    """
    Configuration and utilities for F0 (fundamental frequency) features.

    This class provides default configurations and utility functions
    for F0 extraction and processing.

    Example:
        ```python
        config = F0Features.get_default_config()
        f0 = F0Features.extract(audio, **config)
        ```
    """

    # Default F0 range for human voice (in Hz)
    F0_MIN = 50.0  # Lowest pitch for bass voices
    F0_MAX = 1000.0  # Highest pitch for soprano voices

    # Default configurations
    CONFIG_DEFAULT = {
        "f0_min": F0_MIN,
        "f0_max": F0_MAX,
        "sample_rate": 24000,
        "hop_length": 256,
        "method": "pyin",  # pyin, dio, crepe
    }

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default F0 configuration.

        Returns:
            Configuration dictionary.
        """
        return cls.CONFIG_DEFAULT.copy()

    @classmethod
    def f0_to_midi(cls, f0: torch.Tensor) -> torch.Tensor:
        """
        Convert F0 (Hz) to MIDI note numbers.

        Args:
            f0: F0 values in Hz.

        Returns:
            MIDI note numbers.
        """
        # MIDI note 69 = A4 = 440 Hz
        # formula: midi = 69 + 12 * log2(f0 / 440)
        f0_safe = f0.clone()
        f0_safe[f0_safe < 1] = 1  # Avoid log(0)
        midi = 69 + 12 * torch.log2(f0_safe / 440.0)
        return midi

    @classmethod
    def midi_to_f0(cls, midi: torch.Tensor) -> torch.Tensor:
        """
        Convert MIDI note numbers to F0 (Hz).

        Args:
            midi: MIDI note numbers.

        Returns:
            F0 values in Hz.
        """
        f0 = 440.0 * (2 ** ((midi - 69) / 12))
        return f0

    @classmethod
    def f0_to_semitones(
        cls,
        f0: torch.Tensor,
        f0_ref: float = 440.0,
    ) -> torch.Tensor:
        """
        Convert F0 to semitones relative to reference frequency.

        Args:
            f0: F0 values in Hz.
            f0_ref: Reference frequency (default: A4 = 440 Hz).

        Returns:
            Pitch in semitones relative to reference.
        """
        f0_safe = f0.clone()
        f0_safe[f0_safe < 1] = 1
        semitones = 12 * torch.log2(f0_safe / f0_ref)
        return semitones

    @classmethod
    def interpolate_unvoiced(
        cls,
        f0: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Interpolate unvoiced (zero) F0 values.

        Args:
            f0: F0 contour.
            threshold: Threshold below which values are considered unvoiced.

        Returns:
            Interpolated F0 contour.
        """
        f0_interp = f0.clone()
        unvoiced_mask = f0 <= threshold

        if unvoiced_mask.all():
            return f0_interp

        # Find voiced indices
        voiced_mask = ~unvoiced_mask
        voiced_indices = torch.where(voiced_mask)[0]

        if len(voiced_indices) < 2:
            return f0_interp

        # Linear interpolation
        f0_voiced = f0[voiced_mask]
        f0_interp[unvoiced_mask] = torch.interp(
            torch.where(unvoiced_mask)[0].float(),
            voiced_indices.float(),
            f0_voiced.float(),
        )

        return f0_interp

    @classmethod
    def smooth_f0(
        cls,
        f0: torch.Tensor,
        window_size: int = 5,
    ) -> torch.Tensor:
        """
        Smooth F0 contour using moving average.

        Args:
            f0: F0 contour.
            window_size: Window size for smoothing.

        Returns:
            Smoothed F0 contour.
        """
        if window_size < 2:
            return f0

        # Pad for same output length
        padding = window_size // 2
        f0_padded = torch.nn.functional.pad(f0, (padding, padding), mode="reflect")

        # Moving average
        kernel = torch.ones(window_size) / window_size
        f0_smooth = torch.nn.functional.conv1d(
            f0_padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()

        return f0_smooth

    @classmethod
    def compute_f0_statistics(
        cls,
        f0: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute F0 statistics.

        Args:
            f0: F0 contour.

        Returns:
            Dictionary of F0 statistics.
        """
        voiced_mask = f0 > 0
        f0_voiced = f0[voiced_mask]

        if len(f0_voiced) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "voiced_ratio": 0.0,
            }

        return {
            "mean": f0_voiced.mean().item(),
            "std": f0_voiced.std().item(),
            "min": f0_voiced.min().item(),
            "max": f0_voiced.max().item(),
            "median": f0_voiced.median().item(),
            "voiced_ratio": voiced_mask.sum().item() / len(f0),
        }

    @classmethod
    def normalize_f0(
        cls,
        f0: torch.Tensor,
        mean: float,
        std: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize F0 using mean and std.

        Args:
            f0: F0 contour.
            mean: Mean for normalization.
            std: Standard deviation for normalization.
            eps: Epsilon for numerical stability.

        Returns:
            Normalized F0 contour.
        """
        return (f0 - mean) / (std + eps)

    @classmethod
    def denormalize_f0(
        cls,
        f0_normalized: torch.Tensor,
        mean: float,
        std: float,
    ) -> torch.Tensor:
        """
        Denormalize F0 using mean and std.

        Args:
            f0_normalized: Normalized F0 contour.
            mean: Mean used for normalization.
            std: Standard deviation used for normalization.

        Returns:
            Denormalized F0 contour.
        """
        return f0_normalized * std + mean
