# coding=utf-8
"""DiffSVC model configuration."""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional
import math


class DiffSVCConfig(PretrainedConfig):
    """
    Configuration class for the DiffSVC singing voice conversion model.

    DiffSVC uses diffusion models for high-quality singing voice conversion.

    Args:
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length for audio processing.
        n_mel: Number of mel filterbanks.
        n_fft: FFT size for spectrogram computation.
        hidden_dim: Hidden dimension size.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout probability.
        n_spks: Number of speakers (for multi-speaker training).
        spk_emb_dim: Speaker embedding dimension.
        diffusion_steps: Number of diffusion steps.
        beta_start: Start value for noise schedule.
        beta_end: End value for noise schedule.
        beta_schedule: Type of noise schedule ("linear", "cosine").

    Example:
        ```python
        config = DiffSVCConfig(
            sample_rate=44100,
            hop_length=512,
            n_mel=128,
            hidden_dim=512,
            n_heads=8,
            n_layers=6,
            diffusion_steps=1000,
        )
        ```
    """

    model_type = "diffsvc"

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_mel: int = 128,
        n_fft: int = 2048,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_spks: int = 1,
        spk_emb_dim: int = 512,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.diffusion_steps = diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        super().__init__(**kwargs)
