# coding=utf-8
"""ComoSVC model configuration."""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional


class ComoSVCConfig(PretrainedConfig):
    """
    Configuration class for the ComoSVC singing voice conversion model.

    ComoSVC uses content-rich acoustic features for high-quality
    singing voice conversion.

    Args:
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length for audio processing.
        n_mel: Number of mel filterbanks.
        n_fft: FFT size for spectrogram computation.
        hidden_dim: Hidden dimension size.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        expansion_factor: Expansion factor for feedforward layers.
        dropout: Dropout probability.
        n_codes: Codebook size for discrete representation.

    Example:
        ```python
        config = ComoSVCConfig(
            sample_rate=44100,
            hop_length=512,
            n_mel=128,
            hidden_dim=512,
            n_heads=8,
            n_layers=6,
        )
        ```
    """

    model_type = "comosvc"

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_mel: int = 128,
        n_fft: int = 2048,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        n_codes: int = 1024,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.n_codes = n_codes
        super().__init__(**kwargs)
