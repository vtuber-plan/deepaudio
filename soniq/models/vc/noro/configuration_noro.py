# coding=utf-8
"""Noro VC model configuration."""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional


class NoroConfig(PretrainedConfig):
    """
    Configuration class for the Noro voice conversion model.

    Noro uses neural audio codecs for discrete representation learning
    and flow-based decoding for high-quality voice conversion.

    Args:
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length for audio processing.
        n_mel: Number of mel filterbanks.
        hidden_dim: Hidden dimension size.
        codebook_size: Size of each codebook.
        n_codebooks: Number of codebooks for residual vector quantization.
        codebook_dim: Dimension of codebook embeddings.
        n_heads: Number of attention heads.
        n_layers: Number of transformer/conformer layers.
        expansion_factor: Expansion factor for feedforward layers.
        dropout: Dropout probability.
        speaker_dim: Dimension of speaker embedding.

    Example:
        ```python
        config = NoroConfig(
            sample_rate=24000,
            hop_length=256,
            n_mel=80,
            hidden_dim=512,
            codebook_size=1024,
            n_codebooks=8,
            codebook_dim=512,
            n_heads=8,
            n_layers=6,
        )
        ```
    """

    model_type = "noro"

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_mel: int = 80,
        hidden_dim: int = 512,
        codebook_size: int = 1024,
        n_codebooks: int = 8,
        codebook_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        speaker_dim: int = 512,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mel = n_mel
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.speaker_dim = speaker_dim
        super().__init__(**kwargs)
