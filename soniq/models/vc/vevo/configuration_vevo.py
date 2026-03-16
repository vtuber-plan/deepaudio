# coding=utf-8
"""Vevo VC model configuration."""

from soniq.models.base.configuration_base import SoniqModelConfig


class VevoConfig(SoniqModelConfig):
    """
    Configuration class for Vevo voice conversion model.

    Vevo is a voice conversion model that uses semantic tokens and flow matching
    for high-quality voice conversion.

    Args:
        n_mel: Number of mel bins.
        hidden_dim: Hidden dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        codebook_size: Size of the semantic codebook.
        codebook_dim: Dimension of semantic embeddings.
        sample_rate: Audio sample rate.
    """

    model_type = "vevo"

    def __init__(
        self,
        n_mel: int = 80,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        codebook_size: int = 1024,
        codebook_dim: int = 512,
        sample_rate: int = 24000,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.n_mel = n_mel
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.sample_rate = sample_rate

        super().__init__(initializer_range=initializer_range, **kwargs)
