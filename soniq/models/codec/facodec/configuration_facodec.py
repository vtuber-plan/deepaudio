# coding=utf-8
"""FACodec configuration."""

from soniq.models.base.configuration_base import SoniqModelConfig


class FACodecConfig(SoniqModelConfig):
    """
    Configuration class for FACodec.

    FACodec (Factorized Codec) is a neural audio codec that factorizes
    audio representation into prosody, content, timbre, and residual components.

    Args:
        in_dim: Input dimension of encoder output.
        n_filters: Base number of filters in encoder/decoder.
        hidden_dim: Hidden dimension for internal layers.
        n_p_codebooks: Number of prosody codebooks.
        n_c_codebooks: Number of content codebooks.
        n_t_codebooks: Number of timbre codebooks.
        n_r_codebooks: Number of residual codebooks.
        codebook_size: Size of each codebook.
        codebook_dim: Dimension of codebook embeddings.
        quantizer_dropout: Dropout rate for quantizer.
        causal: Whether to use causal convolutions.
        separate_prosody_encoder: Whether to use separate prosody encoder.
        timbre_norm: Whether to use timbre normalization.
        sample_rate: Audio sample rate.
    """

    model_type = "facodec"

    def __init__(
        self,
        in_dim: int = 1024,
        n_filters: int = 32,
        hidden_dim: int = 512,
        n_p_codebooks: int = 1,
        n_c_codebooks: int = 2,
        n_t_codebooks: int = 2,
        n_r_codebooks: int = 3,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        causal: bool = False,
        separate_prosody_encoder: bool = False,
        timbre_norm: bool = False,
        sample_rate: int = 24000,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.in_dim = in_dim
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.n_p_codebooks = n_p_codebooks
        self.n_c_codebooks = n_c_codebooks
        self.n_t_codebooks = n_t_codebooks
        self.n_r_codebooks = n_r_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.causal = causal
        self.separate_prosody_encoder = separate_prosody_encoder
        self.timbre_norm = timbre_norm
        self.sample_rate = sample_rate

        super().__init__(initializer_range=initializer_range, **kwargs)
