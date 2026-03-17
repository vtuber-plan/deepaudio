# coding=utf-8
"""DualCodec configuration."""

from typing import List, Union
from transformers import PretrainedConfig


class DualCodecConfig(PretrainedConfig):
    """
    Configuration for DualCodec model.

    DualCodec is a low frame-rate semantic-enhanced neural audio codec
    that combines semantic features (from W2V-BERT) with acoustic features.
    """

    model_type = "dualcodec"

    def __init__(
        self,
        # Encoder parameters
        encoder_dim: int = 64,
        encoder_rates: List[int] = None,
        latent_dim: int = None,
        # Decoder parameters
        decoder_dim: int = 1536,
        decoder_rates: List[int] = None,
        # Quantizer parameters
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        # Semantic parameters
        semantic_codebook_size: int = 16384,
        semantic_codebook_dim: int = 8,
        semantic_downsample_factor: int = 2,
        # ConvNeXt semantic encoder
        convnext_dim: int = 768,
        convnext_layers: int = 4,
        # W2V-BERT input
        semantic_input_dim: int = 1024,  # W2V-BERT output dim
        # Training
        quantizer_dropout: float = 0.0,
        decode_semantic_for_codec: bool = True,
        is_causal: bool = False,
        # Audio
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates or [2, 4, 5, 6]  # Default: 240 hop
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates or [6, 5, 4, 2]

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.semantic_codebook_size = semantic_codebook_size
        self.semantic_codebook_dim = semantic_codebook_dim
        self.semantic_downsample_factor = semantic_downsample_factor

        self.convnext_dim = convnext_dim
        self.convnext_layers = convnext_layers
        self.semantic_input_dim = semantic_input_dim

        self.quantizer_dropout = quantizer_dropout
        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.is_causal = is_causal
        self.sample_rate = sample_rate

    @property
    def hop_length(self) -> int:
        """Calculate hop length from encoder rates."""
        import numpy as np
        return int(np.prod(self.encoder_rates))