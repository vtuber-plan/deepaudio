# coding=utf-8
"""
DualCodec: Low frame-rate semantic-enhanced neural audio codec.

DualCodec combines semantic features from W2V-BERT with acoustic features
using a dual-path quantization approach.

Reference: "DualCodec: A Low Frame-Rate Semantic-Enhanced Neural Audio Codec"
"""

import math
import random
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import PreTrainedModel
from .configuration_dualcodec import DualCodecConfig
from .dac import DAC
from .cnn import ConvNeXtBlock
from .quantize import ResidualVectorQuantize
from .dac_layers import WNConv1d


class DualCodec(PreTrainedModel):
    """
    DualCodec: Low frame-rate semantic-enhanced neural audio codec.

    This codec uses:
    - Semantic encoder (ConvNeXt) to process W2V-BERT features
    - Semantic VQ to quantize semantic information
    - DAC encoder to extract acoustic features
    - Semantic subtraction to separate acoustic from semantic
    - RVQ to quantize acoustic residuals

    Example:
        ```python
        config = DualCodecConfig()
        model = DualCodec(config)

        # Encode
        semantic_codes, acoustic_codes = model.encode(
            audio, semantic_repr=w2v_bert_features
        )

        # Decode
        audio = model.decode(semantic_codes, acoustic_codes)
        ```
    """

    config_class = DualCodecConfig
    base_model_prefix = "dualcodec"

    def __init__(self, config: DualCodecConfig):
        super().__init__(config)
        self.config = config

        self.semantic_downsample_factor = config.semantic_downsample_factor

        # DAC codec for acoustic encoding/decoding
        self.dac = DAC(
            encoder_dim=config.encoder_dim,
            encoder_rates=config.encoder_rates,
            latent_dim=config.latent_dim,
            decoder_dim=config.decoder_dim,
            decoder_rates=config.decoder_rates,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
        )

        self.decode_semantic_for_codec = config.decode_semantic_for_codec
        self.encoder_rates = config.encoder_rates

        # Semantic encoder (ConvNeXt)
        self.convnext_encoder = nn.Sequential(
            WNConv1d(config.semantic_input_dim, config.convnext_dim, kernel_size=1),
            *[
                ConvNeXtBlock(
                    dim=config.convnext_dim,
                    intermediate_dim=2048,
                    is_causal=config.is_causal,
                )
                for _ in range(config.convnext_layers)
            ],
        )

        # Semantic VQ
        self.semantic_vq = ResidualVectorQuantize(
            input_dim=config.convnext_dim,
            n_codebooks=1,
            codebook_size=config.semantic_codebook_size,
            codebook_dim=config.semantic_codebook_dim,
        )

        # Semantic decoder (ConvNeXt)
        self.convnext_decoder = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=config.convnext_dim,
                    intermediate_dim=2048,
                    is_causal=config.is_causal,
                )
                for _ in range(config.convnext_layers)
            ],
            WNConv1d(config.convnext_dim, config.semantic_input_dim, kernel_size=1),
        )

        if not self.decode_semantic_for_codec:
            assert config.convnext_dim == config.semantic_input_dim

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    @property
    def hop_length(self) -> int:
        """Get the hop length."""
        return self.dac.hop_length

    def semantic_quantize(
        self,
        semantic_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantize semantic representation.

        Args:
            semantic_repr: W2V-BERT features (B, D, T).

        Returns:
            Semantic codes (B, T).
        """
        semantic = self.convnext_encoder(semantic_repr)
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer = \
            self.semantic_vq(semantic)
        codes = rearrange(codes, "b 1 t -> b t")
        return codes

    @torch.no_grad()
    def encode(
        self,
        audio_data: torch.Tensor,
        semantic_repr: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio with semantic features.

        Args:
            audio_data: Audio tensor (B, 1, T).
            semantic_repr: W2V-BERT features (B, D, T).
            sample_rate: Sample rate.
            n_quantizers: Number of acoustic quantizers.

        Returns:
            semantic_codes: Semantic codes (B, T).
            acoustic_codes: Acoustic codes (B, N, T).
        """
        assert not self.training

        # Encode semantic
        semantic = self.convnext_encoder(semantic_repr)
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer = \
            self.semantic_vq(semantic)

        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        semantic_codes = codes

        if n_quantizers == 1:
            return semantic_codes, None

        if n_quantizers is not None:
            n_quantizers -= 1

        # Encode acoustic with semantic subtraction
        acoustic_codes = self.dac.encode(
            audio_data,
            sample_rate=sample_rate or self.config.sample_rate,
            n_quantizers=n_quantizers,
            subtracted_latent=semantic,
        )[1]

        return semantic_codes, acoustic_codes

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: torch.Tensor,
        acoustic_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode from codes.

        Args:
            semantic_codes: Semantic codes (B, T) or (B, 1, T).
            acoustic_codes: Acoustic codes (B, N, T).

        Returns:
            Decoded audio (B, 1, T').
        """
        # Handle semantic codes shape
        if semantic_codes.dim() == 2:
            semantic_codes = semantic_codes.unsqueeze(1)  # (B, 1, T)

        semantic = self.semantic_vq.from_codes(semantic_codes)[0]

        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        audio = self.dac.decode_from_codes(acoustic_codes, semantic)
        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        semantic_repr: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        bypass_quantize_rate: float = 0.125,
        possibly_no_quantizer: bool = False,
    ) -> Tuple[Dict, Dict]:
        """
        Forward pass for training.

        Args:
            audio_data: Audio tensor (B, 1, T).
            semantic_repr: W2V-BERT features (B, D, T).
            sample_rate: Sample rate.
            n_quantizers: Number of quantizers.
            bypass_quantize_rate: Rate to bypass quantization for training.
            possibly_no_quantizer: Allow zero quantizers.

        Returns:
            acoustic_output: Dictionary with acoustic outputs.
            semantic_output: Dictionary with semantic outputs.
        """
        # Encode semantic
        semantic = self.convnext_encoder(semantic_repr)
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer = \
            self.semantic_vq(semantic)

        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        # Determine bypass
        bypass_quantize = random.random() < bypass_quantize_rate
        if not self.training:
            bypass_quantize = False
        if n_quantizers == 1:
            bypass_quantize = True
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1

        # Encode acoustic
        acoustic_output = self.dac(
            audio_data,
            sample_rate or self.config.sample_rate,
            n_quantizers,
            subtracted_latent=semantic,
            bypass_quantize=bypass_quantize,
            possibly_no_quantizer=possibly_no_quantizer,
        )

        if not self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        semantic_output = {
            "x": semantic,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "bypassed_quantize": bypass_quantize,
        }

        return acoustic_output, semantic_output

    @torch.no_grad()
    def compress(
        self,
        audio_data: torch.Tensor,
        semantic_repr: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compress audio to discrete codes.

        Args:
            audio_data: Audio tensor (B, 1, T).
            semantic_repr: W2V-BERT features (B, D, T).
            sample_rate: Sample rate.
            n_quantizers: Number of acoustic quantizers.

        Returns:
            Dictionary with semantic_codes and acoustic_codes.
        """
        semantic_codes, acoustic_codes = self.encode(
            audio_data, semantic_repr, sample_rate, n_quantizers
        )
        return {
            "semantic_codes": semantic_codes,
            "acoustic_codes": acoustic_codes,
        }

    @torch.no_grad()
    def decompress(
        self,
        codes: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decompress codes to audio.

        Args:
            codes: Dictionary with semantic_codes and acoustic_codes.

        Returns:
            Decoded audio (B, 1, T).
        """
        return self.decode(codes["semantic_codes"], codes["acoustic_codes"])