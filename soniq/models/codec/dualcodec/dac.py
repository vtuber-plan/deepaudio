# coding=utf-8
"""
DAC (Descript Audio Codec) encoder and decoder.

Based on: https://github.com/descriptinc/descript-audio-codec
"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dac_layers import Snake1d, WNConv1d, WNConvTranspose1d


def init_weights(m: nn.Module):
    """Initialize convolution weights."""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def pad_to_length(x: torch.Tensor, length: int, pad_value: float = 0) -> torch.Tensor:
    """Pad tensor to specified length."""
    current_length = x.shape[-1]
    if length > current_length:
        pad_amount = length - current_length
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        x_padded = x[..., :length]
    return x_padded


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions."""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Encoder block with downsampling."""

    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    DAC Encoder.

    Args:
        d_model: Initial model dimension.
        strides: Downsampling strides.
        d_latent: Latent dimension.
    """

    def __init__(
        self,
        d_model: int = 64,
        strides: List[int] = None,
        d_latent: int = 64,
    ):
        super().__init__()
        strides = strides or [2, 4, 8, 8]

        # First convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Encoder blocks
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling."""

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """
    DAC Decoder.

    Args:
        input_channel: Input channel dimension.
        channels: Base channel dimension.
        rates: Upsampling rates.
        d_out: Output channels.
    """

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        d_out: int = 1,
    ):
        super().__init__()

        # First convolution
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Upsampling blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Final layers
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DAC(nn.Module):
    """
    Descript Audio Codec (DAC) model.

    Args:
        encoder_dim: Initial encoder dimension.
        encoder_rates: Encoder downsampling rates.
        latent_dim: Latent dimension.
        decoder_dim: Decoder dimension.
        decoder_rates: Decoder upsampling rates.
        n_codebooks: Number of codebooks.
        codebook_size: Codebook size.
        codebook_dim: Codebook dimension.
        quantizer_dropout: Quantizer dropout rate.
        sample_rate: Audio sample rate.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = None,
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = None,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 24000,
    ):
        super().__init__()
        encoder_rates = encoder_rates or [2, 4, 5, 6]
        decoder_rates = decoder_rates or [6, 5, 4, 2]

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = int(np.prod(encoder_rates))

        # Encoder
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        # Quantizer - import here to avoid circular import
        from .quantize import ResidualVectorQuantize
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        # Decoder
        self.decoder = Decoder(latent_dim, decoder_dim, decoder_rates)

        self.apply(init_weights)

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int = None) -> torch.Tensor:
        """Preprocess audio data."""
        if sample_rate is None:
            sample_rate = self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        subtracted_latent: torch.Tensor = None,
    ):
        """
        Encode audio data.

        Args:
            audio_data: Audio tensor (B, 1, T).
            sample_rate: Sample rate.
            n_quantizers: Number of quantizers to use.
            subtracted_latent: Semantic latent to subtract.

        Returns:
            z: Quantized latent.
            codes: Codebook indices.
            latents: Projected latents.
            commitment_loss: Commitment loss.
            codebook_loss: Codebook loss.
            first_layer_quantized: First layer quantized output.
        """
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)

        if subtracted_latent is not None:
            assert abs(z.shape[-1] - subtracted_latent.shape[-1]) <= 2
            z = z[..., : subtracted_latent.shape[-1]] - subtracted_latent

        z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = \
            self.quantizer(z, n_quantizers, possibly_no_quantizer=False)

        if subtracted_latent is not None:
            z = z + subtracted_latent

        return z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized

    def decode_from_codes(
        self,
        acoustic_codes: torch.Tensor,
        semantic_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode from codes with semantic latent.

        Args:
            acoustic_codes: Acoustic codebook indices (B, N, T).
            semantic_latent: Semantic latent tensor.

        Returns:
            Decoded audio (B, 1, T').
        """
        z = 0.0
        if acoustic_codes is not None:
            z = self.quantizer.from_codes(acoustic_codes)[0]
        z = z + semantic_latent

        audio = self.decoder(z)
        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        subtracted_latent: torch.Tensor = None,
        bypass_quantize: bool = False,
        possibly_no_quantizer: bool = False,
    ):
        """
        Forward pass.

        Args:
            audio_data: Audio tensor (B, 1, T).
            sample_rate: Sample rate.
            n_quantizers: Number of quantizers.
            subtracted_latent: Semantic latent to subtract.
            bypass_quantize: Whether to bypass quantization.
            possibly_no_quantizer: Allow zero quantizers.

        Returns:
            Dictionary with outputs.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)

        if subtracted_latent is not None:
            assert (z.shape[-1] - subtracted_latent.shape[-1]) <= 2
            z = z[..., : subtracted_latent.shape[-1]] - subtracted_latent

        if bypass_quantize:
            codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
                None, None, 0.0, 0.0, None
            )
            z = 0.0
        else:
            z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = \
                self.quantizer(z, n_quantizers, possibly_no_quantizer=possibly_no_quantizer)

        if subtracted_latent is not None:
            z = z + subtracted_latent

        x = self.decoder(z)
        x = pad_to_length(x, length)

        return {
            "audio": x,
            "z": z,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "first_layer_quantized": first_layer_quantized,
        }