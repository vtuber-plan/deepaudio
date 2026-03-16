# coding=utf-8
"""
SpeechTokenizer components.

This module contains the core components for SpeechTokenizer:
- SEANet Encoder/Decoder
- Residual Vector Quantizer (RVQ)
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple
import numpy as np


# ============================================================================
# SEANet Encoder/Decoder Components
# ============================================================================

class SEANetResnetBlock(nn.Module):
    """Residual block for SEANet."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilations: List[int] = None,
        activation: str = "ELU",
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.activation = getattr(nn, activation)() if activation != "Snake" else Snake(dim)

        conv_class = nn.Conv1d

        self.conv1 = conv_class(dim, dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.ModuleList()
        for dilation in dilations:
            self.conv2.append(
                conv_class(dim, dim, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
            )
        self.conv3 = conv_class(dim, dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, dim, seq_len).

        Returns:
            Output tensor with same shape as input.
        """
        x = self.activation(x)
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        for conv in self.conv2:
            x = x + conv(x)
        x = self.conv3(x)
        return x + residual


class Snake(nn.Module):
    """
    Snake activation function: x + (1/b) * sin^2(x * a)

    References:
        - Liu Ziyin et al. "Neural Networks Fail to Learn Periodic Functions and How to Fix It"
    """

    def __init__(self, dim: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * alpha)
        self.beta = nn.Parameter(torch.ones(dim) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        x = x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class SEANetEncoder(nn.Module):
    """
    SEANet Encoder for SpeechTokenizer.

    Encodes audio into latent representations with progressive downsampling.
    """

    def __init__(
        self,
        n_filters: int = 32,
        dimension: int = 512,
        ratios: List[int] = None,
        lstm_layers: int = 3,
        bidirectional: bool = False,
        dilation_base: int = 4,
        residual_kernel_size: int = 3,
        n_residual_layers: int = 3,
        activation: str = "ELU",
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 6, 5, 4]

        self.n_filters = n_filters
        self.dimension = dimension
        self.ratios = ratios
        self.lstm_layers = lstm_layers

        conv_class = nn.Conv1d

        # Initial convolution
        self.conv1 = nn.Sequential(
            conv_class(1, n_filters, 7, padding=3),
            getattr(nn, activation)() if activation != "Snake" else Snake(n_filters),
        )

        # Downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        for ratio in ratios:
            block = nn.Sequential(
                conv_class(n_filters, n_filters * 2, stride=ratio, kernel_size=ratio * 2, padding=ratio // 2),
                getattr(nn, activation)() if activation != "Snake" else Snake(n_filters * 2),
            )
            self.downsample_blocks.append(block)
            n_filters = n_filters * 2

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        dilations = [dilation_base ** i for i in range(n_residual_layers)]
        for _ in range(n_residual_layers):
            self.residual_blocks.append(
                SEANetResnetBlock(n_filters, residual_kernel_size, dilations, activation)
            )

        # Final projection
        self.final_proj = conv_class(n_filters, dimension, 1)

        # LSTM (optional)
        if lstm_layers > 0:
            self.lstm = nn.LSTM(
                dimension,
                dimension,
                lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            if bidirectional:
                self.lstm_proj = conv_class(dimension * 2, dimension, 1)
            else:
                self.lstm_proj = conv_class(dimension, dimension, 1)
        else:
            self.lstm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio (batch, 1, seq_len).

        Returns:
            Encoded latent representation (batch, dimension, seq_len // product(ratios)).
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initial conv
        x = self.conv1(x)

        # Downsampling
        for block in self.downsample_blocks:
            x = block(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Final projection
        x = self.final_proj(x)

        # LSTM (optional)
        if self.lstm is not None:
            x = x.transpose(1, 2)  # (batch, seq_len, dim)
            x, _ = self.lstm(x)
            x = x.transpose(1, 2)  # (batch, dim, seq_len)
            x = self.lstm_proj(x)

        return x


class SEANetDecoder(nn.Module):
    """
    SEANet Decoder for SpeechTokenizer.

    Decodes latent representations back to audio with progressive upsampling.
    """

    def __init__(
        self,
        n_filters: int = 32,
        dimension: int = 512,
        ratios: List[int] = None,
        lstm_layers: int = 0,
        bidirectional: bool = False,
        dilation_base: int = 4,
        residual_kernel_size: int = 3,
        n_residual_layers: int = 3,
        activation: str = "ELU",
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 6, 5, 4]

        self.n_filters = n_filters
        self.dimension = dimension
        self.ratios = ratios
        self.lstm_layers = lstm_layers

        conv_class = nn.Conv1d

        # Initial projection
        self.initial_proj = conv_class(dimension, n_filters * (2 ** len(ratios)), 1)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        dilations = [dilation_base ** i for i in range(n_residual_layers)]
        for _ in range(n_residual_layers):
            self.residual_blocks.append(
                SEANetResnetBlock(n_filters * (2 ** len(ratios)), residual_kernel_size, dilations, activation)
            )

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        filters = n_filters * (2 ** len(ratios))
        for ratio in reversed(ratios):
            block = nn.Sequential(
                conv_class(filters, filters // 2, stride=ratio, kernel_size=ratio * 2, padding=ratio // 2),
                nn.Upsample(scale_factor=ratio, mode="nearest"),
                conv_class(filters // 2, filters // 2, 3, padding=1),
                getattr(nn, activation)() if activation != "Snake" else Snake(filters // 2),
            )
            self.upsample_blocks.append(block)
            filters = filters // 2

        # Final convolution
        self.final_conv = nn.Sequential(
            conv_class(n_filters, n_filters // 2, 7, padding=3),
            getattr(nn, activation)() if activation != "Snake" else Snake(n_filters // 2),
            conv_class(n_filters // 2, 1, 7, padding=3),
        )

        # LSTM (optional)
        if lstm_layers > 0:
            self.lstm = nn.LSTM(
                dimension,
                dimension,
                lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.lstm_proj = conv_class(dimension, dimension, 1)
        else:
            self.lstm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent representation (batch, dimension, seq_len).

        Returns:
            Reconstructed audio (batch, 1, seq_len * product(ratios)).
        """
        # LSTM (optional)
        if self.lstm is not None:
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            x = x.transpose(1, 2)
            x = self.lstm_proj(x)

        # Initial projection
        x = self.initial_proj(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Upsampling
        for block in self.upsample_blocks:
            x = block(x)

        # Final convolution
        x = self.final_conv(x)

        return x


# ============================================================================
# Residual Vector Quantization
# ============================================================================

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) for SpeechTokenizer.

    Quantizes input into discrete codes using multiple codebooks.
    """

    def __init__(
        self,
        dimension: int = 512,
        n_q: int = 8,
        bins: int = 1024,
    ):
        super().__init__()
        self.dimension = dimension
        self.n_q = n_q
        self.bins = bins

        # Create codebooks
        self.codebooks = nn.ModuleList()
        for _ in range(n_q):
            codebook = VQCodebook(bins, dimension)
            self.codebooks.append(codebook)

    def forward(
        self,
        x: torch.Tensor,
        n_q: Optional[int] = None,
        layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, dimension, seq_len).
            n_q: Number of quantizers to use.
            layers: Layers to return quantized output from.

        Returns:
            quantized: Final quantized output.
            codes: Quantization codes for each layer.
            commit_loss: Commitment loss.
            quantized_list: Quantized outputs from selected layers.
        """
        if n_q is None:
            n_q = self.n_q
        if layers is None:
            layers = list(range(n_q))

        quantized_list = []
        codes = []
        commit_loss = torch.tensor(0.0, device=x.device)

        residual = x
        for i in range(n_q):
            # Get codebook
            codebook = self.codebooks[i]

            # Quantize
            codes_i, quantized_i, loss_i = codebook(residual)
            codes.append(codes_i)
            commit_loss = commit_loss + loss_i

            # Update residual
            residual = residual - quantized_i.detach()

            # Store quantized output for selected layers
            if i in layers:
                quantized_list.append(quantized_i)

        # Sum all quantized outputs
        quantized = sum(quantized_list)

        # Stack codes
        codes = torch.stack(codes, dim=0)  # (n_q, batch, seq_len)

        return quantized, codes, commit_loss, quantized_list

    def encode(
        self,
        x: torch.Tensor,
        n_q: Optional[int] = None,
        st: int = 0,
    ) -> torch.Tensor:
        """
        Encode input to codes.

        Args:
            x: Input tensor (batch, dimension, seq_len).
            n_q: Number of quantizers to use.
            st: Start quantizer index.

        Returns:
            codes: Quantization codes (n_q, batch, seq_len).
        """
        if n_q is None:
            n_q = self.n_q

        codes = []
        residual = x
        for i in range(st, n_q):
            codebook = self.codebooks[i]
            # Use codebook forward to get codes and quantized output
            codes_i, quantized_i, _ = codebook(residual)
            codes.append(codes_i)
            # Update residual
            residual = residual - quantized_i.detach()

        return torch.stack(codes, dim=0)

    def decode(
        self,
        codes: torch.Tensor,
        st: int = 0,
    ) -> torch.Tensor:
        """
        Decode codes to quantized output.

        Args:
            codes: Quantization codes (n_q, batch, seq_len).
            st: Start quantizer index.

        Returns:
            quantized: Reconstructed tensor (batch, dimension, seq_len).
        """
        quantized = 0
        for i in range(st, codes.shape[0]):
            codebook = self.codebooks[i]
            # codebook.embed returns (batch, seq_len, dim), need to transpose to (batch, dim, seq_len)
            quantized_i = codebook.embed(codes[i]).transpose(1, 2)
            quantized = quantized + quantized_i
        return quantized


class VQCodebook(nn.Module):
    """Vector Quantization Codebook."""

    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps

        # Initialize embeddings
        self.embed = nn.Embedding(num_codes, dim)
        self.embed.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

        # EMA tracking
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", torch.zeros(num_codes, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, dim, seq_len).

        Returns:
            codes: Quantization codes (batch, seq_len).
            quantized: Quantized output (batch, dim, seq_len).
            commit_loss: Commitment loss.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        # Flatten input
        x_flat = x.transpose(1, 2).contiguous().view(-1, self.dim)  # (batch * seq_len, dim)

        # Find nearest neighbors
        distances = (
            x_flat.pow(2).sum(1, keepdim=True)
            - 2 * x_flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )
        codes = torch.argmin(distances, dim=1)

        # Reshape codes back to (batch, seq_len)
        codes = codes.view(batch_size, seq_len)

        # Quantize
        quantized = self.embed(codes)
        quantized = quantized.view(x.shape[0], x.shape[2], self.dim).transpose(1, 2)

        # Commitment loss
        commit_loss = F.mse_loss(quantized.detach(), x)

        # EMA updates (training only)
        if self.training:
            # Flatten codes for EMA
            codes_flat = codes.view(-1)

            # One-hot encoding
            one_hot = F.one_hot(codes_flat, self.num_codes).float()

            # Update cluster sizes and embeddings
            self.cluster_size = self.cluster_size * self.decay + (1 - self.decay) * one_hot.sum(0)
            self.embed_avg = self.embed_avg * self.decay + (1 - self.decay) * (one_hot.t() @ x_flat)

            # Normalize
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            self.embed.weight.data = self.embed_avg / cluster_size.unsqueeze(1)

        return codes, quantized, commit_loss
