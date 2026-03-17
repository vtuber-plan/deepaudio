# coding=utf-8
"""
ConvNeXt blocks for DualCodec semantic encoder.

Adapted from https://github.com/facebookresearch/ConvNeXt for 1D audio.
"""

from typing import Optional
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block for 1D audio signals.

    Adapted from https://github.com/facebookresearch/ConvNeXt

    Args:
        dim: Number of input channels.
        intermediate_dim: Dimension of intermediate layer.
        layer_scale_init_value: Initial value for layer scale.
        is_causal: Whether to use causal convolution.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.is_causal = is_causal

        # Depthwise convolution
        if not is_causal:
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        else:
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=0, groups=dim)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise convolutions
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer scale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, T).

        Returns:
            Output tensor (B, C, T).
        """
        residual = x

        if self.is_causal:
            x = nn.functional.pad(x, (6, 0))

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        return residual + x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.

    Args:
        num_embeddings: Number of conditioning classes.
        embedding_dim: Dimension of embeddings.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim

        self.scale = nn.Embedding(num_embeddings, embedding_dim)
        self.shift = nn.Embedding(num_embeddings, embedding_dim)

        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.

        Args:
            x: Input tensor (B, T, C).
            cond_embedding_id: Conditioning indices (B,).

        Returns:
            Normalized and conditioned tensor.
        """
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x