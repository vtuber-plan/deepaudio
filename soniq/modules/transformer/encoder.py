# coding=utf-8
"""
Transformer encoder modules for Soniq.
"""

import torch
from torch import nn
from typing import Optional
from .attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Args:
        d_model: Dimension of input embeddings.
        n_heads: Number of attention heads.
        d_ff: Dimension of feed-forward hidden layer.
        dropout: Dropout probability.
        norm_first: Whether to use pre-norm (True) or post-norm (False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        if self.norm_first:
            # Pre-norm
            attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.dropout(attn_output)
            x = x + self.feed_forward(self.norm2(x))
        else:
            # Post-norm
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            x = self.norm2(x + self.feed_forward(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.

    Args:
        d_model: Dimension of input embeddings.
        n_heads: Number of attention heads.
        d_ff: Dimension of feed-forward hidden layer.
        n_layers: Number of encoder layers.
        dropout: Dropout probability.
        norm_first: Whether to use pre-norm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, n_heads, d_ff, dropout, norm_first
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
