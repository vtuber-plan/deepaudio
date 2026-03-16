# coding=utf-8
"""Vevo VC model components."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


class ConformerBlock(nn.Module):
    """Conformer block for Vevo."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # FFN 1
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )
        self.ffn1_norm = nn.LayerNorm(dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(dim)

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Dropout(dropout),
        )
        self.conv_norm = nn.LayerNorm(dim)

        # FFN 2
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )
        self.ffn2_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        # FFN 1
        x = x + self.ffn1(self.ffn1_norm(x))

        # Self-attention
        x_norm = self.attention_norm(x)
        if mask is not None:
            # Use key_padding_mask for batched padding (True means padding)
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=~mask,  # True means padding
            )
        else:
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Convolution
        x_conv = self.conv_norm(x).transpose(1, 2)
        x_conv = self.conv(x_conv).transpose(1, 2)
        x = x + x_conv

        # FFN 2
        x = x + self.ffn2(self.ffn2_norm(x))

        return x


class SemanticEncoder(nn.Module):
    """Semantic encoder for Vevo."""

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            ConformerBlock(dim=dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            codes: Semantic codes (batch, seq_len).
            mask: Padding mask (batch, seq_len).

        Returns:
            Encoded features (batch, seq_len, dim).
        """
        x = self.embedding(codes)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class FlowMatchingDecoder(nn.Module):
    """Flow matching decoder for Vevo."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 80,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(dim=hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_dim).
            mask: Padding mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, out_dim).
        """
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.out_proj(x)


class MelDecoder(nn.Module):
    """Mel spectrogram decoder for Vevo."""

    def __init__(
        self,
        dim: int = 512,
        out_dim: int = 80,
        n_layers: int = 4,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            *[nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ) for _ in range(n_layers - 1)],
            nn.Linear(dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, dim).

        Returns:
            Mel spectrogram (batch, out_dim).
        """
        return self.layers(x)
