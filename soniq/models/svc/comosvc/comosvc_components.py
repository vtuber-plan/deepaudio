# coding=utf-8
"""ComoSVC model components."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for ComoSVC.

    Conditions the normalization on speaker and pitch embeddings.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            cond: Condition tensor (batch, cond_dim).

        Returns:
            Normalized output (batch, seq_len, dim).
        """
        x_norm = self.norm(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.dim = dim

        # Adaptive layer normalization
        self.attn_norm = AdaLN(dim, cond_dim)
        self.ffn_norm = AdaLN(dim, cond_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            cond: Condition tensor (batch, cond_dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        # Self-attention
        x_norm = self.attn_norm(x, cond)
        if mask is not None:
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=~mask,
            )
        else:
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feedforward
        x = x + self.ffn(self.ffn_norm(x, cond))

        return x


class F0Encoder(nn.Module):
    """
    F0 (pitch) encoder for ComoSVC.

    Encodes F0 contour into pitch embeddings.
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: F0 contour (batch, seq_len, 1).

        Returns:
            Pitch embedding (batch, seq_len, out_dim).
        """
        x = self.in_proj(f0)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)


class ContentEncoder(nn.Module):
    """
    Content encoder for ComoSVC.

    Extracts content features from mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                cond_dim=cond_dim,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_dim).
            cond: Condition tensor (batch, cond_dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Content features (batch, seq_len, hidden_dim).
        """
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x, cond, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """
    Decoder for ComoSVC.

    Decodes content features to mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_dim).

        Returns:
            Output tensor (batch, seq_len, out_dim).
        """
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.out_proj(x)


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for ComoSVC.

    Extracts speaker embeddings from mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 512,
        out_dim: int = 512,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_dim).

        Returns:
            Speaker embedding (batch, out_dim).
        """
        return self.layers(x)
