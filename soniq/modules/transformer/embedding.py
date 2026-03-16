# coding=utf-8
"""
Embedding modules for Soniq.
"""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module.

    Adds positional information to input embeddings.

    Args:
        d_model: Dimension of input embeddings.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Positionally encoded tensor.
        """
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal position embedding.

    Similar to PositionalEncoding but returns embeddings separately.

    Args:
        d_model: Dimension of embeddings.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create embedding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.embedding = nn.Embedding(max_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. If integer tensor, used as positions.
               If float tensor, returns positional encoding for sequence length.

        Returns:
            Positional embeddings.
        """
        if x.dtype in [torch.int, torch.long]:
            return self.embedding(x)
        else:
            seq_len = x.shape[1] if x.dim() > 1 else 1
            return self.embedding.weight[:seq_len, :].unsqueeze(0)


class LearnableEmbedding(nn.Module):
    """
    Learnable position embedding.

    Args:
        d_model: Dimension of embeddings.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional embeddings added.
        """
        seq_len = x.shape[1]
        return x + self.embedding[:, :seq_len, :]
