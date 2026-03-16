# coding=utf-8
"""
VALL-E components.

This module contains the core components for VALL-E:
- Token Embedding
- Sine Positional Embedding
- AR Decoder
- NAR Decoder
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Args:
        embedding_dim: Embedding dimension.
        vocab_size: Vocabulary size.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)
        if self.padding_idx is not None:
            self.emb.weight.data[self.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Token IDs of shape (batch, seq_len).

        Returns:
            Embedded tokens of shape (batch, seq_len, embedding_dim).
        """
        return self.emb(tokens) * math.sqrt(self.embedding_dim)


class SinePositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding.

    Args:
        embedding_dim: Embedding dimension.
        dropout: Dropout rate.
        scale: Whether to scale by sqrt(embedding_dim).
        alpha: Learnable scaling factor.
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        scale: bool = False,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.scale = scale
        self.alpha = nn.Parameter(torch.ones(1) * alpha) if scale else None

        # Register buffer for positional encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim).

        Returns:
            Position-encoded tensor of same shape.
        """
        b, seq_len, _ = x.shape

        # Create position sequence
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)

        # Compute positional encoding
        pe = torch.zeros(seq_len, self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * self.inv_freq)
        pe[:, 1::2] = torch.cos(position * self.inv_freq)

        # Apply scaling
        if self.scale:
            pe = pe * self.alpha

        # Add dropout
        x = x + pe.unsqueeze(0)
        return x


class ARDecoder(nn.Module):
    """
    Autoregressive decoder for VALL-E.

    Args:
        decoder_dim: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
        norm_first: Use pre-norm architecture.
    """

    def __init__(
        self,
        decoder_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.nhead = nhead
        self.num_layers = num_layers

        # Transformer decoder layer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((1000, 1000), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, decoder_dim).
            mask: Optional key padding mask.

        Returns:
            Output tensor of shape (batch, seq_len, decoder_dim).
        """
        seq_len = x.shape[1]
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        output = self.decoder(x, mask=causal_mask, src_key_padding_mask=mask)
        return output


class NARDecoder(nn.Module):
    """
    Non-Autoregressive decoder for VALL-E.

    Args:
        decoder_dim: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
        norm_first: Use pre-norm architecture.
    """

    def __init__(
        self,
        decoder_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.nhead = nhead
        self.num_layers = num_layers

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, decoder_dim).
            mask: Optional key padding mask.

        Returns:
            Output tensor of shape (batch, seq_len, decoder_dim).
        """
        output = self.decoder(x, src_key_padding_mask=mask)
        return output


class Prenet(nn.Module):
    """
    Prenet for VALL-E.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        hidden_features: Hidden feature dimension.
        num_layers: Number of conv layers.
        kernel_size: Conv kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 256,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        in_ch = in_features
        for i in range(num_layers):
            out_ch = hidden_features if i < num_layers - 1 else out_features
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, in_features).

        Returns:
            Output tensor of shape (batch, seq_len, out_features).
        """
        x = x.transpose(1, 2)  # (batch, in_features, seq_len)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # (batch, seq_len, out_features)
