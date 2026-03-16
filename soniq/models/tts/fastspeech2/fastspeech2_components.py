# coding=utf-8
"""
FastSpeech2 modules.

This module contains the core components for FastSpeech2:
- Encoder
- Decoder
- Duration Predictor
- Pitch Predictor
- Energy Predictor
- Length Regulator
"""

import math
import torch
from torch import nn
from torch.nn import Conv1d, LayerNorm, Embedding
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, Any
from soniq.modules.transformer.encoder import TransformerEncoder, TransformerEncoderLayer
from soniq.utils.model_utils import get_padding


# ============================================================================
# Utility Functions
# ============================================================================

def get_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create mask from lengths.

    Args:
        lengths: Length tensor of shape (batch,).
        max_len: Maximum length.

    Returns:
        Boolean mask of shape (batch, max_len).
    """
    if max_len is None:
        max_len = lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def expand(values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    Expand values according to durations.

    Args:
        values: Values to expand of shape (batch, seq_len, channels).
        durations: Durations of shape (batch, seq_len).

    Returns:
        Expanded values of shape (batch, sum(durations), channels).
    """
    b, t, c = values.shape
    out = []
    for i in range(b):
        for j in range(t):
            out.extend([values[i, j]] * durations[i, j].item())
    return torch.stack(out).view(b, -1, c)


# ============================================================================
# FFT Block (FastSpeech Transformer Block)
# ============================================================================

class FFTBlock(nn.Module):
    """
    FastSpeech Transformer Block.

    Similar to TransformerEncoderLayer but with Conv1d-based FFN.

    Args:
        d_model: Dimension of input embeddings.
        n_heads: Number of attention heads.
        d_ff: Dimension of feed-forward hidden layer.
        kernel_size: Kernel size for convolutions in FFN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        d_ff: int = 1024,
        kernel_size: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(d_model)

        # Conv1d-based FFN
        self.conv_1 = Conv1d(d_model, d_ff, kernel_size, padding=kernel_size // 2)
        self.conv_2 = Conv1d(d_ff, d_model, kernel_size, padding=kernel_size // 2)
        self.norm2 = LayerNorm(d_model)
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
            mask: Attention mask (True for valid positions).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention with residual
        attn_output, _ = self.slf_attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN with residual
        residual = x
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv_2(F.gelu(self.conv_1(x)))
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.norm2(residual + self.dropout(x))

        return x


# ============================================================================
# Variance Predictor
# ============================================================================

class VariancePredictor(nn.Module):
    """
    Variance Predictor for duration, pitch, and energy.

    Args:
        in_channels: Input channels.
        filter_channels: Filter channels.
        kernel_size: Kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.conv_1 = Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = Conv1d(filter_channels, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, in_channels).
            mask: Optional mask.

        Returns:
            Predictions of shape (batch, seq_len, 1).
        """
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.norm_1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (batch, seq_len, 1)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


# ============================================================================
# Length Regulator
# ============================================================================

class LengthRegulator(nn.Module):
    """
    Length Regulator for expanding encoder output based on duration.

    Args:
        pad_value: Padding value for output.
    """

    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Encoder output of shape (batch, seq_len, channels).
            duration: Duration tensor of shape (batch, seq_len).
            max_len: Maximum output length.

        Returns:
            Tuple of (expanded output, output lengths).
        """
        expanded = expand(x, duration)
        lengths = duration.sum(dim=1)

        if max_len is not None:
            # Pad to max_len if needed
            b, t, c = expanded.shape
            if t < max_len:
                pad = torch.full((b, max_len - t, c), self.pad_value, device=x.device, dtype=x.dtype)
                expanded = torch.cat([expanded, pad], dim=1)

        return expanded, lengths


# ============================================================================
# Variance Embedding
# ============================================================================

class VarianceEmbedding(nn.Module):
    """
    Embedding for discretized pitch and energy values.

    Args:
        n_bins: Number of bins for discretization.
        channels: Embedding dimension.
    """

    def __init__(self, n_bins: int, channels: int):
        super().__init__()
        self.embedding = Embedding(n_bins, channels)

    def forward(
        self,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            variance: Discretized variance values of shape (batch, seq_len).

        Returns:
            Embedded variance of shape (batch, seq_len, channels).
        """
        return self.embedding(variance.long())


# ============================================================================
# Encoder
# ============================================================================

class FS2Encoder(nn.Module):
    """
    Encoder for FastSpeech2.

    Args:
        n_vocab: Vocabulary size.
        hidden_channels: Hidden dimension.
        n_layers: Number of layers.
        n_heads: Number of heads.
        filter_channels: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        n_layers: int,
        n_heads: int,
        filter_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_channels, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            FFTBlock(hidden_channels, n_heads, filter_channels, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len).
            x_lengths: Length tensor of shape (batch,).

        Returns:
            Tuple of (encoded output, mask).
        """
        x = self.emb(x)
        x = self.pos_enc(x)

        mask = get_mask_from_lengths(x_lengths)

        for layer in self.layers:
            x = layer(x, mask)

        return x, mask


# ============================================================================
# Decoder
# ============================================================================

class FS2Decoder(nn.Module):
    """
    Decoder for FastSpeech2.

    Args:
        hidden_channels: Hidden dimension.
        n_layers: Number of layers.
        n_heads: Number of heads.
        filter_channels: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_layers: int,
        n_heads: int,
        filter_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_channels, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            FFTBlock(hidden_channels, n_heads, filter_channels, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_channels).
            mask: Optional mask.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_channels).
        """
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Args:
        d_model: Dimension of input embeddings.
        dropout: Dropout rate.
        max_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
            Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# PostNet
# ============================================================================

class PostNet(nn.Module):
    """
    PostNet for refining mel spectrogram predictions.

    Args:
        n_mel: Number of mel bins.
        hidden_channels: Hidden channels.
        kernel_size: Kernel size for convolutions.
        n_layers: Number of convolutional layers.
    """

    def __init__(
        self,
        n_mel: int,
        hidden_channels: int = 512,
        kernel_size: int = 5,
        n_layers: int = 5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        in_ch = n_mel
        for i in range(n_layers):
            out_ch = hidden_channels if i < n_layers - 1 else n_mel
            conv = Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            norm = nn.BatchNorm1d(out_ch)
            self.convs.append(nn.Sequential(conv, norm))
            in_ch = out_ch

        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Mel spectrogram of shape (batch, n_mel, time).

        Returns:
            Refined mel spectrogram.
        """
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = self.dropout(F.relu(conv(x)))
            else:
                x = self.dropout(conv(x))
        return x
