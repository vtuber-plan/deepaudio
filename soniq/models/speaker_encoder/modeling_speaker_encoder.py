# coding=utf-8
"""
Speaker Encoder modules for Soniq.

Provides speaker embedding extraction for:
- Speaker verification
- Multi-speaker TTS
- Voice conversion
- Singing voice conversion

Includes:
- ECAPA-TDNN style encoder
- Style encoder for VC
- Speaker embedding utilities
"""

from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import functional as F
import math


# ============================================================================
# Basic Building Blocks
# ============================================================================

class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class Res2Conv1dBlock(nn.Module):
    """
    Res2Net-style convolution block with multi-scale processing.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        scale: Number of scale groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.width = in_channels // scale

        self.convs = nn.ModuleList()
        for i in range(scale - 1):
            self.convs.append(nn.Sequential(
                nn.Conv1d(
                    self.width, self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size * dilation - dilation) // 2,
                ),
                nn.BatchNorm1d(self.width),
                nn.ReLU(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spx = torch.chunk(x, self.scale, dim=1)
        sp_outs = []
        sp = spx[0]
        sp_outs.append(sp)

        for i, conv in enumerate(self.convs):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = sp + spx[i + 1]
            sp = conv(sp)
            sp_outs.append(sp)

        return torch.cat(sp_outs, dim=1)


class Conv1dBlock(nn.Module):
    """1D convolution block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "mish":
            self.act = nn.Mish()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ============================================================================
# ECAPA-TDNN Style Encoder
# ============================================================================

class ECAPA_TDNN_Block(nn.Module):
    """
    ECAPA-TDNN block with Res2Net, SE-Module, and residual connection.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        scale: Number of scale groups for Res2Net
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size=1)
        self.res2conv = Res2Conv1dBlock(out_channels, out_channels, kernel_size, dilation, scale)
        self.se = SEModule(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        x = self.conv1(x)
        x = self.res2conv(x)
        x = self.se(x)
        x = self.conv2(x)
        x = self.norm(x)

        return F.relu(x + residual)


class AttentiveStatsPool(nn.Module):
    """
    Attentive statistics pooling for speaker embedding.

    Computes weighted mean and standard deviation using attention.

    Args:
        in_channels: Input channels
        attention_channels: Hidden dimension for attention
    """

    def __init__(self, in_channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        w = self.attention(x)

        # Weighted mean
        mean = torch.sum(x * w, dim=2)

        # Weighted std
        std = torch.sqrt(torch.sum((x - mean.unsqueeze(2)) ** 2 * w, dim=2) + 1e-8)

        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN.

    A state-of-the-art speaker encoder architecture.

    Args:
        input_channels: Input feature dimension
        channels: Base channel dimension
        emb_size: Embedding output dimension
        kernel_size: Convolution kernel size
        dilation_rates: List of dilation rates for each block
        scale: Number of scale groups for Res2Net

    Reference:
        "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
        in TDNN Based Speaker Verification" (Desplanques et al., 2020)
    """

    def __init__(
        self,
        input_channels: int = 80,
        channels: int = 512,
        emb_size: int = 192,
        kernel_size: int = 3,
        dilation_rates: Tuple[int, ...] = (1, 2, 3, 4, 5),
        scale: int = 8,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.emb_size = emb_size

        # Input projection
        self.conv1 = Conv1dBlock(input_channels, channels, kernel_size=1)

        # ECAPA blocks with different dilations
        self.blocks = nn.ModuleList()
        for dilation in dilation_rates:
            self.blocks.append(ECAPA_TDNN_Block(
                channels, channels, kernel_size, dilation, scale
            ))

        # Attentive statistics pooling
        self.asp = AttentiveStatsPool(channels * len(dilation_rates), channels)

        # Output projection
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels * len(dilation_rates) * 2, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, emb_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract speaker embedding.

        Args:
            x: Input features (B, T, D) or (B, D, T)
            lengths: Optional lengths for masking

        Returns:
            Speaker embedding (B, emb_size)
        """
        # Ensure (B, D, T) format
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)

        # Input projection
        x = self.conv1(x)

        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(x.shape[2], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            x = x * mask

        # ECAPA blocks with multi-resolution aggregation
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)

        # Attentive statistics pooling
        x = self.asp(x)

        # Output projection
        x = x.unsqueeze(2)
        x = self.conv2(x)

        return x.squeeze(2)


# ============================================================================
# Style Encoder for Voice Conversion
# ============================================================================

class Mish(nn.Module):
    """Mish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
    """Conv1d with Gated Linear Unit activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels * 2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class StyleEncoder(nn.Module):
    """
    Style/Timbre encoder for voice conversion.

    Extracts speaker style embeddings from reference audio.

    Args:
        in_dim: Input mel dimension
        hidden_dim: Hidden dimension
        out_dim: Output embedding dimension
        kernel_size: Convolution kernel size
        n_layers: Number of Conv1dGLU layers
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_dim: int = 80,
        hidden_dim: int = 128,
        out_dim: int = 256,
        kernel_size: int = 5,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Spectral processing
        self.spectral = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            Mish(),
        )

        # Temporal processing
        self.temporal = nn.ModuleList([
            Conv1dGLU(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Self-attention
        self.slf_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract style embedding.

        Args:
            x: Mel spectrogram (B, T, D) or (B, D, T)
            mask: Optional mask (B, T)

        Returns:
            Style embedding (B, out_dim)
        """
        # Ensure (B, D, T) format
        if x.dim() == 3 and x.shape[2] == self.in_dim:
            x = x.transpose(1, 2)

        # Spectral processing
        x = self.spectral(x)

        # Temporal processing
        for layer in self.temporal:
            x = x + layer(x)

        # Self-attention
        x = x.transpose(1, 2)  # (B, T, D)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        attn_out, _ = self.slf_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)

        # Temporal average pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        # Output projection
        x = self.proj(x)

        return x


# ============================================================================
# Reference Encoder with Query Attention
# ============================================================================

class ReferenceEncoder(nn.Module):
    """
    Reference encoder with query-based attention for speaker representation.

    Args:
        in_dim: Input mel dimension
        hidden_dim: Hidden dimension
        out_dim: Output embedding dimension
        n_queries: Number of query embeddings
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
    """

    def __init__(
        self,
        in_dim: int = 80,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_queries: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_queries = n_queries

        # Input projection
        self.proj_in = nn.Linear(in_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Query embeddings
        self.query_embs = nn.Embedding(n_queries, hidden_dim)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract speaker embeddings with query attention.

        Args:
            x: Mel spectrogram (B, T, D)
            mask: Optional mask (B, T)

        Returns:
            spk_embs: Speaker embeddings (B, n_queries, out_dim)
            encoded: Encoded features (B, T, hidden_dim)
        """
        batch_size = x.shape[0]

        # Input projection
        x = self.proj_in(x)

        # Encode
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Query attention
        query = self.query_embs.weight.unsqueeze(0).expand(batch_size, -1, -1)
        spk_embs, _ = self.cross_attn(
            query, encoded, encoded,
            key_padding_mask=src_key_padding_mask,
        )

        # Output projection
        spk_embs = self.proj_out(spk_embs)

        return spk_embs, encoded


# ============================================================================
# Speaker ID Encoder
# ============================================================================

class SpeakerIDEncoder(nn.Module):
    """
    Simple lookup table encoder for speaker IDs.

    Args:
        n_speakers: Number of speakers
        emb_dim: Embedding dimension
    """

    def __init__(self, n_speakers: int, emb_dim: int = 256):
        super().__init__()
        self.n_speakers = n_speakers
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(n_speakers, emb_dim)
        nn.init.normal_(self.embedding.weight, 0.0, emb_dim ** -0.5)

    def forward(self, speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Get speaker embedding from ID.

        Args:
            speaker_ids: Speaker IDs (B,) or (B, 1)

        Returns:
            Speaker embeddings (B, emb_dim) or (B, 1, emb_dim)
        """
        if speaker_ids.dim() == 2:
            return self.embedding(speaker_ids)
        return self.embedding(speaker_ids)


__all__ = [
    # Blocks
    "SEModule",
    "Res2Conv1dBlock",
    "Conv1dBlock",
    "Mish",
    "Conv1dGLU",
    "AttentiveStatsPool",
    # Encoders
    "ECAPA_TDNN",
    "ECAPA_TDNN_Block",
    "StyleEncoder",
    "ReferenceEncoder",
    "SpeakerIDEncoder",
]