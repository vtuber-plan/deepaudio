# coding=utf-8
"""
VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

Reference: "VITS: Conditional Variational Autoencoder with Adversarial Learning
for End-to-End Text-to-Speech" (Kim et al., 2021)
"""

import math
from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Layer normalization with optional channel-first format."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    """Convolution with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList()

        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == n_layers - 1 else hidden_channels
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size // 2),
                LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p_dropout),
            ))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1) if n_layers > 0 else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        if self.proj is not None:
            x = self.proj(x)
        if mask is not None:
            x = x * mask
        return x


# ============================================================================
# Attention Modules
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        # Reshape for multi-head attention
        b, d, t = q.shape
        q = q.view(b, self.n_heads, self.head_dim, t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.head_dim, t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.head_dim, t).transpose(2, 3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(b, d, t)
        out = self.conv_o(out)

        return out


class FFN(nn.Module):
    """Feed-forward network with convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if mask is not None:
            x = x * mask
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and FFN."""

    def __init__(
        self,
        channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, channels, n_heads, p_dropout)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, channels, filter_channels, kernel_size, p_dropout)
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, mask)
        x = self.dropout(x)
        x = residual + x

        if mask is not None:
            x = x * mask
        return x


class TextEncoder(nn.Module):
    """
    Text encoder for VITS.

    Encodes text tokens into latent representation with mean and variance.

    Args:
        n_vocab: Vocabulary size
        out_channels: Output channels (latent dimension)
        hidden_channels: Hidden dimension
        filter_channels: FFN filter dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        kernel_size: Convolution kernel size
        p_dropout: Dropout rate
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text tokens.

        Args:
            x: Text tokens (B, T)
            x_lengths: Text lengths (B,)

        Returns:
            m_p: Mean projection (B, D, T)
            logs_p: Log variance projection (B, D, T)
            x_mask: Attention mask (B, 1, T)
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # (B, D, T)

        # Create mask
        x_mask = torch.arange(x.shape[2], device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1).float()  # (B, 1, T)

        # Encode
        for layer in self.encoder:
            x = layer(x, x_mask)

        # Project to mean and log variance
        stats = self.proj(x) * x_mask
        m_p, logs_p = stats.chunk(2, dim=1)

        return m_p, logs_p, x_mask


# ============================================================================
# Posterior Encoder
# ============================================================================

class WN(torch.nn.Module):
    """
    WaveNet-style dilated convolution module.

    Used in posterior encoder and flow modules.

    Args:
        hidden_channels: Hidden dimension
        kernel_size: Convolution kernel size
        dilation_rate: Dilation rate for each layer
        n_layers: Number of layers
        gin_channels: Global conditioning channels
        p_dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        else:
            self.cond_layer = None

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(nn.Sequential(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding),
            ))

            # Last layer outputs hidden_channels, others output 2 * hidden_channels
            if i == n_layers - 1:
                self.res_skip_layers.append(nn.Conv1d(hidden_channels, hidden_channels, 1))
            else:
                self.res_skip_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional global conditioning."""
        output = torch.zeros_like(x)

        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x) * x_mask

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self._fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts) * x_mask

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def _fused_add_tanh_sigmoid_multiply(self, x, y, n_channels):
        """Fused activation: tanh(a) * sigmoid(b) where x = [a, b]."""
        x_tanh, x_sigmoid = x[:, :n_channels, :], x[:, n_channels:, :]
        y_tanh, y_sigmoid = y[:, :n_channels, :], y[:, n_channels:, :]
        return torch.tanh(x_tanh + y_tanh) * torch.sigmoid(x_sigmoid + y_sigmoid)


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder for VITS.

    Encodes mel spectrogram into latent representation z.

    Args:
        in_channels: Input channels (mel dimension)
        out_channels: Output channels (latent dimension)
        hidden_channels: Hidden dimension
        kernel_size: Convolution kernel size
        dilation_rate: Dilation rate
        n_layers: Number of WaveNet layers
        gin_channels: Global conditioning channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode mel spectrogram.

        Args:
            x: Mel spectrogram (B, D, T)
            x_lengths: Mel lengths (B,)
            g: Global conditioning (B, gin_channels, 1)

        Returns:
            z: Sampled latent (B, D, T)
            m: Mean (B, D, T)
            logs: Log variance (B, D, T)
            x_mask: Mask (B, 1, T)
        """
        x_mask = torch.arange(x.shape[2], device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1).float()

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)

        stats = self.proj(x) * x_mask
        m, logs = stats.chunk(2, dim=1)

        # Reparameterization trick
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask


__all__ = [
    "LayerNorm",
    "MultiHeadAttention",
    "FFN",
    "EncoderLayer",
    "TextEncoder",
    "WN",
    "PosteriorEncoder",
]