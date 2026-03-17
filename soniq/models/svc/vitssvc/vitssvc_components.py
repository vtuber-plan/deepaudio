# coding=utf-8
"""
VitsSVC components.

This module contains the core components for VitsSVC:
- ContentEncoder: Prior encoder for content features + F0
- ConditionEncoder: Multi-modal condition encoder
- Various decoder wrappers
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Optional, Tuple

# ============================================================================
# Attention Module (VITS-style)
# ============================================================================

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
        self.attn = MultiHeadAttention(channels, n_heads, p_dropout)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, filter_channels, kernel_size, p_dropout)
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


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
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
            # Handle both float (0/1) and boolean masks
            if attn_mask.dtype == torch.bool:
                mask = attn_mask.unsqueeze(1)
            else:
                mask = (attn_mask == 0).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

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
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if mask is not None:
            x = x * mask
        return x


class Encoder(nn.Module):
    """
    VITS-style Transformer Encoder.

    Args:
        hidden_channels: Hidden dimension
        filter_channels: FFN filter dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        kernel_size: Convolution kernel size
        p_dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================================
# VitsSVC Content Encoder (Prior Encoder)
# ============================================================================


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Create sequence mask."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def f0_to_coarse(f0: torch.Tensor, n_bins: int = 256, f0_min: float = 50.0, f0_max: float = 1100.0) -> torch.Tensor:
    """
    Convert continuous F0 to discrete bins.

    Args:
        f0: F0 values (batch, seq_len).
        n_bins: Number of bins.
        f0_min: Minimum F0.
        f0_max: Maximum F0.

    Returns:
        Discrete F0 indices (batch, seq_len).
    """
    # Log-scale binning
    f0_mel_min = 1127 * math.log(1 + f0_min / 700)
    f0_mel_max = 1127 * math.log(1 + f0_max / 700)

    # Convert to mel scale
    f0_mel = 1127 * torch.log(1 + f0 / 700)

    # Quantize
    f0_mel = torch.clamp(f0_mel, f0_mel_min, f0_mel_max)
    bins = torch.linspace(f0_mel_min, f0_mel_max, n_bins, device=f0.device)
    f0_coarse = torch.bucketize(f0_mel, bins) - 1
    f0_coarse = torch.clamp(f0_coarse, 0, n_bins - 1)

    return f0_coarse


# ============================================================================
# F0 Encoder (Melody Encoder)
# ============================================================================

class MelodyEncoder(nn.Module):
    """F0 (melody) encoder with optional UV embedding."""

    def __init__(
        self,
        n_bins: int = 256,
        output_dim: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        use_uv: bool = True,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.output_dim = output_dim
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_uv = use_uv

        self.f0_emb = nn.Embedding(n_bins, output_dim)
        if use_uv:
            self.uv_emb = nn.Embedding(2, output_dim)

    def forward(
        self,
        f0: torch.Tensor,
        uv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            f0: F0 values (batch, seq_len).
            uv: UV flags (batch, seq_len), 1 for voiced, 0 for unvoiced.

        Returns:
            F0 embedding (batch, seq_len, output_dim).
        """
        # Quantize F0
        f0_coarse = f0_to_coarse(f0, self.n_bins, self.f0_min, self.f0_max)

        # Embed
        f0_emb = self.f0_emb(f0_coarse)

        # Add UV embedding
        if self.use_uv and uv is not None:
            uv = uv.long()
            f0_emb = f0_emb + self.uv_emb(uv)

        return f0_emb


# ============================================================================
# Content Encoder (for ContentVec/Whisper/WeNet features)
# ============================================================================

class ContentFeatureEncoder(nn.Module):
    """Encoder for content features (ContentVec, Whisper, etc.)."""

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Content features (batch, seq_len, input_dim).

        Returns:
            Projected features (batch, seq_len, output_dim).
        """
        return self.proj(x)


# ============================================================================
# Speaker Encoder
# ============================================================================

class SpeakerEncoder(nn.Module):
    """Speaker ID encoder."""

    def __init__(
        self,
        n_speakers: int = 1,
        output_dim: int = 256,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_speakers, output_dim)

    def forward(self, speaker_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speaker_id: Speaker IDs (batch,) or (batch, 1).

        Returns:
            Speaker embedding (batch, 1, output_dim).
        """
        if speaker_id.dim() == 2:
            speaker_id = speaker_id.squeeze(1)
        return self.emb(speaker_id).unsqueeze(1)


# ============================================================================
# Condition Encoder
# ============================================================================

class ConditionEncoder(nn.Module):
    """
    Multi-modal condition encoder for VitsSVC.

    Combines content features, F0, and speaker information.
    """

    def __init__(
        self,
        # Content features
        use_contentvec: bool = True,
        contentvec_dim: int = 256,
        use_whisper: bool = False,
        whisper_dim: int = 1024,
        use_wenet: bool = False,
        wenet_dim: int = 512,
        # F0
        use_f0: bool = True,
        n_bins_f0: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        use_uv: bool = True,
        # Speaker
        use_spkid: bool = True,
        n_speakers: int = 1,
        # Output
        content_encoder_dim: int = 256,
        output_melody_dim: int = 256,
        output_singer_dim: int = 256,
        merge_mode: str = "add",
    ):
        super().__init__()
        self.use_contentvec = use_contentvec
        self.use_whisper = use_whisper
        self.use_wenet = use_wenet
        self.use_f0 = use_f0
        self.use_spkid = use_spkid
        self.merge_mode = merge_mode

        # Content feature encoders
        if use_contentvec:
            self.contentvec_encoder = ContentFeatureEncoder(
                contentvec_dim, content_encoder_dim
            )
        if use_whisper:
            self.whisper_encoder = ContentFeatureEncoder(
                whisper_dim, content_encoder_dim
            )
        if use_wenet:
            self.wenet_encoder = ContentFeatureEncoder(
                wenet_dim, content_encoder_dim
            )

        # F0 encoder
        if use_f0:
            self.melody_encoder = MelodyEncoder(
                n_bins=n_bins_f0,
                output_dim=output_melody_dim,
                f0_min=f0_min,
                f0_max=f0_max,
                use_uv=use_uv,
            )

        # Speaker encoder
        if use_spkid:
            self.speaker_encoder = SpeakerEncoder(
                n_speakers=n_speakers,
                output_dim=output_singer_dim,
            )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            data: Dictionary containing:
                - contentvec_feat: (batch, seq_len, dim) or whisper_feat/wenet_feat
                - frame_pitch: (batch, seq_len)
                - frame_uv: (batch, seq_len) optional
                - spk_id: (batch,) or (batch, 1)

        Returns:
            Condition embedding (batch, seq_len, output_dim).
        """
        outputs = []
        seq_len = None

        # Content features
        if self.use_contentvec and "contentvec_feat" in data:
            enc_out = self.contentvec_encoder(data["contentvec_feat"])
            outputs.append(enc_out)
            seq_len = enc_out.shape[1]

        if self.use_whisper and "whisper_feat" in data:
            enc_out = self.whisper_encoder(data["whisper_feat"])
            outputs.append(enc_out)
            seq_len = enc_out.shape[1]

        if self.use_wenet and "wenet_feat" in data:
            enc_out = self.wenet_encoder(data["wenet_feat"])
            outputs.append(enc_out)
            seq_len = enc_out.shape[1]

        # F0
        if self.use_f0 and "frame_pitch" in data:
            uv = data.get("frame_uv", None)
            f0_emb = self.melody_encoder(data["frame_pitch"], uv)
            outputs.append(f0_emb)
            if seq_len is None:
                seq_len = f0_emb.shape[1]

        # Speaker
        if self.use_spkid and "spk_id" in data:
            spk_emb = self.speaker_encoder(data["spk_id"])
            if seq_len is not None:
                spk_emb = spk_emb.expand(-1, seq_len, -1)
            outputs.append(spk_emb)

        # Merge
        if self.merge_mode == "concat":
            return torch.cat(outputs, dim=-1)
        elif self.merge_mode == "add":
            return torch.stack(outputs, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")


# ============================================================================
# VitsSVC Content Encoder (Prior Encoder)
# ============================================================================

class VitsSVCContentEncoder(nn.Module):
    """
    Prior encoder for VitsSVC.

    Processes condition features and outputs prior distribution.
    """

    def __init__(
        self,
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.enc = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, hidden_channels, seq_len).
            x_mask: Mask (batch, 1, seq_len).
            noise_scale: Noise scale for sampling.

        Returns:
            z: Sampled latent (batch, out_channels, seq_len).
            m: Mean (batch, out_channels, seq_len).
            logs: Log std (batch, out_channels, seq_len).
            x_mask: Same as input.
        """
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale) * x_mask

        return z, m, logs, x_mask


# ============================================================================
# Utility Functions
# ============================================================================

def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """Slice segments from tensor."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str