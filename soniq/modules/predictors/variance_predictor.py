# coding=utf-8
"""Variance predictors for TTS models: Duration, F0 (pitch), and Energy predictors."""

from typing import Dict, Optional, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F


class VariancePredictor(nn.Module):
    """
    Variance predictor for Duration, F0, and Energy.

    Based on FastSpeech2 architecture with convolutional layers.

    Args:
        input_size: Input feature dimension (default: 256 for encoder hidden)
        filter_size: Hidden layer dimension (default: 256)
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.5)
        output_dim: Output dimension (default: 1 for scalar prediction)
    """

    def __init__(
        self,
        input_size: int = 256,
        filter_size: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output_dim = output_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            input_size,
            filter_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            filter_size,
            filter_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(filter_size)
        self.norm2 = nn.LayerNorm(filter_size)

        # Output projection
        self.linear = nn.Linear(filter_size, output_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Activation
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict variance (duration, pitch, or energy).

        Args:
            x: Input features, shape (batch, seq_len, input_size)
            mask: Optional mask, shape (batch, seq_len, 1) or (batch, 1, seq_len)

        Returns:
            Predictions, shape (batch, seq_len, output_dim)
        """
        # Transpose for Conv1d: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.dropout_layer(x)

        # Apply mask if provided
        if mask is not None:
            x = x.masked_fill(mask.transpose(1, 2).bool(), 0.0)

        # Second conv block
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout_layer(x)

        # Apply mask if provided
        if mask is not None:
            x = x.masked_fill(mask.transpose(1, 2).bool(), 0.0)

        # Output projection
        x = self.linear(x)

        return x


class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor using normalizing flows.

    Provides probabilistic duration modeling for diverse generation.
    Based on VITS architecture.

    Args:
        in_channels: Input channels (encoder hidden size)
        filter_channels: Hidden dimension (default: 256)
        kernel_size: Conv kernel size (default: 3)
        p_dropout: Dropout probability (default: 0.5)
        n_flows: Number of flow steps (default: 4)
        gin_channels: Speaker embedding channels (default: 256)
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int = 256,
        kernel_size: int = 3,
        p_dropout: float = 0.5,
        n_flows: int = 4,
        gin_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.n_flows = n_flows

        # Speaker conditioning
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
        else:
            self.cond = None

        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, filter_channels, 1, 1)

        # Flow layers
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ConvFlow(filter_channels, filter_channels, kernel_size, n_layers=2)
            )

        # Post-processing
        self.conv_post = nn.Conv1d(filter_channels, 2 * filter_channels, 1, 1)

        # Output projection
        self.proj = nn.Conv1d(filter_channels, 2, 1)

        # Normalization
        self.norm = nn.LayerNorm(filter_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize final conv to near-zero for stable training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for duration prediction or sampling.

        Args:
            x: Input features (B, T, D)
            mask: Mask tensor (B, 1, T)
            g: Optional speaker embedding (B, gin_channels)
            reverse: If True, sample durations; if False, compute log prob

        Returns:
            If reverse=False: (log_duration_preds, log_prob)
            If reverse=True: (sampled_durations, None)
        """
        # Transpose to (B, D, T)
        x = x.transpose(1, 2)
        mask = mask.squeeze(1)  # (B, 1, T) -> (B, T)

        # Apply speaker conditioning
        if g is not None and self.cond is not None:
            g = g.unsqueeze(-1)  # (B, gin_channels, 1)
            x = x + self.cond(g)

        x = self.conv_pre(x)

        # Apply flows
        logdet_tot = 0
        for flow in self.flows:
            x, logdet = flow(x, mask, g=None, reverse=reverse)
            if not reverse:
                logdet_tot = logdet_tot + logdet

        # Post-processing
        x = self.conv_post(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        # Project to mean and log scale
        stats = self.proj(x * mask).transpose(1, 2)
        m, logs = stats.chunk(2, dim=-1)

        if reverse:
            # Sample from the flow
            z = torch.randn_like(m) * torch.exp(logs) + m
            duration = z * mask
            return duration.clamp(min=0), None
        else:
            # Compute negative log-likelihood
            # Target duration should be passed separately
            return m * mask, logs * mask, logdet_tot


class ConvFlow(nn.Module):
    """
    Convolutional flow layer for stochastic duration predictor.

    Implements affine coupling with convolutional transform.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        # Half channels for coupling
        self.half_channels = in_channels // 2

        # Initial conv
        self.conv_pre = nn.Conv1d(
            self.half_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )

        # Conv layers
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    filter_channels,
                    filter_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )

        # Output conv (scale and shift)
        self.conv_out = nn.Conv1d(filter_channels, self.half_channels * 2, 1)

        # Normalization
        self.norm = nn.LayerNorm(filter_channels)

        # Activation
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Forward pass for coupling layer.

        Args:
            x: Input (B, C, T)
            mask: Mask (B, T)
            g: Ignored (for compatibility)
            reverse: If True, invert the transform

        Returns:
            Transformed output and log determinant (if not reverse)
        """
        # Split input
        x0, x1 = x[:, : self.half_channels], x[:, self.half_channels :]

        # Transform x0 to get affine parameters
        h = self.conv_pre(x0 * mask)
        for conv in self.conv_layers:
            h = self.relu(h)
            h = conv(h)
            h = h.transpose(1, 2)
            h = self.norm(h)
            h = h.transpose(1, 2)

        h = self.conv_out(h)

        # Split into scale and shift
        m, logs = h.chunk(2, dim=1)
        logs = logs.clamp(-10, 10)  # Clamp for stability
        m = m * mask
        logs = logs * mask

        # Affine transform
        if reverse:
            # Inverse: (x - m) / exp(logs)
            x1_out = (x1 - m) * torch.exp(-logs)
            logdet = None
        else:
            # Forward: x * exp(logs) + m
            x1_out = x1 * torch.exp(logs) + m
            logdet = torch.sum(logs * mask, dim=[1, 2])

        x_out = torch.cat([x0, x1_out], dim=1)
        return x_out, logdet


class DurationEmbedding(nn.Module):
    """
    Duration embedding for phoneme-to-frame alignment.

    Converts phoneme-level features to frame-level using predicted durations.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand phoneme-level features to frame-level.

        Args:
            x: Phoneme features (B, T_phoneme, D)
            duration: Predicted durations (B, T_phoneme) or (B, T_phoneme, 1)
            mask: Optional phoneme mask

        Returns:
            Expanded features (B, T_frame, D) and frame mask
        """
        # Ensure duration is integer
        if duration.dim() == 2:
            duration = duration.unsqueeze(-1)

        duration = duration.clamp(min=0).long()

        # Expand features
        batch_size, seq_len, dim = x.shape
        expanded = []
        for b in range(batch_size):
            expanded_frames = []
            for t in range(seq_len):
                dur = duration[b, t, 0].item()
                if dur > 0:
                    expanded_frames.append(x[b, t : t + 1].repeat(dur, 1, 1))
            if expanded_frames:
                expanded.append(torch.cat(expanded_frames, dim=1))
            else:
                expanded.append(x[b : b + 1, :1])  # Keep at least one frame

        # Pad to same length
        max_len = max(e.shape[1] for e in expanded)
        output = torch.zeros(batch_size, max_len, dim, device=x.device, dtype=x.dtype)
        output_mask = torch.zeros(
            batch_size, max_len, 1, device=x.device, dtype=torch.bool
        )

        for b, e in enumerate(expanded):
            actual_len = e.shape[1]
            output[b, :actual_len] = e
            output_mask[b, :actual_len] = True

        return output, output_mask


class LengthRegulator(nn.Module):
    """
    Length regulator for duration-based expansion.

    Converts encoder outputs (phoneme-level) to decoder inputs (frame-level)
    based on predicted or target durations.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Regulate length based on duration predictions.

        Args:
            x: Encoder output (B, T_enc, D)
            duration: Duration predictions (B, T_enc)
            alpha: Speed adjustment factor (>1 for faster, <1 for slower)

        Returns:
            Expanded output (B, T_dec, D) and length tensor
        """
        # Apply speed adjustment
        if alpha != 1.0:
            duration = torch.ceil(duration * alpha).long()

        duration = duration.clamp(min=0)

        # Expand
        output = []
        out_lengths = []

        for b in range(x.shape[0]):
            expanded = []
            for t in range(x.shape[1]):
                dur = duration[b, t].item() if duration.dim() > 1 else duration[b].item()
                if dur > 0:
                    expanded.append(x[b, t : t + 1].repeat(dur, 1, 1))
            if expanded:
                expanded_cat = torch.cat(expanded, dim=0)
                output.append(expanded_cat)
                out_lengths.append(expanded_cat.shape[0])
            else:
                output.append(x[b : b + 1, :1])
                out_lengths.append(1)

        # Pad to max length
        max_len = max(out_lengths)
        output_padded = torch.zeros(
            x.shape[0], max_len, x.shape[2], device=x.device, dtype=x.dtype
        )
        length_tensor = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for b, (out, length) in enumerate(zip(output, out_lengths)):
            output_padded[b, :length] = out
            length_tensor[b] = length

        return output_padded, length_tensor


__all__ = [
    "VariancePredictor",
    "StochasticDurationPredictor",
    "ConvFlow",
    "DurationEmbedding",
    "LengthRegulator",
]
