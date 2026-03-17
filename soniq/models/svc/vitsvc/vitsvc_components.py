# coding=utf-8
"""VitsSVC model components."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from typing import Optional, Tuple
import math


class LayerNorm(nn.Module):
    """Layer normalization for audio."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Normalized output (batch, channels, seq_len).
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ResidualBlock(nn.Module):
    """Residual block for VitsSVC."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv1 = Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm1 = LayerNorm(channels)
        self.conv2 = Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=1,
        )
        self.norm2 = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Output tensor (batch, channels, seq_len).
        """
        residual = x
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        x = self.conv1(x)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class TransformerEncoder(nn.Module):
    """Transformer encoder for content features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = Conv1d(in_channels, hidden_channels, 1)
        self.residuals = nn.ModuleList([
            ResidualBlock(hidden_channels, kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_proj = Conv1d(hidden_channels, out_channels, 1)
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            Output tensor (batch, out_channels, seq_len).
        """
        x = self.in_proj(x)
        for layer in self.residuals:
            x = layer(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class ContentEncoder(nn.Module):
    """Content encoder for VitsSVC with flexible input channels."""

    def __init__(
        self,
        in_channels: int = 128,  # n_mel or content feature dim
        hidden_channels: int = 192,
        out_channels: int = 192,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = Conv1d(in_channels, hidden_channels, 1)
        self.residuals = nn.ModuleList([
            ResidualBlock(hidden_channels, kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_proj = Conv1d(hidden_channels, out_channels, 1)
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            Output tensor (batch, out_channels, seq_len).
        """
        x = self.in_proj(x)
        for layer in self.residuals:
            x = layer(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class FlowResidualBlock(nn.Module):
    """Residual block for flow-based decoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 9),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(channels, channels, kernel_size,
                       padding=(kernel_size * d - d) // 2,
                       dilation=d),
                LayerNorm(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for d in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Output tensor (batch, channels, seq_len).
        """
        for conv in self.convs:
            x = x + conv(x)
        return x


class FlowCouplingLayer(nn.Module):
    """Affine coupling layer for flow."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = FlowResidualBlock(channels, kernel_size, dropout=dropout)
        self.proj = Conv1d(channels, channels * 2, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).
            reverse: Whether to reverse the flow.

        Returns:
            Output tensor and logdet.
        """
        x0, x1 = x.chunk(2, dim=1)
        h = self.block(x0)
        h = self.proj(h)
        m, logs = h.chunk(2, dim=1)

        if reverse:
            x1 = (x1 - m) * torch.exp(-logs)
        else:
            x1 = x1 * torch.exp(logs) + m

        logdet = torch.sum(logs, dim=[1, 2]) if not reverse else -torch.sum(logs, dim=[1, 2])
        return torch.cat([x0, x1], dim=1), logdet


class ResidualCouplingLayer(nn.Module):
    """Residual coupling layer for flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 9),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.half_channels = channels // 2
        self.pre = Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = FlowResidualBlock(hidden_channels, kernel_size, dropout=dropout)
        self.proj = Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).
            reverse: Whether to reverse the flow.

        Returns:
            Output tensor and logdet.
        """
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0)
        h = self.enc(h)
        stats = self.proj(h)
        m, logs = stats.chunk(2, dim=1)

        if reverse:
            x1 = (x1 - m) * torch.exp(-logs)
        else:
            x1 = x1 * torch.exp(logs) + m

        logdet = torch.sum(logs, dim=[1, 2]) if not reverse else -torch.sum(logs, dim=[1, 2])
        return torch.cat([x0, x1], dim=1), logdet


class ResidualCouplingFlow(nn.Module):
    """Flow-based decoder with residual coupling layers."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.flows = nn.ModuleList([
            ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dropout=dropout
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len).
            reverse: Whether to reverse the flow.

        Returns:
            Output tensor (batch, channels, seq_len).
        """
        if reverse:
            for flow in reversed(self.flows):
                x, _ = flow(x, reverse=True)
        else:
            for flow in self.flows:
                x, _ = flow(x)
        return x


class PosteriorEncoder(nn.Module):
    """Posterior encoder for VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.enc = FlowResidualBlock(hidden_channels, kernel_size, dropout=dropout)
        self.proj_mean = Conv1d(hidden_channels, out_channels, 1)
        self.proj_logvar = Conv1d(hidden_channels, out_channels, 1)
        self.flow = ResidualCouplingFlow(out_channels, hidden_channels, kernel_size, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            z: Latent variable (batch, out_channels, seq_len).
            m: Mean (batch, out_channels, seq_len).
            logs: Log variance (batch, out_channels, seq_len).
        """
        x = self.pre(x)
        x = self.enc(x)
        m = self.proj_mean(x)
        logs = self.proj_logvar(x)
        logs = torch.clamp(logs, min=-10, max=10)

        # Reparameterization trick
        std = torch.exp(0.5 * logs)
        z = m + std * torch.randn_like(m)

        # Flow-based refinement
        z = self.flow(z)

        return z, m, logs


class PriorEncoder(nn.Module):
    """Prior encoder for VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.enc = FlowResidualBlock(hidden_channels, kernel_size, dropout=dropout)
        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)
        self.flow = ResidualCouplingFlow(out_channels, hidden_channels, kernel_size, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            z: Prior latent (batch, out_channels, seq_len).
            m: Mean (batch, out_channels, seq_len).
        """
        x = self.pre(x)
        x = self.enc(x)
        stats = self.proj(x)
        m, logs = stats.chunk(2, dim=1)
        logs = torch.clamp(logs, min=-10, max=10)

        # Sample from prior
        std = torch.exp(0.5 * logs)
        z = m + std * torch.randn_like(m)

        # Flow-based refinement
        z = self.flow(z)

        return z, m


class ResBlock(nn.Module):
    """Residual block with dilation for Generator."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv = Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm = LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.norm(x)
        x = self.conv(x)
        return x + residual


class Generator(nn.Module):
    """VITS-style generator with upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre = Conv1d(in_channels, hidden_channels, 1)

        # Upsampling layers
        self.upsampler = nn.ModuleList()
        for rate, kernel_size in zip(upsample_rates, upsample_kernel_sizes):
            self.upsampler.append(
                ConvTranspose1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2,
                )
            )

        # Residual blocks for each upsampling stage
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            stage_blocks = nn.ModuleList()
            for j, kernel_size in enumerate(resblock_kernel_sizes):
                dilations = resblock_dilations[j % len(resblock_dilations)]
                for d in dilations:
                    stage_blocks.append(ResBlock(hidden_channels, kernel_size, d))
            self.resblocks.append(stage_blocks)

        self.post = nn.Sequential(
            nn.ReLU(),
            Conv1d(hidden_channels, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent variable (batch, in_channels, seq_len).

        Returns:
            waveform: Output waveform (batch, 1, seq_len * hop_factor).
        """
        z = self.pre(z)

        for i, upsample in enumerate(self.upsampler):
            z = F.relu(z)
            z = upsample(z)
            # Apply residual blocks for this stage
            for resblock in self.resblocks[i]:
                z = resblock(z)

        return self.post(z)


class SpeakerEncoder(nn.Module):
    """Speaker encoder for VitsSVC."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 192,
        out_channels: int = 256,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            LayerNorm(hidden_channels),
            Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            LayerNorm(hidden_channels),
            Conv1d(hidden_channels, out_channels, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            Speaker embedding (batch, out_channels).
        """
        return self.layers(x).squeeze(-1)
