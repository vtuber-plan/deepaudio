# coding=utf-8
"""
Normalizing Flow modules for VITS.

Contains residual coupling layers and stochastic duration predictor.
"""

from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math

from .encoders import WN, LayerNorm


class Flip(nn.Module):
    """Flip layer for flow - reverses channel order."""

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, [1])
        log_det = torch.zeros(x.shape[0], device=x.device)
        return x, log_det


class ResidualCouplingLayer(nn.Module):
    """
    Residual coupling layer for normalizing flow.

    Args:
        channels: Number of channels
        hidden_channels: Hidden dimension
        kernel_size: Convolution kernel size
        dilation_rate: Dilation rate
        n_layers: Number of WaveNet layers
        p_dropout: Dropout rate
        gin_channels: Global conditioning channels
        mean_only: If True, only shift without scale
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or reverse pass.

        Args:
            x: Input tensor (B, C, T)
            x_mask: Mask (B, 1, T)
            g: Global conditioning
            reverse: If True, run in reverse direction

        Returns:
            Output tensor and log determinant
        """
        x0, x1 = x.chunk(2, dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = stats.chunk(2, dim=1)
            s = torch.exp(logs) * x_mask
        else:
            m = stats
            logs = torch.zeros_like(m)
            s = torch.ones_like(m)

        if not reverse:
            x1 = (m + x1 * s) * x_mask
            log_det = torch.sum(logs, dim=[1, 2])
        else:
            x1 = ((x1 - m) / s) * x_mask
            log_det = -torch.sum(logs, dim=[1, 2])

        x = torch.cat([x0, x1], dim=1)
        return x, log_det


class ResidualCouplingBlock(nn.Module):
    """
    Stack of residual coupling layers.

    Args:
        channels: Number of channels
        hidden_channels: Hidden dimension
        kernel_size: Convolution kernel size
        dilation_rate: Dilation rate
        n_layers: Number of WaveNet layers per coupling layer
        n_flows: Number of coupling layers
        gin_channels: Global conditioning channels
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, mean_only=False
            ))
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward or reverse pass through all flows.

        Args:
            x: Input tensor (B, C, T)
            x_mask: Mask (B, 1, T)
            g: Global conditioning
            reverse: If True, run in reverse direction

        Returns:
            Output tensor
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g, reverse=True)
        return x


# ============================================================================
# Stochastic Duration Predictor
# ============================================================================

class DDSConv(nn.Module):
    """
    Dilated Depth-Separable Convolution.

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        n_layers: Number of layers
        p_dropout: Dropout rate
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
                LayerNorm(channels),
                nn.ReLU(),
                nn.Dropout(p_dropout),
            ))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x * x_mask) * x_mask
        return x


class ConvFlow(nn.Module):
    """
    Convolutional flow using piecewise rational quadratic transform.

    Args:
        in_channels: Input channels
        filter_channels: Filter dimension
        kernel_size: Kernel size
        n_layers: Number of layers
        gin_channels: Global conditioning channels
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            gin_ch = gin_channels if i == 0 else 0
            self.convs.append(nn.Conv1d(
                in_channels if i == 0 else filter_channels,
                filter_channels,
                kernel_size,
                padding=kernel_size // 2,
            ))

        self.proj = nn.Conv1d(filter_channels, 2 * in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input (B, D, T)
            x_mask: Mask (B, 1, T)
            g: Global conditioning
            reverse: If True, reverse transform

        Returns:
            Output and log determinant
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h * x_mask)
            if i == 0 and g is not None:
                h = h + g
            h = F.relu(h) * x_mask

        h = self.proj(h) * x_mask

        # Simple affine transform for flow
        h0, h1 = h.chunk(2, dim=1)

        if not reverse:
            log_s = torch.tanh(h0)
            m = h1
            x = x * torch.exp(log_s) + m
            log_det = torch.sum(log_s * x_mask, dim=[1, 2])
        else:
            log_s = torch.tanh(h0)
            m = h1
            x = (x - m) * torch.exp(-log_s)
            log_det = -torch.sum(log_s * x_mask, dim=[1, 2])

        return x, log_det


class ElementwiseAffine(nn.Module):
    """Elementwise affine transform."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            x = (x + self.m) * torch.exp(self.logs) * x_mask
            log_det = torch.sum(self.logs) * x_mask.shape[2]
            return x, log_det
        else:
            x = x * torch.exp(-self.logs) - self.m
            log_det = -torch.sum(self.logs) * x_mask.shape[2]
            return x, log_det


class Log(nn.Module):
    """Log transform for flow."""

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            y = torch.log(torch.clamp(x, min=1e-5)) * x_mask
            log_det = torch.sum(-y, dim=[1, 2])
            return y, log_det
        else:
            x = torch.exp(x) * x_mask
            return x, torch.zeros(x.shape[0], device=x.device)


class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor using normalizing flows.

    Predicts duration distribution using invertible transformations.

    Args:
        in_channels: Input channels
        filter_channels: Filter dimension
        kernel_size: Kernel size
        p_dropout: Dropout rate
        n_flows: Number of flow layers
        gin_channels: Global conditioning channels
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))

        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, 3, p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))

        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, 3, p_dropout)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (B, D, T)
            x_mask: Mask (B, 1, T)
            w: Duration target (B, 1, T) for training
            g: Global conditioning
            reverse: If True, sample from prior
            noise_scale: Noise scale for sampling

        Returns:
            Duration prediction or loss
        """
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            # Training: compute negative log-likelihood
            assert w is not None
            logdet_tot_q = 0

            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn_like(w)
            z_q = e_q * x_mask

            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w), reverse=False)
                logdet_tot_q += logdet_q

            return torch.sum(logdet_tot_q) / torch.sum(x_mask)

        else:
            # Inference: sample from prior
            z = torch.randn(x.shape[0], 2, x.shape[2], device=x.device) * noise_scale

            for flow in reversed(self.post_flows):
                z, _ = flow(z, x_mask, g=x, reverse=True)

            # Take mean of the two channels as duration
            w = z.mean(dim=1, keepdim=True)
            w = torch.exp(w) * x_mask
            w = torch.clamp(w, min=1.0)

            return w


__all__ = [
    "Flip",
    "ResidualCouplingLayer",
    "ResidualCouplingBlock",
    "DDSConv",
    "ConvFlow",
    "ElementwiseAffine",
    "Log",
    "StochasticDurationPredictor",
]