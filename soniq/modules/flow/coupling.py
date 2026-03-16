# coding=utf-8
"""
Flow-based modules for Soniq.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
from ..transformer.encoder import TransformerEncoderLayer


class CouplingLayer(nn.Module):
    """
    Base coupling layer for normalizing flows.
    """

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class AffineCouplingLayer(CouplingLayer):
    """
    Affine coupling layer.

    Splits input in half and applies affine transformation
    to one half based on the other half.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Hidden layer channels.
        mask: Mask for splitting input.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(mask)
        self.in_channels = in_channels
        self.half_channels = in_channels // 2

        # Transformation network
        self.nn = nn.Sequential(
            nn.Conv1d(self.half_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, self.half_channels * 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).
            reverse: Whether to run in reverse mode.

        Returns:
            Tuple of (output tensor, log determinant).
        """
        batch_size = x.shape[0]

        if self.mask is not None:
            x = x * self.mask + (1 - self.mask) * x
        else:
            # Default: split in half
            x1, x2 = x[:, : self.half_channels], x[:, self.half_channels :]

        if not reverse:
            # Forward: transform x2 based on x1
            if self.mask is None:
                x1, x2 = x[:, : self.half_channels], x[:, self.half_channels :]

            h = self.nn(x1)
            shift, scale = h[:, : self.half_channels], h[:, self.half_channels :]
            scale = torch.sigmoid(scale + 2)

            y2 = x2 * scale + shift
            log_det = torch.sum(torch.log(scale), dim=[1, 2])

            if self.mask is None:
                y = torch.cat([x1, y2], dim=1)
            else:
                y = x * (1 - self.mask) + torch.cat([x1, y2], dim=1) * self.mask

            return y, log_det
        else:
            # Reverse: recover x2 from y2
            if self.mask is None:
                y1, y2 = x[:, : self.half_channels], x[:, self.half_channels :]
            else:
                y1 = x * self.mask

            h = self.nn(y1)
            shift, scale = h[:, : self.half_channels], h[:, self.half_channels :]
            scale = torch.sigmoid(scale + 2)

            x2 = (y2 - shift) / scale

            if self.mask is None:
                x_recovered = torch.cat([y1, x2], dim=1)
            else:
                x_recovered = x * (1 - self.mask) + torch.cat([y1, x2], dim=1) * self.mask

            log_det = -torch.sum(torch.log(scale), dim=[1, 2])

            return x_recovered, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model.

    Args:
        in_channels: Number of input channels.
        n_layers: Number of flow layers.
        hidden_channels: Hidden layer channels.
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 4,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers

        # Create alternating coupling layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(1, in_channels, 1)
            mask[:, i % 2::2, :] = 1
            self.layers.append(AffineCouplingLayer(in_channels, hidden_channels, mask))

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor.
            reverse: Whether to run in reverse mode.

        Returns:
            Tuple of (output tensor, total log determinant).
        """
        log_det_total = 0

        if reverse:
            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse=True)
                log_det_total += log_det
        else:
            for layer in self.layers:
                x, log_det = layer(x, reverse=False)
                log_det_total += log_det

        return x, log_det_total

    def compute_log_likelihood(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log likelihood under the flow.

        Assumes base distribution is standard normal.

        Args:
            x: Input tensor.

        Returns:
            Log likelihood.
        """
        z, log_det = self(x, reverse=False)

        # Log likelihood under standard normal
        log_prob_z = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi)))
        log_prob_z = torch.sum(log_prob_z, dim=[1, 2])

        return log_prob_z + log_det


class FlowSequence(nn.Module):
    """
    Sequence of flow transformations.

    Args:
        in_channels: Number of input channels.
        flow_config: Configuration for each flow.
    """

    def __init__(
        self,
        in_channels: int,
        flow_config: list = None,
    ):
        super().__init__()
        self.in_channels = in_channels

        if flow_config is None:
            flow_config = [
                {"type": "affine", "hidden_channels": 256, "n_layers": 4},
            ]

        self.flows = nn.ModuleList()
        for config in flow_config:
            if config["type"] == "affine":
                self.flows.append(
                    NormalizingFlow(
                        in_channels,
                        config.get("n_layers", 4),
                        config.get("hidden_channels", 256),
                    )
                )

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = 0

        if reverse:
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_total += log_det
        else:
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_total += log_det

        return x, log_det_total
