# coding=utf-8
"""
Snake activation functions for BigVGAN.

Snake activation is a periodic activation function that enables better
modeling of periodic signals like audio waveforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Snake(nn.Module):
    """
    Snake activation function: x + (1/a) * sin^2(a*x)

    This is a periodic activation function that can model periodic signals
    better than ReLU-based activations.

    Args:
        channels: Number of input channels.
        alpha_logscale: If True, the alpha parameter is learned in log scale.
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Activated tensor of same shape.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        else:
            alpha = alpha + 1e-8  # Avoid division by zero

        return x + (1.0 / alpha) * torch.sin(x * alpha) ** 2


class SnakeBeta(nn.Module):
    """
    SnakeBeta activation function: x + (1/b) * sin^2(a*x)

    This is an extended version of Snake with separate parameters for
    frequency (alpha) and magnitude (beta) scaling.

    Args:
        channels: Number of input channels.
        alpha_logscale: If True, parameters are learned in log scale.
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Activated tensor of same shape.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)

        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        else:
            alpha = alpha + 1e-8
            beta = beta + 1e-8

        return x + (1.0 / beta) * torch.sin(x * alpha) ** 2
