# coding=utf-8
"""
DAC (Descript Audio Codec) layers.

Contains Snake activation and weight-normalized convolution layers.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs) -> nn.Conv1d:
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    """Weight-normalized 1D transposed convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Snake activation function.

    Args:
        x: Input tensor.
        alpha: Learnable parameter.

    Returns:
        Activated tensor.
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """
    Snake activation for 1D sequences.

    Introduced in "Neural Networks Fail to Learn Periodic Functions
    and How to Fix It" (Liu et al., 2020).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)