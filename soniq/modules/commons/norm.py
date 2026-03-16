# coding=utf-8
"""
Normalization modules for Soniq.
"""

import torch
from torch import nn
from torch.nn.utils import weight_norm as torch_weight_norm


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Args:
        channels: Number of input channels.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Normalized tensor.
        """
        # Transpose for layer norm over channels
        x = x.transpose(1, -1)
        x = nn.functional.layer_norm(x, [self.channels], self.weight, self.bias, self.eps)
        return x.transpose(1, -1)


class WeightNorm:
    """
    Weight normalization utility.

    Usage:
        conv = Conv1d(...)
        conv = WeightNorm.apply(conv)

        # To remove:
        WeightNorm.remove(conv)
    """

    @staticmethod
    def apply(module, name="weight"):
        """
        Apply weight normalization to a module.

        Args:
            module: Module to apply weight normalization.
            name: Name of weight parameter.

        Returns:
            Module with weight normalization.
        """
        torch_weight_norm.apply(module, name=name)
        return module

    @staticmethod
    def remove(module):
        """
        Remove weight normalization from a module.

        Args:
            module: Module to remove weight normalization.
        """
        torch.nn.utils.remove_weight_norm(module)


class InstanceNorm1d(nn.InstanceNorm1d):
    """1D Instance Normalization with optional affine."""

    def __init__(self, channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(channels, affine=affine, eps=eps)


class InstanceNorm2d(nn.InstanceNorm2d):
    """2D Instance Normalization with optional affine."""

    def __init__(self, channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(channels, affine=affine, eps=eps)


class GroupNorm(nn.GroupNorm):
    """Group Normalization."""

    def __init__(
        self,
        num_groups: int,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__(num_groups, channels, eps=eps, affine=affine)
