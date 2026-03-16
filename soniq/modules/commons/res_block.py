# coding=utf-8
"""
Residual block modules for Soniq.
"""

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm
from typing import Tuple


class ResBlock1d(nn.Module):
    """
    1D Residual Block.

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation rates for residual connections.
        activation: Activation function to use.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation

        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                )
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for conv in self.convs:
            residual = x
            x = self.activation(x)
            if mask is not None:
                x = x * mask
            x = conv(x)
            x = self.activation(x)
            if mask is not None:
                x = x * mask
            x = conv(x)
            x = x + residual
        if mask is not None:
            x = x * mask
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            torch.nn.utils.remove_weight_norm(conv)


class ResBlock2d(nn.Module):
    """
    2D Residual Block.

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation rates for residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = weight_norm(
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation[0],
                padding=((kernel_size[0] - 1) * dilation[0]) // 2,
                padding_mode="reflect",
            )
        )

        self.conv2 = weight_norm(
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation[1],
                padding=((kernel_size[1] - 1) * dilation[1]) // 2,
                padding_mode="reflect",
            )
        )

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        return x

    def remove_weight_norm(self):
        torch.nn.utils.remove_weight_norm(self.conv1)
        torch.nn.utils.remove_weight_norm(self.conv2)
