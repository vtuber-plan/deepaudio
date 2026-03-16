# coding=utf-8
"""
Anti-aliasing modules for BigVGAN.

These modules provide learnable upsampling and downsampling with
anti-aliasing filters to prevent aliasing artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from math import ceil


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
    sample_rate: float = 1.0,
) -> torch.Tensor:
    """
    Create a Kaiser-windowed sinc filter for anti-aliasing.

    Args:
        cutoff: Cutoff frequency as fraction of sample rate.
        half_width: Transition width as fraction of sample rate.
        kernel_size: Size of the filter kernel.
        sample_rate: Sample rate.

    Returns:
        1D filter tensor.
    """
    # Compute ideal sinc filter
    t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) / sample_rate
    sinc = torch.sinc(2 * cutoff * t)

    # Compute Kaiser window
    beta = 3.2331  # For 40dB attenuation
    alpha = (kernel_size - 1) / 2.0
    window = torch.zeros_like(t)
    for i in range(len(t)):
        idx = abs(i - alpha) / alpha
        if idx <= 1:
            window[i] = torch.i0(beta * torch.sqrt(torch.tensor(1.0 - idx ** 2)))
    window = window / torch.i0(torch.tensor(beta))

    return sinc * window


class UpSample1d(nn.Module):
    """
    Learnable anti-aliased upsampling.

    This module upsamples the input signal using a learnable filter
    with anti-aliasing to prevent imaging artifacts.

    Args:
        ratio: Upsampling ratio.
        kernel_size: Size of the filter kernel (default: auto-computed).
    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size

        # Create filter
        filter_coef = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter_coef)

        # Padding for causal filtering
        self.pad = self.kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Upsampled tensor of shape (batch, channels, time * ratio).
        """
        c = x.shape[1]
        in_len = x.shape[-1]

        # Pad input
        x = F.pad(x, (self.pad, self.pad), mode="replicate")

        # Apply filter with upsampling
        filter_coef = self.filter.unsqueeze(0).expand(c, -1, -1)
        x = F.conv_transpose1d(x, filter_coef, stride=self.ratio, groups=c)

        # Calculate output padding to ensure exact upsampling
        out_len = in_len * self.ratio
        extra = x.shape[-1] - out_len
        if extra > 0:
            x = x[..., :out_len]
        elif extra < 0:
            x = F.pad(x, (0, -extra))

        return x


class LowPassFilter1d(nn.Module):
    """
    Low-pass filter for anti-aliasing.

    Args:
        cutoff: Cutoff frequency as fraction of sample rate.
        stride: Downsampling stride.
        kernel_size: Size of the filter kernel.
    """

    def __init__(
        self,
        cutoff: float,
        stride: int = 1,
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.stride = stride

        if kernel_size is None:
            kernel_size = int(6 / cutoff)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

        self.kernel_size = kernel_size
        self.pad = self.kernel_size // 2

        # Create filter
        filter_coef = kaiser_sinc_filter1d(
            cutoff=cutoff,
            half_width=0.6 * cutoff,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter_coef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Filtered tensor.
        """
        c = x.shape[1]
        in_len = x.shape[-1]

        # Pad input
        x = F.pad(x, (self.pad, self.pad), mode="replicate")

        # Apply filter
        filter_coef = self.filter.unsqueeze(0).expand(c, -1, -1)
        x = F.conv1d(x, filter_coef, stride=self.stride, groups=c)

        # Calculate output padding to ensure exact downsampling
        out_len = (in_len + 2 * self.pad - self.kernel_size) // self.stride + 1
        extra = x.shape[-1] - out_len
        if extra > 0:
            x = x[..., :out_len]
        elif extra < 0:
            x = F.pad(x, (0, -extra))

        return x


class DownSample1d(nn.Module):
    """
    Anti-aliased downsampling.

    This module combines low-pass filtering with strided convolution
    to downsample the input signal without aliasing.

    Args:
        ratio: Downsampling ratio.
        kernel_size: Size of the filter kernel.
    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, stride=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Downsampled tensor of shape (batch, channels, time // ratio).
        """
        return self.lowpass(x)


class Activation1d(nn.Module):
    """
    1D activation with anti-aliasing.

    This module wraps an activation function with upsampling and downsampling
    to apply the activation at a higher resolution, preventing aliasing.

    Args:
        activation: The activation function to apply.
        up_ratio: Upsampling ratio for anti-aliasing.
        down_ratio: Downsampling ratio for anti-aliasing.
        up_kernel_size: Kernel size for upsampling filter.
        down_kernel_size: Kernel size for downsampling filter.
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: Optional[int] = None,
        down_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.activation = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Activated tensor of same shape.
        """
        x = self.upsample(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x
