# coding=utf-8
"""
Model utilities for Soniq.
"""

import torch
from torch import nn
from typing import Union


def init_weights(module: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    """
    Initialize network weights.

    Args:
        module: Network module.
        init_type: Initialization type ("normal", "xavier", "kaiming", "orthogonal").
        init_gain: Scaling factor for initialization.
    """
    classname = module.__class__.__name__

    if hasattr(module, "weight") and (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
        or isinstance(module, nn.ConvTranspose1d)
        or isinstance(module, nn.ConvTranspose2d)
    ):
        if init_type == "normal":
            nn.init.normal_(module.weight.data, 0.0, init_gain)
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight.data, gain=init_gain)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(f"Initialization method {init_type} not implemented")

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, 0.0, init_gain)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias.data, 0.0)
        nn.init.constant_(module.weight.data, 1.0)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Calculate padding size for convolutional layers.

    Args:
        kernel_size: Kernel size.
        dilation: Dilation rate.

    Returns:
        Padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_1d(
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
) -> int:
    """
    Calculate padding for 1D convolution to maintain input length.

    Args:
        input_length: Input length.
        kernel_size: Kernel size.
        stride: Stride.
        dilation: Dilation rate.

    Returns:
        Total padding size.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    output_length = (input_length + 2 * get_padding(kernel_size, dilation) - effective_kernel_size) // stride + 1
    total_padding = max(0, (output_length - 1) * stride + effective_kernel_size - input_length)
    return total_padding


def slice_segments(
    x: torch.Tensor,
    start_indices: torch.Tensor,
    segment_length: int,
) -> torch.Tensor:
    """
    Slice segments from tensor.

    Args:
        x: Input tensor of shape (batch, channels, time).
        start_indices: Start indices for each batch.
        segment_length: Length of each segment.

    Returns:
        Sliced segments of shape (batch, channels, segment_length).
    """
    batch_size, channels, _ = x.shape
    output = x.new_zeros((batch_size, channels, segment_length))

    for i in range(batch_size):
        idx = start_indices[i]
        output[i] = x[i, :, idx : idx + segment_length]

    return output


def rand_slice_segments(
    x: torch.Tensor,
    segment_length: int,
) -> torch.Tensor:
    """
    Randomly slice segments from tensor.

    Args:
        x: Input tensor of shape (batch, channels, time).
        segment_length: Length of each segment.

    Returns:
        Sliced segments and start indices.
    """
    batch_size, channels, time_length = x.shape

    if time_length > segment_length:
        max_start_idx = time_length - segment_length
        start_indices = torch.randint(0, max_start_idx, (batch_size,), device=x.device)
        x = slice_segments(x, start_indices, segment_length)
    else:
        start_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)

    return x, start_indices


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate KL divergence between two Gaussian distributions.

    Args:
        m_p: Mean of distribution p.
        logs_p: Log standard deviation of distribution p.
        m_q: Mean of distribution q.
        logs_q: Log standard deviation of distribution q.
        mask: Mask tensor.

    Returns:
        KL divergence.
    """
    kl = logs_q - logs_p - 0.5
    kl += 0.5 * ((m_p - m_q) ** 2) * torch.exp(-2.0 * logs_q)
    kl += 0.5 * torch.exp(2.0 * (logs_p - logs_q))
    kl = kl * mask
    return kl


def sequence_mask(
    length: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Create sequence mask.

    Args:
        length: Sequence lengths.
        max_length: Maximum length.

    Returns:
        Boolean mask tensor.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(
    duration: torch.Tensor,
    length_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Generate path from duration.

    Args:
        duration: Duration tensor of shape (batch, time).
        length_mask: Length mask.

    Returns:
        Path tensor.
    """
    device = duration.device

    batch, t_max = duration.shape
    cum_duration = torch.cumsum(duration, -1)

    path = torch.zeros((batch, t_max, t_max), device=device)
    for b in range(batch):
        for t in range(t_max):
            start = cum_duration[b, t - 1] if t > 0 else 0
            end = cum_duration[b, t]
            path[b, t, int(start) : int(end)] = 1

    return path * length_mask.unsqueeze(1)
