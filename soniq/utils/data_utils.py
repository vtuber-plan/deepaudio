# coding=utf-8
"""
Data utilities for Soniq.
"""

import torch
from typing import List, Dict, Any
import numpy as np


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of tensors with varying lengths.
        batch_first: If True, output shape is (batch, ...).
        padding_value: Value to use for padding.

    Returns:
        Padded tensor.
    """
    if len(sequences) == 0:
        return torch.tensor([])

    # Get maximum length
    max_length = max(seq.shape[0] for seq in sequences)

    # Create padded tensor
    if sequences[0].dim() == 1:
        output = torch.full(
            (len(sequences), max_length),
            padding_value,
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )
    elif sequences[0].dim() == 2:
        max_dim1 = max(seq.shape[1] for seq in sequences)
        output = torch.full(
            (len(sequences), max_length, max_dim1),
            padding_value,
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )
    else:
        raise ValueError(f"Unsupported sequence dimension: {sequences[0].dim()}")

    # Fill in values
    for i, seq in enumerate(sequences):
        if seq.dim() == 1:
            output[i, : seq.shape[0]] = seq
        elif seq.dim() == 2:
            output[i, : seq.shape[0], : seq.shape[1]] = seq

    if not batch_first:
        output = output.transpose(0, 1)

    return output


def batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Move batch to device.

    Args:
        batch: Batch dictionary.
        device: Target device.

    Returns:
        Batch on device.
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def normalize_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize tensor along specified dimension.

    Args:
        tensor: Input tensor.
        dim: Dimension to normalize along.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True) + eps
    return (tensor - mean) / std


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Denormalize tensor.

    Args:
        tensor: Normalized tensor.
        mean: Original mean.
        std: Original standard deviation.
        eps: Epsilon for numerical stability.

    Returns:
        Denormalized tensor.
    """
    return tensor * (std + eps) + mean


def make_pad_mask(
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Create padding mask.

    Args:
        lengths: Sequence lengths.
        max_length: Maximum length.

    Returns:
        Boolean mask (True for padding).
    """
    batch_size = lengths.shape[0]
    seq_range = torch.arange(0, max_length, dtype=torch.long, device=lengths.device)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length)
    return seq_range >= lengths.unsqueeze(1)


def make_non_pad_mask(
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Create non-padding mask.

    Args:
        lengths: Sequence lengths.
        max_length: Maximum length.

    Returns:
        Boolean mask (True for non-padding).
    """
    return ~make_pad_mask(lengths, max_length)


def subsequent_mask(
    size: int,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """
    Create subsequent mask for decoder.

    Args:
        size: Sequence length.
        device: Device.

    Returns:
        Lower triangular mask.
    """
    mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0)
    return mask == 0


def get_length_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Get sequence lengths from mask.

    Args:
        mask: Boolean mask (True for valid positions).

    Returns:
        Sequence lengths.
    """
    return mask.sum(dim=-1)
