# coding=utf-8
"""
CUDA utilities for Soniq.
"""

import os
import torch
from typing import Optional, List, Union, Dict


def set_cuda_device(
    device: Optional[Union[int, str, torch.device]] = None,
    allow_growth: bool = True,
    deterministic: bool = False,
    benchmark: bool = False,
) -> torch.device:
    """
    Set CUDA device and configure environment.

    Args:
        device: CUDA device ID or "cpu" or "cuda".
        allow_growth: Whether to allow GPU memory growth.
        deterministic: Whether to use deterministic algorithms.
        benchmark: Whether to use cudnn benchmark.

    Returns:
        Configured torch.device.
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        if device == "cpu":
            device = torch.device("cpu")
        elif device.startswith("cuda"):
            device = torch.device(device)
        else:
            device = torch.device(f"cuda:{device}")

    # Set environment variables
    if allow_growth and torch.cuda.is_available():
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Configure CUDA behavior
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = benchmark

    return device


def get_available_gpus() -> List[int]:
    """
    Get available GPU IDs.

    Returns:
        List of available GPU IDs.
    """
    if not torch.cuda.is_available():
        return []

    num_gpus = torch.cuda.device_count()
    return list(range(num_gpus))


def get_gpu_memory_info() -> List[Dict]:
    """
    Get GPU memory information.

    Returns:
        List of dicts with GPU memory info.
    """
    if not torch.cuda.is_available():
        return []

    info = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        info.append({
            "device_id": i,
            "device_name": torch.cuda.get_device_name(i),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
            "allocated_memory": torch.cuda.memory_allocated(i),
            "reserved_memory": torch.cuda.memory_reserved(i),
        })

    return info


def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_fp16_supported() -> bool:
    """
    Check if FP16 mixed precision is supported.

    Returns:
        True if FP16 is supported.
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    return major >= 7  # Volta and newer


def set_seed(seed: int = 42, cuda_deterministic: bool = True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
        cuda_deterministic: Whether to use deterministic CUDA operations.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
