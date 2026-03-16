# coding=utf-8
"""
Audio utilities for Soniq.
"""

import torch
import torchaudio
from typing import Union, Optional
import numpy as np


def load_audio(
    path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio from file.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate (optional).
        mono: Whether to convert to mono.

    Returns:
        Audio tensor of shape (channels, time).
    """
    waveform, sr = torchaudio.load(path)

    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate is not None and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    return waveform


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
    format: str = "wav",
):
    """
    Save audio to file.

    Args:
        waveform: Audio tensor of shape (channels, time) or (time,).
        path: Path to save audio file.
        sample_rate: Sample rate of the audio.
        normalize: Whether to normalize the audio.
        format: Audio format (default: "wav").
    """
    # Ensure 2D tensor
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Normalize
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-8)

    torchaudio.save(path, waveform, sample_rate, format=format)


def normalize_audio(
    waveform: torch.Tensor,
    method: str = "peak",
    target_db: float = -3.0,
) -> torch.Tensor:
    """
    Normalize audio waveform.

    Args:
        waveform: Audio tensor.
        method: Normalization method ("peak" or "rms").
        target_db: Target level in dB.

    Returns:
        Normalized audio tensor.
    """
    if method == "peak":
        peak = waveform.abs().max()
        if peak > 0:
            target = 10 ** (target_db / 20)
            waveform = waveform * (target / peak)
    elif method == "rms":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            target = 10 ** (target_db / 20)
            waveform = waveform * (target / rms)

    return waveform


def trim_silence(
    waveform: torch.Tensor,
    top_db: float = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Trim silence from audio.

    Args:
        waveform: Audio tensor.
        top_db: Threshold in dB.
        frame_length: FFT frame length.
        hop_length: Hop length.

    Returns:
        Trimmed audio tensor.
    """
    # Compute RMS
    waveform_np = waveform.numpy()
    rms = torchaudio.functional.compute_dct(waveform).abs().mean(dim=1)

    # Find non-silent regions
    mask = rms > (10 ** (-top_db / 20))

    if not mask.any():
        return waveform

    # Find start and end
    indices = torch.where(mask)[0]
    start = indices[0] * hop_length
    end = (indices[-1] + 1) * hop_length

    return waveform[..., start:end]


def pad_audio(
    waveform: torch.Tensor,
    target_length: int,
    mode: str = "constant",
) -> torch.Tensor:
    """
    Pad audio to target length.

    Args:
        waveform: Audio tensor.
        target_length: Target length.
        mode: Padding mode.

    Returns:
        Padded audio tensor.
    """
    current_length = waveform.shape[-1]
    if current_length >= target_length:
        return waveform[..., :target_length]

    pad_length = target_length - current_length
    pad_left = pad_length // 2
    pad_right = pad_length - pad_left

    return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode=mode)
