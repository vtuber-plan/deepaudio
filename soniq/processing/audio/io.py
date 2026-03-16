# coding=utf-8
"""Audio I/O utilities."""

from typing import Optional
import torch
import torchaudio
import numpy as np
import soundfile as sf


def load_audio(path: str, sample_rate: Optional[int] = None,
               mono: bool = True) -> torch.Tensor:
    """Load audio from file using soundfile."""
    # Load audio with soundfile
    waveform, sr = sf.read(path, dtype='float32')
    waveform = torch.from_numpy(waveform)

    # Ensure correct shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)  # (T, C) -> (C, T)

    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    return waveform


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int,
               normalize: bool = True) -> None:
    """Save audio to file."""
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-8)

    # Convert to numpy
    waveform_np = waveform.cpu().numpy()
    if waveform_np.shape[0] == 1:
        waveform_np = waveform_np.T  # (1, T) -> (T, 1)

    sf.write(path, waveform_np, sample_rate)
