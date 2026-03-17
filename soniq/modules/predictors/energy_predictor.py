# coding=utf-8
"""Energy extraction and prediction utilities."""

from typing import Dict, Optional, Union
import torch
from torch import nn
import numpy as np


def extract_energy_from_mel(
    mel_spectrogram: torch.Tensor,
    frame_level: bool = True,
) -> torch.Tensor:
    """
    Extract energy from mel spectrogram.

    Args:
        mel_spectrogram: Mel spectrogram (batch, n_mels, time) or (n_mels, time)
        frame_level: If True, return frame-level energy; else utterance-level

    Returns:
        Energy values (batch, time) or (batch,) if frame_level=False
    """
    # Compute energy as L2 norm of mel spectrum at each frame
    if mel_spectrogram.dim() == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)

    # Energy = sqrt(sum(mel^2))
    energy = torch.sqrt((mel_spectrogram.exp() ** 2).sum(dim=1))

    if not frame_level:
        energy = energy.mean(dim=-1)

    return energy


def extract_energy_from_waveform(
    waveform: torch.Tensor,
    hop_length: int,
    frame_level: bool = True,
) -> torch.Tensor:
    """
    Extract energy from waveform (RMS energy).

    Args:
        waveform: Audio waveform (batch, 1, time) or (batch, time)
        hop_length: Hop length in samples
        frame_level: If True, return frame-level energy

    Returns:
        Energy values
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)

    # Ensure mono
    if waveform.shape[1] > 1:
        waveform = waveform.mean(dim=1, keepdim=True)

    # Compute frame-level RMS
    batch_size, _, audio_len = waveform.shape
    n_frames = (audio_len // hop_length) + 1

    energy = []
    for b in range(batch_size):
        frame_energy = []
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + hop_length, audio_len)
            frame = waveform[b, 0, start:end]
            rms = torch.sqrt((frame ** 2).mean())
            frame_energy.append(rms)
        energy.append(torch.stack(frame_energy))

    return torch.stack(energy)


def normalize_energy(
    energy: torch.Tensor,
    mean: float = -4.92,
    std: float = 2.85,
) -> torch.Tensor:
    """
    Normalize energy values.

    Args:
        energy: Raw energy values
        mean: Mean for normalization
        std: Standard deviation for normalization

    Returns:
        Normalized energy
    """
    return (energy - mean) / std


def denormalize_energy(
    energy: torch.Tensor,
    mean: float = -4.92,
    std: float = 2.85,
) -> torch.Tensor:
    """
    Denormalize energy values.

    Args:
        energy: Normalized energy values
        mean: Mean used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized energy
    """
    return energy * std + mean


class EnergyPredictor(nn.Module):
    """
    Energy predictor module.

    Architecture similar to VariancePredictor but specialized for energy.

    Args:
        input_size: Input feature dimension
        filter_size: Hidden dimension
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int = 256,
        filter_size: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.filter_size = filter_size

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            input_size,
            filter_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            filter_size,
            filter_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Normalization
        self.norm1 = nn.LayerNorm(filter_size)
        self.norm2 = nn.LayerNorm(filter_size)

        # Output projection
        self.linear = nn.Linear(filter_size, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict energy.

        Args:
            x: Input features (B, T, D)
            mask: Optional mask (B, T, 1) or (B, 1, T)

        Returns:
            Energy predictions (B, T)
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)

        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.dropout(x)

        if mask is not None:
            x = x.masked_fill(mask.transpose(1, 2).bool(), 0.0)

        # Second conv block
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout(x)

        if mask is not None:
            x = x.masked_fill(mask.transpose(1, 2).bool(), 0.0)

        # Output projection
        energy_pred = self.linear(x).squeeze(-1)

        return energy_pred


class EnergyEmbedding(nn.Module):
    """
    Energy embedding layer.

    Converts continuous energy to discrete bins and embeds.

    Args:
        n_bins: Number of energy bins
        embedding_dim: Embedding dimension
        energy_min: Minimum energy value
        energy_max: Maximum energy value
        quantization_type: "linear" or "log" scale
    """

    def __init__(
        self,
        n_bins: int = 256,
        embedding_dim: int = 256,
        energy_min: float = -10.0,
        energy_max: float = 10.0,
        quantization_type: str = "linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.quantization_type = quantization_type

        # Embedding table
        self.embedding = nn.Embedding(n_bins, embedding_dim)

        # Register bin boundaries as buffer
        if quantization_type == "log":
            bin_edges = self._log_scale_bin_edges(n_bins, energy_min, energy_max)
        else:
            bin_edges = torch.linspace(energy_min, energy_max, n_bins + 1)
        self.register_buffer("bin_edges", bin_edges)

    def _log_scale_bin_edges(
        self, n_bins: int, energy_min: float, energy_max: float
    ) -> torch.Tensor:
        """Compute log-scale bin edges."""
        # Avoid log of negative numbers
        energy_min = max(energy_min, -50.0)
        energy_max = max(energy_max, 0.0)

        # Create log-spaced edges
        log_edges = torch.linspace(energy_min, energy_max, n_bins + 1)

        return log_edges

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Convert energy to embeddings.

        Args:
            energy: Energy values (B, T)

        Returns:
            Energy embeddings (B, T, embedding_dim)
        """
        # Convert energy to bin indices
        indices = torch.bucketize(energy, self.bin_edges)
        indices = indices.clamp(0, self.n_bins - 1)

        # Get embeddings
        embeddings = self.embedding(indices)

        return embeddings


class PhonemeLevelEnergyAggregator(nn.Module):
    """
    Aggregate frame-level energy to phone-level.

    Uses duration-based averaging.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        frame_energy: torch.Tensor,
        duration: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate frame-level energy to phone-level.

        Args:
            frame_energy: Frame-level energy (B, T_frame)
            duration: Phone durations (B, T_phone)
            mask: Optional frame mask

        Returns:
            Phone-level energy (B, T_phone)
        """
        batch_size, n_phones = duration.shape

        phone_energy = []
        for b in range(batch_size):
            phone_eng = []
            frame_idx = 0
            for p in range(n_phones):
                dur = duration[b, p].item()
                if dur > 0:
                    eng_sum = frame_energy[b, frame_idx : frame_idx + dur].sum()
                    phone_eng.append(eng_sum / dur)
                    frame_idx += dur
                else:
                    phone_eng.append(torch.tensor(0.0, device=frame_energy.device))
            phone_energy.append(torch.stack(phone_eng))

        return torch.stack(phone_energy)


__all__ = [
    "extract_energy_from_mel",
    "extract_energy_from_waveform",
    "normalize_energy",
    "denormalize_energy",
    "EnergyPredictor",
    "EnergyEmbedding",
    "PhonemeLevelEnergyAggregator",
]
