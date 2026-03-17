# coding=utf-8
"""F0 (pitch) extraction and processing utilities."""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import numpy as np


def get_f0_features_using_parselmouth(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    f0_min: float = 50.0,
    f0_max: float = 800.0,
) -> np.ndarray:
    """
    Extract F0 features using Praat (parselmouth).

    Args:
        audio: Audio waveform (numpy array)
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz

    Returns:
        F0 sequence (frame-level)
    """
    try:
        import parselmouth
    except ImportError:
        raise ImportError("parselmouth is required for F0 extraction. Install with: pip install praat-parselmouth")

    # Create Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    # Extract F0 using Praat's autocorrelation method
    f0 = sound.to_ac()
    f0 = f0.to_pitch_ac(
        time_step=hop_length / sample_rate,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )

    # Get F0 values
    f0_values = f0.selected_array["frequency"]

    # Pad to match expected length
    expected_length = int(np.ceil(len(audio) / hop_length))
    if len(f0_values) < expected_length:
        f0_values = np.pad(f0_values, (0, expected_length - len(f0_values)))

    return f0_values


def get_f0_features_using_pyin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    f0_min: float = 50.0,
    f0_max: float = 800.0,
) -> np.ndarray:
    """
    Extract F0 features using librosa's pyin.

    Args:
        audio: Audio waveform (numpy array)
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz

    Returns:
        F0 sequence (frame-level)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for pyin F0 extraction")

    f0, _, _ = librosa.pyin(
        audio,
        fmin=f0_min,
        fmax=f0_max,
        sr=sample_rate,
        frame_length=2048,
        win_length=2048,
        hop_length=hop_length,
    )

    # Fill NaN (unvoiced) with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    return f0


def get_f0_features_using_crepe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    f0_min: float = 50.0,
    f0_max: float = 800.0,
    model: str = "full",
) -> np.ndarray:
    """
    Extract F0 features using CREPE (neural network based).

    Args:
        audio: Audio waveform (numpy array)
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz
        model: Model size ("tiny" or "full")

    Returns:
        F0 sequence (frame-level)
    """
    try:
        import crepe
    except ImportError:
        raise ImportError("crepe is required for CREPE F0 extraction")

    # Run CREPE
    f0, _, _, _ = crepe.predict(
        audio,
        sr=sample_rate,
        viterbi=True,
        model=model,
        step_size=int(hop_length / sample_rate * 1000),  # Convert to ms
    )

    # Apply min/max clipping
    f0 = np.where(f0 < f0_min, 0.0, f0)
    f0 = np.where(f0 > f0_max, 0.0, f0)

    return f0


class F0Extractor:
    """
    Unified F0 extractor supporting multiple backends.

    Args:
        extractor_type: Type of extractor ("parselmouth", "pyin", "crepe", "dio", "harvest")
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz
    """

    def __init__(
        self,
        extractor_type: str = "parselmouth",
        sample_rate: int = 24000,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 800.0,
    ):
        self.extractor_type = extractor_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract F0 from audio.

        Args:
            audio: Audio waveform

        Returns:
            F0 sequence
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim == 2:
            audio = audio.mean(0)  # Convert to mono

        if self.extractor_type == "parselmouth":
            return get_f0_features_using_parselmouth(
                audio, self.sample_rate, self.hop_length, self.f0_min, self.f0_max
            )
        elif self.extractor_type == "pyin":
            return get_f0_features_using_pyin(
                audio, self.sample_rate, self.hop_length, self.f0_min, self.f0_max
            )
        elif self.extractor_type == "crepe":
            return get_f0_features_using_crepe(
                audio, self.sample_rate, self.hop_length, self.f0_min, self.f0_max
            )
        else:
            raise ValueError(f"Unknown extractor type: {self.extractor_type}")


def f0_to_coarse(f0: np.ndarray, f0_min: float = 50.0, f0_max: float = 1100.0) -> np.ndarray:
    """
    Convert F0 (Hz) to mel-scale coarse representation.

    Args:
        f0: F0 values in Hz
        f0_min: Minimum F0
        f0_max: Maximum F0

    Returns:
        Coarse F0 (mel-scale)
    """
    # Convert to mel scale
    f0_mel = 1127 * np.log(1 + f0 / 700)

    # Normalize to [0, 1]
    f0_min_mel = 1127 * np.log(1 + f0_min / 700)
    f0_max_mel = 1127 * np.log(1 + f0_max / 700)

    f0_norm = (f0_mel - f0_min_mel) / (f0_max_mel - f0_min_mel)

    return f0_norm


def coarse_to_f0(coarse: np.ndarray, f0_min: float = 50.0, f0_max: float = 1100.0) -> np.ndarray:
    """
    Convert coarse mel-scale back to F0 (Hz).

    Args:
        coarse: Coarse F0 values [0, 1]
        f0_min: Minimum F0
        f0_max: Maximum F0

    Returns:
        F0 values in Hz
    """
    f0_min_mel = 1127 * np.log(1 + f0_min / 700)
    f0_max_mel = 1127 * np.log(1 + f0_max / 700)

    # Convert from [0, 1] to mel scale
    f0_mel = coarse * (f0_max_mel - f0_min_mel) + f0_min_mel

    # Convert back to Hz
    f0 = 700 * (np.exp(f0_mel / 1127) - 1)

    return f0


def interpolate_f0(f0: np.ndarray, uv_threshold: float = 0.0) -> np.ndarray:
    """
    Interpolate unvoiced regions in F0.

    Args:
        f0: F0 sequence (0 for unvoiced)
        uv_threshold: Threshold for unvoiced detection

    Returns:
        Interpolated F0
    """
    # Find unvoiced frames
    uv = f0 <= uv_threshold

    # If all voiced or all unvoiced, return as is
    if uv.all() or (~uv).all():
        return f0

    # Linear interpolation
    f0_interp = f0.copy()
    f0_interp[uv] = np.interp(
        np.where(uv)[0],
        np.where(~uv)[0],
        f0[~uv]
    )

    return f0_interp


def get_log_f0(f0: np.ndarray, min_f0: float = 5.0) -> np.ndarray:
    """
    Convert F0 to log scale.

    Args:
        f0: F0 values in Hz
        min_f0: Minimum F0 for log stability

    Returns:
        Log F0
    """
    return np.log(np.clip(f0, min_f0, None))


class F0Predictor(nn.Module):
    """
    Neural F0 predictor module.

    Can be used as a standalone predictor or as part of a larger TTS model.

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

        # Output: F0 + UV (unvoiced flag)
        self.linear_f0 = nn.Linear(filter_size, 1)
        self.linear_uv = nn.Linear(filter_size, 1)

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
    ) -> Dict[str, torch.Tensor]:
        """
        Predict F0 and unvoiced flag.

        Args:
            x: Input features (B, T, D)
            mask: Optional mask (B, T, 1) or (B, 1, T)

        Returns:
            Dictionary with 'f0' and 'uv' predictions
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

        # Output projections
        f0_pred = self.linear_f0(x).squeeze(-1)
        uv_pred = self.linear_uv(x).squeeze(-1)

        # Convert UV to probability
        uv_pred = torch.sigmoid(uv_pred)

        return {"f0": f0_pred, "uv": uv_pred}


class PitchEmbedding(nn.Module):
    """
    Pitch (F0) embedding layer.

    Converts continuous F0 to discrete bins and embeds.

    Args:
        n_bins: Number of pitch bins
        embedding_dim: Embedding dimension
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz
        quantization_type: "linear" or "mel" scale
    """

    def __init__(
        self,
        n_bins: int = 256,
        embedding_dim: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        quantization_type: str = "mel",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.quantization_type = quantization_type

        # Embedding table
        self.embedding = nn.Embedding(n_bins, embedding_dim)

        # Register bin boundaries as buffer
        if quantization_type == "mel":
            bin_edges = self._mel_scale_bin_edges(n_bins, f0_min, f0_max)
        else:
            bin_edges = torch.linspace(f0_min, f0_max, n_bins + 1)
        self.register_buffer("bin_edges", bin_edges)

    def _mel_scale_bin_edges(
        self, n_bins: int, f0_min: float, f0_max: float
    ) -> torch.Tensor:
        """Compute mel-scale bin edges."""
        f0_min_mel = 1127 * np.log(1 + f0_min / 700)
        f0_max_mel = 1127 * np.log(1 + f0_max / 700)

        mel_edges = np.linspace(f0_min_mel, f0_max_mel, n_bins + 1)
        hz_edges = 700 * (np.exp(mel_edges / 1127) - 1)

        return torch.tensor(hz_edges, dtype=torch.float32)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Convert F0 to embeddings.

        Args:
            f0: F0 values in Hz (B, T)

        Returns:
            Pitch embeddings (B, T, embedding_dim)
        """
        # Convert F0 to bin indices
        indices = torch.bucketize(f0, self.bin_edges)
        indices = indices.clamp(0, self.n_bins - 1)

        # Get embeddings
        embeddings = self.embedding(indices)

        return embeddings


__all__ = [
    "F0Extractor",
    "F0Predictor",
    "PitchEmbedding",
    "f0_to_coarse",
    "coarse_to_f0",
    "interpolate_f0",
    "get_log_f0",
]
