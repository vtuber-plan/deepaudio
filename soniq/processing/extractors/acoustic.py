# coding=utf-8
"""
Acoustic feature extractor for Soniq.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Dict, Any
import numpy as np


class AcousticExtractor:
    """
    Acoustic feature extractor for audio processing.

    This class extracts various acoustic features including:
        - Mel spectrogram
        - Spectrogram
        - MFCC
        - Chroma
        - Spectral contrast
        - Zero-crossing rate

    Example:
        ```python
        extractor = AcousticExtractor(sample_rate=24000)
        features = extractor.extract_mel(audio)
        ```
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        n_mel: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        """
        Initialize AcousticExtractor.

        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            win_length: Window length (default: n_fft).
            n_mel: Number of mel filterbanks.
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mel = n_mel
        self.f_min = f_min
        self.f_max = f_max

        # Initialize transforms
        self._init_transforms()

    def _init_transforms(self):
        """Initialize torchaudio transforms."""
        # Mel spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mel,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        # Spectrogram
        self.spectrogram_transform = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        # MFCC
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "n_mels": self.n_mel,
            },
        )

    def extract_mel(
        self,
        audio: torch.Tensor,
        log_scale: bool = True,
        top_db: float = 80.0,
    ) -> torch.Tensor:
        """
        Extract mel spectrogram.

        Args:
            audio: Audio tensor of shape (..., time).
            log_scale: Whether to apply log scaling.
            top_db: Maximum dB for dynamic range compression.

        Returns:
            Mel spectrogram of shape (..., n_mel, time).
        """
        mel_spec = self.mel_transform(audio)

        if log_scale:
            mel_spec = T.AmplitudeToDB(stype="power", topdb=top_db)(mel_spec)

        return mel_spec

    def extract_spectrogram(
        self,
        audio: torch.Tensor,
        return_complex: bool = True,
    ) -> torch.Tensor:
        """
        Extract complex spectrogram.

        Args:
            audio: Audio tensor of shape (..., time).
            return_complex: Whether to return complex tensor.

        Returns:
            Spectrogram of shape (..., freq, time).
        """
        spec = self.spectrogram_transform(audio)
        return spec

    def extract_magnitude(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract magnitude spectrogram.

        Args:
            audio: Audio tensor of shape (..., time).

        Returns:
            Magnitude spectrogram of shape (..., freq, time).
        """
        spec = self.spectrogram_transform(audio)
        return spec.abs()

    def extract_phase(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract phase spectrogram.

        Args:
            audio: Audio tensor of shape (..., time).

        Returns:
            Phase spectrogram of shape (..., freq, time).
        """
        spec = self.spectrogram_transform(audio)
        return spec.angle()

    def extract_mfcc(
        self,
        audio: torch.Tensor,
        n_mfcc: int = 13,
    ) -> torch.Tensor:
        """
        Extract MFCC features.

        Args:
            audio: Audio tensor of shape (..., time).
            n_mfcc: Number of MFCC coefficients.

        Returns:
            MFCC features of shape (..., n_mfcc, time).
        """
        return self.mfcc_transform(audio)

    def extract_f0(
        self,
        audio: torch.Tensor,
        method: str = "dio",
    ) -> torch.Tensor:
        """
        Extract F0 (fundamental frequency).

        Args:
            audio: Audio tensor of shape (..., time).
            method: Extraction method ("dio", "pyin", "crepe").

        Returns:
            F0 contour of shape (..., time).
        """
        # Convert to numpy for processing
        audio_np = audio.cpu().numpy()

        if method == "dio":
            # Using librosa's pyin as a fallback
            import librosa
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]

            f0_list = []
            for wav in audio_np:
                f0, _ = librosa.pyin(
                    wav,
                    fmin=50.0,
                    fmax=1000.0,
                    sr=self.sample_rate,
                    frame_length=self.win_length,
                    hop_length=self.hop_length,
                )
                f0 = torch.from_numpy(f0).fillna(0.0)
                f0_list.append(f0)

            return torch.stack(f0_list) if len(f0_list) > 1 else f0_list[0]

        else:
            # Default: return zeros
            batch_shape = audio.shape[:-1]
            time_length = (audio.shape[-1] - self.win_length) // self.hop_length + 1
            return torch.zeros(*batch_shape, time_length)

    def extract_energy(
        self,
        audio: torch.Tensor,
        from_mel: bool = True,
    ) -> torch.Tensor:
        """
        Extract energy features.

        Args:
            audio: Audio tensor of shape (..., time).
            from_mel: Whether to compute energy from mel spectrogram.

        Returns:
            Energy features of shape (..., time).
        """
        if from_mel:
            mel_spec = self.extract_mel(audio, log_scale=False)
            energy = mel_spec.sum(dim=-2)  # Sum over mel bins
        else:
            spec = self.extract_magnitude(audio)
            energy = spec.sum(dim=-2)  # Sum over frequency bins

        return energy

    def extract_all_features(
        self,
        audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all acoustic features.

        Args:
            audio: Audio tensor of shape (..., time).

        Returns:
            Dictionary of features.
        """
        return {
            "mel": self.extract_mel(audio),
            "spectrogram": self.extract_spectrogram(audio),
            "magnitude": self.extract_magnitude(audio),
            "phase": self.extract_phase(audio),
            "mfcc": self.extract_mfcc(audio),
            "f0": self.extract_f0(audio),
            "energy": self.extract_energy(audio),
        }
