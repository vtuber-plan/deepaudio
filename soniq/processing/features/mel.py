# coding=utf-8
"""Mel spectrogram extraction utilities."""

from typing import Any, Dict, Optional
import torch
import torchaudio


def extract_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mel: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Extract mel spectrogram from audio.

    Args:
        audio: Audio waveform (1, T) or (T,)
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mel: Number of mel bins
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Mel spectrogram (n_mel, T)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mel,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
    ).to(audio.device)

    mel = mel_transform(audio)
    return mel.squeeze(0)


class MelSpectrogramExtractor:
    """Extract mel spectrograms from audio (processor interface)."""

    def __init__(self, config=None):
        self.sample_rate = getattr(config, 'sample_rate', 24000) if config else 24000
        self.n_fft = getattr(config, 'n_fft', 1024) if config else 1024
        self.n_mel = getattr(config, 'n_mel', 80) if config else 80
        self.hop_length = getattr(config, 'hop_length', 256) if config else 256

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mel,
            hop_length=self.hop_length,
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'audio' in sample:
            audio = sample['audio']
            mel = self.mel_transform(audio)
            sample['mel'] = mel.squeeze(0)
        return sample
