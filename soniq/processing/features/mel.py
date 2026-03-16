# coding=utf-8
"""Mel spectrogram extractor."""

from typing import Any, Dict
import torch
import torchaudio
from ..base import BaseProcessor


class MelSpectrogramExtractor(BaseProcessor):
    """Extract mel spectrograms from audio."""

    def __init__(self, config):
        super().__init__(config)
        self.sample_rate = getattr(config, 'sample_rate', 24000)
        self.n_fft = getattr(config, 'n_fft', 1024)
        self.n_mel = getattr(config, 'n_mel', 80)
        self.hop_length = getattr(config, 'hop_length', 256)

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
