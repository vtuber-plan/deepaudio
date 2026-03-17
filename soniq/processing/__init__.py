# coding=utf-8
"""
Signal processing utilities for Soniq.

This module provides:
- Audio I/O and processing
- Feature extraction (mel, f0, etc.)
- Dataset preprocessing extractors

Example:
    ```python
    from soniq.processing import AcousticExtractor, extract_mel_spectrogram
    from soniq.processing.features import F0Features

    # Extract mel spectrogram
    mel = extract_mel_spectrogram(audio, sample_rate=24000)

    # Use acoustic extractor
    extractor = AcousticExtractor(sample_rate=24000)
    features = extractor.extract_mel(audio)
    ```
"""

# Audio processing
from .audio import load_audio, save_audio

# Feature extraction
from .features import extract_mel_spectrogram, F0Features, MelFeatures

# Preprocessing extractors
from .extractors import AcousticExtractor, PhoneExtractor

__all__ = [
    # Audio
    "load_audio",
    "save_audio",
    # Features
    "extract_mel_spectrogram",
    "F0Features",
    "MelFeatures",
    # Extractors
    "AcousticExtractor",
    "PhoneExtractor",
]