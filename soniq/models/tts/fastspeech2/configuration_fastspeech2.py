# coding=utf-8
"""
FastSpeech2 configuration.

FastSpeech2 is a non-autoregressive TTS model that uses:
- Transformer encoder-decoder architecture
- Duration predictor for length regulation
- Pitch and energy predictors for prosody modeling
"""

from typing import Optional
from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class FastSpeech2Config(SoniqModelConfig):
    """
    Configuration class for FastSpeech2.

    FastSpeech2 is a non-autoregressive TTS model that directly predicts
    mel spectrogram frames from text using duration-guided length regulation.

    Args:
        n_vocab: Vocabulary size (number of text tokens).
        n_mel: Number of mel frequency bins.
        hidden_channels: Hidden dimension for encoder/decoder.
        filter_channels: FFN hidden dimension in transformer.
        n_heads: Number of attention heads.
        encoder_layers: Number of encoder layers.
        decoder_layers: Number of decoder layers.
        encoder_dropout: Encoder dropout rate.
        decoder_dropout: Decoder dropout rate.
        variance_predictor_filter_size: Filter size for variance predictors.
        variance_predictor_kernel_size: Kernel size for variance predictors.
        variance_predictor_dropout: Dropout for variance predictors.
        pitch_n_bins: Number of pitch bins for discretization.
        energy_n_bins: Number of energy bins for discretization.
        pitch_min: Minimum pitch value (Hz).
        pitch_max: Maximum pitch value (Hz).
        energy_min: Minimum energy value.
        energy_max: Maximum energy value.
        n_speakers: Number of speakers (0 for single-speaker).
        gin_channels: Speaker embedding dimension.
    """

    model_type = "fastspeech2"

    def __init__(
        self,
        n_vocab: int = 512,
        n_mel: int = 80,
        hidden_channels: int = 256,
        filter_channels: int = 1024,
        n_heads: int = 2,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.2,
        variance_predictor_filter_size: int = 256,
        variance_predictor_kernel_size: int = 3,
        variance_predictor_dropout: float = 0.5,
        pitch_n_bins: int = 256,
        energy_n_bins: int = 256,
        pitch_min: float = 50.0,
        pitch_max: float = 1100.0,
        energy_min: float = 0.0,
        energy_max: float = 100.0,
        n_speakers: int = 0,
        gin_channels: int = 256,
        max_seq_len: int = 1000,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.n_vocab = n_vocab
        self.n_mel = n_mel
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.variance_predictor_filter_size = variance_predictor_filter_size
        self.variance_predictor_kernel_size = variance_predictor_kernel_size
        self.variance_predictor_dropout = variance_predictor_dropout
        self.pitch_n_bins = pitch_n_bins
        self.energy_n_bins = energy_n_bins
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.max_seq_len = max_seq_len

        # Validate configuration
        if encoder_layers < 1:
            raise ValueError("encoder_layers must be at least 1")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be at least 1")

        super().__init__(initializer_range=initializer_range, **kwargs)
