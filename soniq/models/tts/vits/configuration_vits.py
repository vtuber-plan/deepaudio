# coding=utf-8
"""
VITS configuration.

VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)
is an end-to-end TTS model that generates waveforms directly from text.
"""

from typing import Optional, Tuple
from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class VITSConfig(SoniqModelConfig):
    """
    Configuration class for VITS.

    VITS is an end-to-end TTS model that combines:
    - Conditional Variational Autoencoder (VAE)
    - Flow-based latent variable transformation
    - Adversarial learning with GAN discriminator

    Args:
        n_vocab: Vocabulary size (number of text tokens).
        n_speakers: Number of speakers (0 for single-speaker).
        inter_channels: Number of intermediate channels.
        hidden_channels: Hidden dimension for encoder/decoder.
        filter_channels: FFN hidden dimension in transformer.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        kernel_size: Kernel size for transformer convolutions.
        p_dropout: Dropout rate.
        resblock: Residual block type ("1" or "2").
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilation_sizes: Dilation sizes for residual blocks.
        upsample_rates: Upsampling rates for generator.
        upsample_initial_channel: Initial channels for generator upsampling.
        upsample_kernel_sizes: Kernel sizes for generator upsampling.
        n_layers_q: Number of posterior encoder layers.
        use_spectral_norm: Whether to use spectral norm in discriminator.
        gin_channels: Speaker embedding dimension.
        use_sdp: Whether to use stochastic duration predictor.
        n_flows: Number of flow steps.
        segment_size: Training segment size.
        hop_length: Hop length for audio generation.
        win_length: Window length for STFT.
        n_fft: FFT size.
        n_mel: Number of mel bins.
        fmin: Minimum frequency for mel filters.
        fmax: Maximum frequency for mel filters.
    """

    model_type = "vits"

    def __init__(
        self,
        n_vocab: int = 512,
        n_speakers: int = 0,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 256,
        use_sdp: bool = True,
        n_flows: int = 4,
        segment_size: int = 8192,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        n_mel: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        c_mel: float = 45.0,
        c_kl: float = 1.0,
        c_dur: float = 1.0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.n_vocab = n_vocab
        self.n_speakers = n_speakers
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_layers_q = n_layers_q
        self.use_spectral_norm = use_spectral_norm
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.n_flows = n_flows
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mel = n_mel
        self.fmin = fmin
        self.fmax = fmax
        self.c_mel = c_mel
        self.c_kl = c_kl
        self.c_dur = c_dur

        # Validate configuration
        if resblock not in ["1", "2"]:
            raise ValueError(f"resblock must be '1' or '2', got {self.resblock}")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must have same length")

        super().__init__(initializer_range=initializer_range, **kwargs)
