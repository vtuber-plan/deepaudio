# coding=utf-8
"""
BigVGAN configuration.
"""

from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class BigVGANConfig(SoniqModelConfig):
    """
    Configuration class for BigVGAN.

    BigVGAN is a neural vocoder that uses periodic activation functions
    (Snake/SnakeBeta) with anti-aliasing for high-quality waveform generation.

    Args:
        inter_channels: Number of intermediate channels.
        upsample_initial_channel: Initial number of channels for upsampling.
        upsample_rates: Upsampling rates for each layer.
        upsample_kernel_sizes: Kernel sizes for each upsampling layer.
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilation_sizes: Dilation sizes for residual blocks.
        resblock: Residual block type ("1" or "2").
        activation: Activation function ("snake" or "snakebeta").
        snake_logscale: If True, snake parameters are learned in log scale.
        lrelu_slope: Slope for LeakyReLU (used in some layers).
        pre_kernel_size: Kernel size for pre-convolution.
        post_kernel_size: Kernel size for post-convolution.
        n_mel: Number of mel frequency bins.
        hop_length: Hop length for audio generation.
    """

    model_type = "bigvgan"

    def __init__(
        self,
        inter_channels: int = 192,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple = (4, 4, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple = (8, 8, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock: str = "1",
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        lrelu_slope: float = 0.1,
        pre_kernel_size: int = 7,
        post_kernel_size: int = 7,
        n_mel: int = 128,
        hop_length: int = 512,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.inter_channels = inter_channels
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.resblock = resblock
        self.activation = activation
        self.snake_logscale = snake_logscale
        self.lrelu_slope = lrelu_slope
        self.pre_kernel_size = pre_kernel_size
        self.post_kernel_size = post_kernel_size
        self.n_mel = n_mel
        self.hop_length = hop_length

        # Validate configuration
        if activation not in ["snake", "snakebeta"]:
            raise ValueError(f"activation must be 'snake' or 'snakebeta', got {activation}")
        if resblock not in ["1", "2"]:
            raise ValueError(f"resblock must be '1' or '2', got {self.resblock}")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must have same length")

        super().__init__(initializer_range=initializer_range, **kwargs)
