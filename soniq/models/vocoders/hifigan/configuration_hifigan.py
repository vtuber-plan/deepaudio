# coding=utf-8
"""HiFiGAN configuration."""

from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class HifiGANConfig(SoniqModelConfig):
    """
    Configuration class for HiFiGAN vocoder.

    Args:
        inter_channels: Intermediate channel size.
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilation_sizes: Dilation sizes for residual blocks.
        upsample_rates: Upsampling rates.
        upsample_initial_channel: Initial number of channels for upsampling.
        upsample_kernel_sizes: Kernel sizes for upsampling layers.
        upsample_dilation_sizes: Dilation sizes for upsampling layers.
        pre_kernel_size: Kernel size for pre-convolution.
        post_kernel_size: Kernel size for post-convolution.
        use_spectral_norm: Whether to use spectral normalization.
        lrelu_slope: Slope for LeakyReLU activation.
        initializer_range: Standard deviation for weight initialization.
        **kwargs: Additional keyword arguments.
    """

    model_type = "hifigan"

    def __init__(
        self,
        inter_channels: int = 128,
        resblock_kernel_sizes: tuple = (3, 7, 11, 13),
        resblock_dilation_sizes: tuple = (1, 3, 5),
        upsample_rates: tuple = (8, 8, 4, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple = (16, 16, 8, 4),
        upsample_dilation_sizes: tuple = (1, 1, 1, 1),
        pre_kernel_size: int = 13,
        post_kernel_size: int = 13,
        use_spectral_norm: bool = False,
        lrelu_slope: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.inter_channels = inter_channels
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_dilation_sizes = upsample_dilation_sizes
        self.pre_kernel_size = pre_kernel_size
        self.post_kernel_size = post_kernel_size
        self.use_spectral_norm = use_spectral_norm
        self.lrelu_slope = lrelu_slope

        super().__init__(initializer_range=initializer_range, **kwargs)
