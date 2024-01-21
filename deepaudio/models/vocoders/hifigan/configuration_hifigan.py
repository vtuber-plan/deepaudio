# coding=utf-8
""" HifiGAN configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class HifiGANConfig(PretrainedConfig):
    model_type = "hifigan"
    attribute_map = {}

    def __init__(
        self,
        inter_channels: int=128,
        resblock_kernel_sizes = [3,7,11,13],
        resblock_dilation_sizes =  [1,3,5],
        upsample_rates = [8,8,4,2],
        upsample_initial_channel = 512,
        upsample_kernel_sizes = [16,16,8,4],
        upsample_dilation_sizes = [1,1,1,1],
        pre_kernel_size = 13,
        post_kernel_size = 13,
        use_spectral_norm = False,
        multi_period_discriminator_periods = [2, 3, 5, 7, 11, 13, 17, 19, 23, 37],
        **kwargs
    ):
        """Constructs HifiGANConfig."""
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
        self.multi_period_discriminator_periods = multi_period_discriminator_periods
        super().__init__(**kwargs)
