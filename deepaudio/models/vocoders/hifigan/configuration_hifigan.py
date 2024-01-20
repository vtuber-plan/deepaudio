""" HifiGAN configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class HifiGANConfig(PretrainedConfig):
    model_type = "hifigan"
    attribute_map = {
        "inter_channels": "inter_channels",
        "resblock_kernel_sizes": "resblock_kernel_sizes",
        "resblock_dilation_sizes": "resblock_dilation_sizes",
        "upsample_rates": "upsample_rates",
        "upsample_initial_channel": "upsample_initial_channel",
        "upsample_kernel_sizes": "upsample_kernel_sizes",
        "upsample_dilation_sizes": "upsample_dilation_sizes",
        "pre_kernel_size": "pre_kernel_size",
        "post_kernel_size": "post_kernel_size",
        "use_spectral_norm": "use_spectral_norm",
        "multi_period_discriminator_periods": "multi_period_discriminator_periods",
    }

    def __init__(
        self,
        inter_channels: int,
        **kwargs
    ):
        """Constructs HifiGANConfig."""
        self.inter_channels = inter_channels
        super().__init__(**kwargs)
