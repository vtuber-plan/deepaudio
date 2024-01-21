""" Multi Resolution Discriminator configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class MultiResolutionConfig(PretrainedConfig):
    model_type = "multiresolution"
    attribute_map = {}

    def __init__(
        self,
        resolutions=[[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
        channel_mult=1,
        use_spectral_norm=False,
        lrelu_slope=0.1,
        **kwargs
    ):
        """Constructs MultiResolutionConfig."""
        self.resolutions = resolutions
        self.channel_mult = channel_mult
        self.use_spectral_norm = use_spectral_norm
        self.lrelu_slope = lrelu_slope
        super().__init__(**kwargs)
