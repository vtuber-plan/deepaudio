""" HifiGAN configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MelGANConfig(PretrainedConfig):
    model_type = "melgan"
    attribute_map = {}

    def __init__(
        self,
        inter_channels: int,
        **kwargs
    ):
        """Constructs HifiGANConfig."""
        self.inter_channels = inter_channels
        super().__init__(**kwargs)
