# coding=utf-8
""" MultiBandHifiGAN configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MultiBandHifiGANConfig(PretrainedConfig):
    model_type = "multibandhifigan"
    attribute_map = {}

    def __init__(
        self,
        **kwargs
    ):
        """Constructs MultiBandHifiGANConfig."""
        super().__init__(**kwargs)
