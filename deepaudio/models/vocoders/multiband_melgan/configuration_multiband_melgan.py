# coding=utf-8
""" MultiBandMelGAN configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MultiBandMelGANConfig(PretrainedConfig):
    model_type = "multibandmelgan"
    attribute_map = {}

    def __init__(
        self,
        **kwargs
    ):
        """Constructs MultiBandMelGANConfig."""
        super().__init__(**kwargs)
