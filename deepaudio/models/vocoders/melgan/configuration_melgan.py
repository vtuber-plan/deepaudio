# coding=utf-8
""" MelGAN configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MelGANConfig(PretrainedConfig):
    model_type = "melgan"
    attribute_map = {}

    def __init__(
        self,
        n_mel=80,
        ngf=32,
        n_residual_layers=3,
        ratios=[8, 8, 2, 2],
        **kwargs
    ):
        """Constructs MelGANConfig."""
        self.n_mel = n_mel
        self.ngf = ngf
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        super().__init__(**kwargs)
