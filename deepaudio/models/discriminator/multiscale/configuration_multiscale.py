""" Multi Scale Discriminator configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class MultiScaleConfig(PretrainedConfig):
    model_type = "multiscale"
    attribute_map = {}

    def __init__(
        self,
        use_spectral_norm=False,
        **kwargs
    ):
        """Constructs MultiScaleConfig."""
        self.use_spectral_norm = use_spectral_norm
        super().__init__(**kwargs)
