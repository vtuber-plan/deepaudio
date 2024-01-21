""" Multi Period Scale Discriminator configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class MultiPSConfig(PretrainedConfig):
    model_type = "multips"
    attribute_map = {}

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11, 17, 23, 37],
        use_spectral_norm=False,
        **kwargs
    ):
        """Constructs MultiPSConfig."""
        self.periods = periods
        self.use_spectral_norm = use_spectral_norm
        super().__init__(**kwargs)
