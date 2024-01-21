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
        **kwargs
    ):
        """Constructs MultiResolutionConfig."""
        super().__init__(**kwargs)
