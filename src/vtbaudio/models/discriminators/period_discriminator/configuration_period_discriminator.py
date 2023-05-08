
import warnings

from ....configuration_utils import DiscriminatorConfig
from ....utils import logging


logger = logging.get_logger(__name__)

class PeriodDiscriminatorConfig(DiscriminatorConfig):
    model_type = "period_discriminator"
    attribute_map = {
        "period": "period",
        "kernel_size": "kernel_size",
        "stride": "stride",
        "use_spectral_norm": "use_spectral_norm",
        "channels": "channels"
    }

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
