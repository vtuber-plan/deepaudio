
import re
from typing import List, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .configuration_utils import PretrainedAudioConfig
from transformers.utils.hub import PushToHubMixin

from transformers.modeling_utils import get_parameter_device, get_parameter_dtype

class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """
    
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    

class PreTrainedAudioModel(nn.Module, ModuleUtilsMixin, PushToHubMixin):
    config_class = None
    base_model_prefix = ""

    is_parallelizable = False
    supports_gradient_checkpointing = False

    def __init__(self, config: PretrainedAudioConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedAudioConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}

    def post_init(self):
        pass