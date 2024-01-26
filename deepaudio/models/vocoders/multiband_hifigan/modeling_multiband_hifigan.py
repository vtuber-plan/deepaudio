# coding=utf-8
from transformers.utils import logging

logger = logging.get_logger(__name__)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import numpy as np

from transformers import PreTrainedModel

from deepaudio.models.vocoders.multiband_hifigan.configuration_multiband_hifigan import MultiBandHifiGANConfig

class MultiBandHifiGANPreTrainedModel(PreTrainedModel):
    config_class = MultiBandHifiGANConfig
    base_model_prefix = "multibandhifigan"
    supports_gradient_checkpointing = False

# This code is adopted from multiband-hifigan
# https://github.com/rishikksh20/multiband-hifigan

