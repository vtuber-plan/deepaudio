# coding=utf-8
from transformers.utils import logging

logger = logging.get_logger(__name__)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import numpy as np

from transformers import PreTrainedModel

from deepaudio.models.vocoders.multiband_melgan.configuration_multiband_melgan import MultiBandMelGANConfig

class MultiBandMelGANPreTrainedModel(PreTrainedModel):
    config_class = MultiBandMelGANConfig
    base_model_prefix = "multibandmelgan"
    supports_gradient_checkpointing = False

# This code is adopted from multiband-hifigan
# https://github.com/rishikksh20/multiband-hifigan

