
from typing import Iterable, List, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from transformers import PreTrainedModel
from deepaudio.models.discriminator.multiscale.configuration_multiscale import MultiScaleConfig

from ....utils.model_utils import init_weights, get_padding

LRELU_SLOPE = 0.1

class MultiScalePreTrainedModel(PreTrainedModel):
    config_class = MultiScaleConfig
    base_model_prefix = "multips"
    supports_gradient_checkpointing = False

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=512, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(MultiScalePreTrainedModel):
    def __init__(self, config: MultiScaleConfig):
        super(MultiScaleDiscriminator, self).__init__(config)
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=config.use_spectral_norm),
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
