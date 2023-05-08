
import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from ....utils.generic import get_padding
from .configuration_period_discriminator import PeriodDiscriminatorConfig

class DiscriminatorP(torch.nn.Module):
    def __init__(self, config: PeriodDiscriminatorConfig):
        super(DiscriminatorP, self).__init__()
        self.config = config
        self.period = config.period
        self.use_spectral_norm = config.use_spectral_norm
        norm_f = weight_norm if config.use_spectral_norm == False else spectral_norm
        kernel_size = config.kernel_size
        stride = config.stride

        convs_list = []
        last_channel = 1
        for channel in config.channels:
            convs_list.append(Conv2d(last_channel, channel, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)))
            last_channel = channel
        convs_list.append(Conv2d(last_channel, last_channel, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0)))

        self.convs = nn.ModuleList(convs_list)
        self.conv_post = norm_f(Conv2d(last_channel, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

