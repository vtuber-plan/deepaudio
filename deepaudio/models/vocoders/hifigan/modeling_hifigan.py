

from functools import reduce
import operator
from typing import Iterable, List, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from transformers import PreTrainedModel

from deepaudio.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig

from ....utils.model_utils import init_weights, get_padding

LRELU_SLOPE = 0.1

class HifiGANResBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int=3, dilation: Iterable[int]=(1, 3, 5, 7), lrelu_slope: float=0.1):
        super(HifiGANResBlock, self).__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList()
        for dilation_size in dilation:
            conv_kernel = weight_norm(
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=get_padding(kernel_size, dilation_size)
                )
            )
            self.convs1.append(conv_kernel)
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList()
        for dilation_size in dilation:
            conv_kernel = weight_norm(
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            )
            self.convs2.append(conv_kernel)
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class HifiGANDiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(HifiGANDiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

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

class HifiGANDiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(HifiGANDiscriminatorS, self).__init__()
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

class HifiGANMultiPeriodDiscriminator(PreTrainedModel):
    def __init__(self, periods: List[int]=[2, 3, 5, 7, 11, 17, 23, 37], use_spectral_norm: bool=False):
        super(HifiGANMultiPeriodDiscriminator, self).__init__()
        self.periods = periods
        discs = [HifiGANDiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [HifiGANDiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, g=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class HifiGANMultiScaleDiscriminator(PreTrainedModel):
    def __init__(self, use_spectral_norm=False):
        super(HifiGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiGANDiscriminatorS(use_spectral_norm=use_spectral_norm),
            HifiGANDiscriminatorS(),
            HifiGANDiscriminatorS(),
            HifiGANDiscriminatorS(),
            HifiGANDiscriminatorS(),
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

class HifiGANGenerator(PreTrainedModel):
    def __init__(self, config: HifiGANConfig):
        super(HifiGANGenerator, self).__init__()

        initial_channel = config.inter_channels
        resblock_kernel_sizes = config.resblock_kernel_sizes
        resblock_dilation_sizes = config.resblock_dilation_sizes
        upsample_rates = config.upsample_rates
        upsample_initial_channel = config.upsample_initial_channel
        upsample_kernel_sizes = config.upsample_kernel_sizes
        upsample_dilation_sizes = config.upsample_dilation_sizes
        pre_kernel_size = config.pre_kernel_size
        post_kernel_size = config.post_kernel_size

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.n_head = 4
        self.lrelu_slope = 0.1

        self.conv_pre = Conv1d(
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=pre_kernel_size,
            stride=1,
            padding=(pre_kernel_size-1)//2
        )

        self.ups = nn.ModuleList()
        for i, (u, k, d) in enumerate(zip(upsample_rates, upsample_kernel_sizes, upsample_dilation_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel//(2**i),
                    out_channels=upsample_initial_channel//(2**(i+1)),
                    kernel_size=k,
                    stride=u,
                    padding=(((k-1)*d+1)-u)//2,
                    dilation=d
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(HifiGANResBlock(channels=ch, kernel_size=k, dilation=d, lrelu_slope=self.lrelu_slope))

        self.conv_post = Conv1d(ch, 1, post_kernel_size, 1, padding=(post_kernel_size-1)//2, bias=False)
        # self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            up_x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](up_x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](up_x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
