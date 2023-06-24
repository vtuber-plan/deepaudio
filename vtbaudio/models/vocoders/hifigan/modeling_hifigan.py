

from functools import reduce
import operator
from typing import Iterable, List, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from ....utils.model_utils import init_weights, get_padding

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

class HifiGANGenerator(torch.nn.Module):
    def __init__(self, initial_channel: int,
                    resblock_kernel_sizes: List[int],
                    resblock_dilation_sizes: List[int],
                    upsample_rates: List[int],
                    upsample_initial_channel: int,
                    upsample_kernel_sizes: List[int],
                    upsample_dilation_sizes: List[int],
                    pre_kernel_size: int=11,
                    post_kernel_size: int=11):
        super(HifiGANGenerator, self).__init__()
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
