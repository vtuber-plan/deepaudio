# This code is a refined MRD adopted from BigVGAN under the MIT License
# https://github.com/NVIDIA/BigVGAN


from typing import Iterable, List, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from transformers import PreTrainedModel
from deepaudio.models.discriminator.multiresolution.configuration_multiresolution import MultiResolutionConfig

from ....utils.model_utils import init_weights, get_padding

class MultiResolutionPreTrainedModel(PreTrainedModel):
    config_class = MultiResolutionConfig
    base_model_prefix = "multiresolution"
    supports_gradient_checkpointing = False

class DiscriminatorR(nn.Module):
    def __init__(self, n_fft, hop_len, win_len, channel_mult_factor, use_spectral_norm=False, lrelu_slope=0.1):
        super().__init__()

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.use_spectral_norm = use_spectral_norm
        self.lrelu_slope = lrelu_slope

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.d_mult = channel_mult_factor

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_len, win_len = self.n_fft, self.hop_len, self.win_len
        x = F.pad(
            x,
            (int((n_fft - hop_len) / 2), int((n_fft - hop_len) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_len,
            win_length=win_len,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(MultiResolutionPreTrainedModel):
    def __init__(self, config: MultiResolutionConfig):
        super().__init__(config)
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    n_fft=n_fft,
                    hop_len=hop_len,
                    win_len=win_len,
                    channel_mult=config.channel_mult,
                    use_spectral_norm=config.use_spectral_norm,
                    lrelu_slope=config.lrelu_slope
                )
                for n_fft, hop_len, win_len in config.resolutions
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs