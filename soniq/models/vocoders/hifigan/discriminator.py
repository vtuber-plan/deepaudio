# coding=utf-8
"""Discriminator for HiFiGAN vocoder."""

import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm
from typing import List, Tuple, Optional
from soniq.utils.model_utils import get_padding


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator for HiFiGAN.

    This discriminator uses multiple sub-discriminators with different
    period patterns to capture fine-grained structure in audio.
    """

    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, use_spectral_norm)
            for period in periods
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_hat: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Real audio waveform of shape (batch, 1, time).
            x_hat: Fake audio waveform of shape (batch, 1, time).

        Returns:
            Tuple of (scores_real, scores_fake, features_real, features_fake).
        """
        scores_real = []
        scores_fake = []
        feats_real = []
        feats_fake = []

        for d in self.discriminators:
            score_r, feat_r = d(x)
            score_f, feat_f = d(x_hat)
            scores_real.append(score_r)
            scores_fake.append(score_f)
            feats_real.append(feat_r)
            feats_fake.append(feat_f)

        return scores_real, scores_fake, feats_real, feats_fake


class PeriodDiscriminator(nn.Module):
    """
    Period discriminator for HiFiGAN.
    """

    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period

        norm_f = weight_norm if not use_spectral_norm else nn.utils.spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (5, 1), (3, 1), padding=get_padding(5, 1))),
            norm_f(Conv2d(32, 128, (5, 1), (3, 1), padding=get_padding(5, 1))),
            norm_f(Conv2d(128, 512, (5, 1), (3, 1), padding=get_padding(5, 1))),
            norm_f(Conv2d(512, 1024, (5, 1), (3, 1), padding=get_padding(5, 1))),
            norm_f(Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Audio waveform of shape (batch, 1, time).

        Returns:
            Tuple of (score, features).
        """
        feat = []

        # 1D to 2D
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (n_pad, 0))
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)

        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class HiFiGANMultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator for HiFiGAN.

    This discriminator uses multiple sub-discriminators at different
    scales to capture both coarse and fine structure in audio.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scales: int = 3,
        downsample_factor: int = 2,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(in_channels, out_channels, use_spectral_norm)
            for _ in range(scales)
        ])
        self.downsample_factor = downsample_factor

    def forward(
        self,
        x: torch.Tensor,
        x_hat: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Real audio waveform of shape (batch, 1, time).
            x_hat: Fake audio waveform of shape (batch, 1, time).

        Returns:
            Tuple of (scores_real, scores_fake, features_real, features_fake) for each scale.
        """
        scores_real = []
        scores_fake = []
        feats_real = []
        feats_fake = []

        x_real = x
        x_fake = x_hat

        for i, d in enumerate(self.discriminators):
            score_r, feat_r = d(x_real)
            score_f, feat_f = d(x_fake)
            scores_real.append(score_r)
            scores_fake.append(score_f)
            feats_real.append(feat_r)
            feats_fake.append(feat_f)

            if i < len(self.discriminators) - 1:
                x_real = torch.nn.functional.avg_pool1d(x_real, self.downsample_factor * 2, stride=self.downsample_factor)
                x_fake = torch.nn.functional.avg_pool1d(x_fake, self.downsample_factor * 2, stride=self.downsample_factor)

        return scores_real, scores_fake, feats_real, feats_fake


class ScaleDiscriminator(nn.Module):
    """
    Scale discriminator for HiFiGAN.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else nn.utils.spectral_norm

        self.conv1 = norm_f(nn.Conv1d(in_channels, 16, 15, 1, padding=7))
        self.conv2 = norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20))
        self.conv3 = norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20))
        self.conv4 = norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20))
        self.conv5 = norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20))
        self.conv_post = norm_f(nn.Conv1d(1024, out_channels, 5, 1, padding=2))

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Audio waveform of shape (batch, 1, time).

        Returns:
            Tuple of (score, features).
        """
        feat = []

        x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        feat.append(x)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        feat.append(x)

        x = self.conv3(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        feat.append(x)

        x = self.conv4(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        feat.append(x)

        x = self.conv5(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        feat.append(x)

        x = self.conv_post(x)
        feat.append(x)

        return x, feat
