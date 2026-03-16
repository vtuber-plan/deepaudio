# coding=utf-8
"""Vocoder models."""

from soniq.models.vocoders.base import BaseVocoderModel
from soniq.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig
from soniq.models.vocoders.hifigan.modeling_hifigan import HifiGAN
from soniq.models.vocoders.hifigan.discriminator import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)
from soniq.models.vocoders.bigvgan.configuration_bigvgan import BigVGANConfig
from soniq.models.vocoders.bigvgan.modeling_bigvgan import BigVGAN

__all__ = [
    "BaseVocoderModel",
    "HifiGAN",
    "HifiGANConfig",
    "HiFiGANMultiPeriodDiscriminator",
    "HiFiGANMultiScaleDiscriminator",
    "BigVGAN",
    "BigVGANConfig",
]
