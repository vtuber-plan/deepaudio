# coding=utf-8
"""BigVGAN vocoder for Soniq."""

from soniq.models.vocoders.bigvgan.configuration_bigvgan import BigVGANConfig
from soniq.models.vocoders.bigvgan.modeling_bigvgan import BigVGAN

__all__ = [
    "BigVGANConfig",
    "BigVGAN",
]
