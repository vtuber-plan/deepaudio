# coding=utf-8
"""HiFiGAN vocoder implementation."""

from .configuration_hifigan import HifiGANConfig
from .modeling_hifigan import HifiGAN, HifiGANResBlock

__all__ = ["HifiGANConfig", "HifiGAN", "HifiGANResBlock"]
