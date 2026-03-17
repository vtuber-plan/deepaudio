# coding=utf-8
"""Vocos vocoder - Modern efficient vocoder based on ConvNeXt + ISTFT."""

from soniq.models.vocoders.vocos.configuration_vocos import VocosConfig
from soniq.models.vocoders.vocos.modeling_vocos import (
    Vocos,
    VocosBackbone,
    ISTFTHead,
    ISTFT,
    ConvNeXtBlock,
    AdaLayerNorm,
)
from soniq.models.vocoders.vocos.discriminator import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    SpecDiscriminator,
    NLayerSpecDiscriminator,
)
from soniq.models.vocoders.vocos.loss import (
    MelSpectrogramLoss,
    GANLoss,
    FeatureMatchingLoss,
    VocosLoss,
)

__all__ = [
    # Config
    "VocosConfig",
    # Model
    "Vocos",
    "VocosBackbone",
    "ISTFTHead",
    "ISTFT",
    "ConvNeXtBlock",
    "AdaLayerNorm",
    # Discriminators
    "HiFiGANMultiPeriodDiscriminator",
    "HiFiGANPeriodDiscriminator",
    "SpecDiscriminator",
    "NLayerSpecDiscriminator",
    # Losses
    "MelSpectrogramLoss",
    "GANLoss",
    "FeatureMatchingLoss",
    "VocosLoss",
]
