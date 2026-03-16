"""Loss functions for Soniq."""

from .generator_loss import GeneratorLoss
from .discriminator_loss import DiscriminatorLoss
from .feature_loss import FeatureLoss
from .gan_loss import (
    DiscriminatorLoss as GANDiscriminatorLoss,
    GeneratorLoss as GANGeneratorLoss,
    FeatureMatchingLoss,
    MelSpectrogramLoss,
    CombinedGANLoss,
)

__all__ = [
    "GeneratorLoss",
    "DiscriminatorLoss",
    "FeatureLoss",
    "GANDiscriminatorLoss",
    "GANGeneratorLoss",
    "FeatureMatchingLoss",
    "MelSpectrogramLoss",
    "CombinedGANLoss",
]
