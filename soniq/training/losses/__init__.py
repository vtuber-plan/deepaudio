"""Loss functions for Soniq."""

from .generator_loss import GeneratorLoss
from .discriminator_loss import DiscriminatorLoss
from .feature_loss import FeatureLoss

__all__ = ["GeneratorLoss", "DiscriminatorLoss", "FeatureLoss"]
