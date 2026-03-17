# coding=utf-8
"""Vocoder objectives."""

from .generator_loss import GeneratorLoss, FeatureMatchingLoss
from .discriminator_loss import DiscriminatorLoss

__all__ = [
    "GeneratorLoss",
    "FeatureMatchingLoss",
    "DiscriminatorLoss",
]