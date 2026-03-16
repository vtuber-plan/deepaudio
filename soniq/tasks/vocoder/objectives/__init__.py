# coding=utf-8
"""Vocoder objectives."""

from soniq.tasks.vocoder.objectives.generator_loss import GeneratorLoss, FeatureMatchingLoss
from soniq.tasks.vocoder.objectives.discriminator_loss import DiscriminatorLoss

__all__ = [
    "GeneratorLoss",
    "FeatureMatchingLoss",
    "DiscriminatorLoss",
]
