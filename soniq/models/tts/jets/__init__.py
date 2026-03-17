# coding=utf-8
"""Jets: End-to-End Non-Autoregressive TTS."""

from soniq.models.tts.jets.modeling_jets import (
    Jets,
    TextEncoder,
    Decoder,
    VariancePredictor,
    LengthRegulator,
    PostNet,
    AlignmentModule,
    get_mask_from_lengths,
)
from soniq.models.tts.jets.loss import (
    MelLoss,
    DurationLoss,
    VarianceLoss,
    ForwardSumLoss,
    BinarizationLoss,
    GeneratorAdversarialLoss,
    DiscriminatorAdversarialLoss,
    FeatureMatchingLoss,
    JetsLoss,
)

__all__ = [
    # Model
    "Jets",
    "TextEncoder",
    "Decoder",
    "VariancePredictor",
    "LengthRegulator",
    "PostNet",
    "AlignmentModule",
    "get_mask_from_lengths",
    # Loss
    "MelLoss",
    "DurationLoss",
    "VarianceLoss",
    "ForwardSumLoss",
    "BinarizationLoss",
    "GeneratorAdversarialLoss",
    "DiscriminatorAdversarialLoss",
    "FeatureMatchingLoss",
    "JetsLoss",
]