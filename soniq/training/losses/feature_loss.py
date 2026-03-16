# coding=utf-8
"""
Feature matching loss for GAN vocoders.
"""

import torch
import torch.nn.functional as F
from typing import List


class FeatureLoss:
    """
    Feature matching loss for GAN training.

    This loss encourages the generator to produce features that match
    the features of real audio in the discriminator.
    """

    def __init__(
        self,
        lambda_fm: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize FeatureLoss.

        Args:
            lambda_fm: Weight for feature matching loss.
            reduction: Reduction method ("mean" or "sum").
        """
        self.lambda_fm = lambda_fm
        self.reduction = reduction

    def __call__(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: Features from discriminator for real audio.
            fake_features: Features from discriminator for generated audio.

        Returns:
            Feature matching loss.
        """
        loss = 0.0
        num_layers = 0

        for r_layer_feats, f_layer_feats in zip(real_features, fake_features):
            for r_feat, f_feat in zip(r_layer_feats, f_layer_feats):
                loss += F.l1_loss(r_feat.detach(), f_feat, reduction=self.reduction)
                num_layers += 1

        if num_layers > 0:
            loss = loss / num_layers

        return self.lambda_fm * loss
