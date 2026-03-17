# coding=utf-8
"""Generator loss for vocoder training."""

from typing import List, Tuple
import torch
from torch import nn
from ...base.objective import BaseObjective


class GeneratorLoss(BaseObjective):
    """
    Generator adversarial loss.

    Computes the loss for the generator to fool the discriminator.
    Uses least squares loss: (D(x) - 1)^2
    """

    def __init__(self):
        super().__init__(config=None)

    def forward(
        self,
        disc_outputs: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute generator loss.

        Args:
            disc_outputs: List of discriminator outputs for fake audio.

        Returns:
            Tuple of (total_loss, individual_losses).
        """
        loss = 0
        losses = []

        for d in disc_outputs:
            # Generator wants discriminator to output 1 (real)
            l = torch.mean((d - 1) ** 2)
            losses.append(l)
            loss += l

        if len(losses) > 0:
            loss = loss / len(losses)

        return loss, losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for GAN training.

    Computes L1 distance between intermediate features
    of real and fake samples in the discriminator.
    """

    def __init__(self, weight: float = 2.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        fmap_r: List[torch.Tensor],
        fmap_g: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            fmap_r: Features from real samples.
            fmap_g: Features from fake/generated samples.

        Returns:
            Feature matching loss.
        """
        loss = 0
        for r, g in zip(fmap_r, fmap_g):
            loss += nn.functional.l1_loss(r.detach(), g)
        return self.weight * loss
