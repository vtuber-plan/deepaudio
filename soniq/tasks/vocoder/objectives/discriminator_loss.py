# coding=utf-8
"""Discriminator loss for vocoder training."""

from typing import List, Tuple
import torch
from torch import nn
from soniq.tasks.base.objective import BaseObjective


class DiscriminatorLoss(BaseObjective):
    """
    Discriminator adversarial loss.

    Uses least squares loss:
    - Real samples: (D(x) - 1)^2
    - Fake samples: D(x)^2
    """

    def __init__(self):
        super().__init__(config=None)

    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute discriminator loss.

        Args:
            disc_real_outputs: Discriminator outputs for real audio.
            disc_fake_outputs: Discriminator outputs for fake audio.

        Returns:
            Tuple of (total_loss, losses_real, losses_fake).
        """
        loss = 0
        losses_r = []
        losses_f = []

        for dr, df in zip(disc_real_outputs, disc_fake_outputs):
            # Real samples should be classified as 1
            l_r = torch.mean((dr - 1) ** 2)
            # Fake samples should be classified as 0
            l_f = torch.mean(df ** 2)
            losses_r.append(l_r)
            losses_f.append(l_f)
            loss += l_r + l_f

        if len(losses_r) > 0:
            loss = loss / len(losses_r)

        return loss, losses_r, losses_f
