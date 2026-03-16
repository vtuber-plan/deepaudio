# coding=utf-8
"""
Discriminator loss for GAN vocoders.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class DiscriminatorLoss:
    """
    Discriminator loss for GAN-based vocoders.

    Uses least squares loss for stable training.
    """

    def __init__(self, lambda_real: float = 1.0, lambda_fake: float = 1.0):
        """
        Initialize DiscriminatorLoss.

        Args:
            lambda_real: Weight for real audio loss.
            lambda_fake: Weight for fake audio loss.
        """
        self.lambda_real = lambda_real
        self.lambda_fake = lambda_fake

    def __call__(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.

        Args:
            real_outputs: Discriminator outputs for real audio.
            fake_outputs: Discriminator outputs for generated audio.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        loss_real = 0.0
        loss_fake = 0.0
        num_real = 0
        num_fake = 0

        for r_out in real_outputs:
            if isinstance(r_out, (list, tuple)):
                r_out = r_out[-1]
            loss_real += F.mse_loss(r_out, torch.ones_like(r_out))
            num_real += 1

        for f_out in fake_outputs:
            if isinstance(f_out, (list, tuple)):
                f_out = f_out[-1]
            loss_fake += F.mse_loss(f_out, torch.zeros_like(f_out))
            num_fake += 1

        if num_real > 0:
            loss_real = loss_real / num_real
        if num_fake > 0:
            loss_fake = loss_fake / num_fake

        total_loss = self.lambda_real * loss_real + self.lambda_fake * loss_fake

        loss_dict = {
            "loss/d_real": loss_real.item() if num_real > 0 else 0.0,
            "loss/d_fake": loss_fake.item() if num_fake > 0 else 0.0,
            "loss/d_total": total_loss.item(),
        }

        return total_loss, loss_dict
