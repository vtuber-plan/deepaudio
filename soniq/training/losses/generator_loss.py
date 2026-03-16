# coding=utf-8
"""
Loss functions for GAN vocoders in Soniq.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class GeneratorLoss:
    """
    Generator loss for GAN-based vocoders.

    This includes:
        - Adversarial loss
        - Feature matching loss
        - Mel spectrogram loss
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_fm: float = 2.0,
        lambda_mel: float = 45.0,
    ):
        """
        Initialize GeneratorLoss.

        Args:
            lambda_adv: Weight for adversarial loss.
            lambda_fm: Weight for feature matching loss.
            lambda_mel: Weight for mel spectrogram loss.
        """
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def adversarial_loss(
        self,
        discriminator_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute adversarial loss.

        Args:
            discriminator_outputs: List of discriminator outputs for generated audio.

        Returns:
            Adversarial loss.
        """
        loss = 0.0
        for d_out in discriminator_outputs:
            if isinstance(d_out, (list, tuple)):
                d_out = d_out[-1]
            loss += F.mse_loss(d_out, torch.ones_like(d_out))
        return loss / len(discriminator_outputs)

    def feature_matching_loss(
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
        for r_feats, f_feats in zip(real_features, fake_features):
            for r_feat, f_feat in zip(r_feats, f_feats):
                loss += F.l1_loss(r_feat.detach(), f_feat)
        return loss / (len(real_features) * len(real_features[0]))

    def mel_loss(
        self,
        real_mel: torch.Tensor,
        fake_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram reconstruction loss.

        Args:
            real_mel: Mel spectrogram of real audio.
            fake_mel: Mel spectrogram of generated audio.

        Returns:
            Mel reconstruction loss.
        """
        return F.l1_loss(real_mel, fake_mel)

    def __call__(
        self,
        discriminator_outputs: List[torch.Tensor],
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]],
        real_mel: torch.Tensor,
        fake_mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total generator loss.

        Args:
            discriminator_outputs: Discriminator outputs for generated audio.
            real_features: Features from discriminator for real audio.
            fake_features: Features from discriminator for generated audio.
            real_mel: Mel spectrogram of real audio.
            fake_mel: Mel spectrogram of generated audio.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        loss_adv = self.adversarial_loss(discriminator_outputs)
        loss_fm = self.feature_matching_loss(real_features, fake_features)
        loss_mel = self.mel_loss(real_mel, fake_mel)

        total_loss = (
            self.lambda_adv * loss_adv
            + self.lambda_fm * loss_fm
            + self.lambda_mel * loss_mel
        )

        loss_dict = {
            "loss/g_adv": loss_adv.item(),
            "loss/g_fm": loss_fm.item(),
            "loss/g_mel": loss_mel.item(),
            "loss/g_total": total_loss.item(),
        }

        return total_loss, loss_dict


class DiscriminatorLoss:
    """
    Discriminator loss for GAN-based vocoders.
    """

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

        for r_out in real_outputs:
            if isinstance(r_out, (list, tuple)):
                r_out = r_out[-1]
            loss_real += F.mse_loss(r_out, torch.ones_like(r_out))

        for f_out in fake_outputs:
            if isinstance(f_out, (list, tuple)):
                f_out = f_out[-1]
            loss_fake += F.mse_loss(f_out, torch.zeros_like(f_out))

        total_loss = (loss_real + loss_fake) / (
            len(real_outputs) + len(fake_outputs)
        )

        loss_dict = {
            "loss/d_real": loss_real.item() / len(real_outputs),
            "loss/d_fake": loss_fake.item() / len(fake_outputs),
            "loss/d_total": total_loss.item(),
        }

        return total_loss, loss_dict


class FeatureLoss:
    """
    Feature matching loss for GAN training.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: Features for real audio.
            fake_features: Features for generated audio.

        Returns:
            Feature matching loss.
        """
        loss = 0.0
        for r_feat, f_feat in zip(real_features, fake_features):
            loss += F.l1_loss(r_feat.detach(), f_feat, reduction=self.reduction)
        return loss / len(real_features)
