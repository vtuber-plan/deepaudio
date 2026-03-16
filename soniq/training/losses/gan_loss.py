# coding=utf-8
"""
Losses for GAN-based vocoder training.

This module contains:
- Discriminator Loss
- Generator Loss
- Feature Matching Loss
- Mel Spectrogram Loss
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Any, Optional, Tuple


# ============================================================================
# Discriminator Losses
# ============================================================================

class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for GAN training.

    Uses least squares loss (LSGAN) by default:
    - Real: (1 - D(x_real))^2
    - Fake: D(x_fake)^2
    """

    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.use_lsgan = use_lsgan

    def forward(
        self,
        scores_real: List[torch.Tensor],
        scores_fake: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss.

        Args:
            scores_real: List of real scores from discriminators.
            scores_fake: List of fake scores from discriminators.

        Returns:
            Dictionary containing total loss and individual losses.
        """
        total_loss = 0.0
        losses = []

        for dr, dg in zip(scores_real, scores_fake):
            if self.use_lsgan:
                # Least squares loss
                loss_real = F.mse_loss(dr, torch.ones_like(dr))
                loss_fake = F.mse_loss(dg, torch.zeros_like(dg))
            else:
                # BCE loss
                loss_real = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr))
                loss_fake = F.binary_cross_entropy_with_logits(dg, torch.zeros_like(dg))

            loss = (loss_real + loss_fake) / 2
            total_loss = total_loss + loss
            losses.append(loss)

        return {
            "loss_disc": total_loss,
            "loss_disc_real": [l.detach() for l in losses[:len(scores_real)]],
            "loss_disc_fake": [l.detach() for l in losses[len(scores_real):]],
        }


# ============================================================================
# Generator Losses
# ============================================================================

class GeneratorLoss(nn.Module):
    """
    Generator loss for GAN training.

    Uses least squares loss (LSGAN) by default:
    - Generator: (1 - D(x_fake))^2
    """

    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.use_lsgan = use_lsgan

    def forward(
        self,
        scores_fake: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator loss.

        Args:
            scores_fake: List of fake scores from discriminators.

        Returns:
            Dictionary containing total loss and individual losses.
        """
        total_loss = 0.0
        losses = []

        for dg in scores_fake:
            if self.use_lsgan:
                # Least squares loss
                loss = F.mse_loss(dg, torch.ones_like(dg))
            else:
                # BCE loss
                loss = F.binary_cross_entropy_with_logits(dg, torch.ones_like(dg))

            total_loss = total_loss + loss
            losses.append(loss)

        return {
            "loss_gen": total_loss,
            "loss_gen_individual": [l.detach() for l in losses],
        }


# ============================================================================
# Feature Matching Loss
# ============================================================================

class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for GAN training.

    Computes L1 distance between real and fake feature maps from
    intermediate layers of the discriminator.

    Args:
        feat_weight: Weight for feature matching loss.
    """

    def __init__(self, feat_weight: float = 2.0):
        super().__init__()
        self.feat_weight = feat_weight

    def forward(
        self,
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature matching loss.

        Args:
            features_real: List of feature maps from real audio.
            features_fake: List of feature maps from fake audio.

        Returns:
            Dictionary containing feature matching loss.
        """
        total_loss = 0.0
        n_features = 0

        for fr_list, ff_list in zip(features_real, features_fake):
            for fr, ff in zip(fr_list, ff_list):
                total_loss = total_loss + F.l1_loss(fr.detach(), ff)
                n_features += 1

        if n_features > 0:
            total_loss = total_loss / n_features

        return {
            "loss_feat": total_loss * self.feat_weight,
        }


# ============================================================================
# Mel Spectrogram Loss
# ============================================================================

class MelSpectrogramLoss(nn.Module):
    """
    Mel spectrogram reconstruction loss.

    Computes L1 distance between real and generated mel spectrograms.

    Args:
        mel_weight: Weight for mel loss.
    """

    def __init__(self, mel_weight: float = 45.0):
        super().__init__()
        self.mel_weight = mel_weight

    def forward(
        self,
        mel_real: torch.Tensor,
        mel_fake: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mel spectrogram loss.

        Args:
            mel_real: Real mel spectrogram.
            mel_fake: Generated mel spectrogram.

        Returns:
            Dictionary containing mel loss.
        """
        loss = F.l1_loss(mel_real, mel_fake)

        return {
            "loss_mel": loss * self.mel_weight,
        }


# ============================================================================
# Combined GAN Loss
# ============================================================================

class CombinedGANLoss(nn.Module):
    """
    Combined loss for GAN vocoder training.

    Includes:
    - Discriminator loss
    - Generator loss
    - Feature matching loss
    - Mel spectrogram loss (optional)
    """

    def __init__(
        self,
        use_lsgan: bool = True,
        feat_weight: float = 2.0,
        mel_weight: float = 45.0,
    ):
        super().__init__()
        self.disc_loss = DiscriminatorLoss(use_lsgan)
        self.gen_loss = GeneratorLoss(use_lsgan)
        self.feat_loss = FeatureMatchingLoss(feat_weight)
        self.mel_loss = MelSpectrogramLoss(mel_weight)

    def forward(
        self,
        scores_real: List[torch.Tensor],
        scores_fake: List[torch.Tensor],
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
        mel_real: Optional[torch.Tensor] = None,
        mel_fake: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute combined GAN loss.

        Args:
            scores_real: Real scores from discriminators.
            scores_fake: Fake scores from discriminators.
            features_real: Feature maps from real audio.
            features_fake: Feature maps from fake audio.
            mel_real: Real mel spectrogram (optional).
            mel_fake: Generated mel spectrogram (optional).

        Returns:
            Dictionary containing all losses.
        """
        # Discriminator loss
        disc_out = self.disc_loss(scores_real, scores_fake)

        # Generator loss
        gen_out = self.gen_loss(scores_fake)

        # Feature matching loss
        feat_out = self.feat_loss(features_real, features_fake)

        # Mel loss
        mel_out = {}
        if mel_real is not None and mel_fake is not None:
            mel_out = self.mel_loss(mel_real, mel_fake)

        # Combine all losses
        total_disc_loss = disc_out["loss_disc"]
        total_gen_loss = (
            gen_out["loss_gen"] +
            feat_out["loss_feat"] +
            mel_out.get("loss_mel", torch.tensor(0.0))
        )

        return {
            "loss_disc": total_disc_loss,
            "loss_gen": total_gen_loss,
            "loss_gen_adv": gen_out["loss_gen"],
            "loss_feat": feat_out["loss_feat"],
            "loss_mel": mel_out.get("loss_mel", torch.tensor(0.0)),
            "discriminator": disc_out,
            "generator": gen_out,
            "feature_matching": feat_out,
            "mel": mel_out,
        }
