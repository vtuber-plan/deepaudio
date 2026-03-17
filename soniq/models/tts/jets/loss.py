# coding=utf-8
"""Loss functions for Jets TTS model."""

from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


class MelLoss(nn.Module):
    """Mel spectrogram reconstruction loss."""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mel loss.

        Args:
            mel_pred: Predicted mel (B, T, n_mel)
            mel_target: Target mel (B, T, n_mel)
            mel_lengths: Optional mel lengths for masking

        Returns:
            Mel loss
        """
        if mel_lengths is not None:
            # Apply mask
            max_len = mel_pred.shape[1]
            mask = torch.arange(max_len, device=mel_pred.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)

            mel_pred = mel_pred * mask
            mel_target = mel_target * mask

            # Normalize by valid frames
            loss = F.l1_loss(mel_pred, mel_target, reduction='none')
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = self.l1_loss(mel_pred, mel_target)

        return loss


class DurationLoss(nn.Module):
    """Duration predictor loss.

    Computed in log domain.
    """

    def __init__(self, offset: float = 1.0):
        super().__init__()
        self.offset = offset
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        log_duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute duration loss.

        Args:
            log_duration_pred: Predicted log durations (B, T)
            duration_target: Target durations (B, T)
            src_mask: Source mask (B, T)

        Returns:
            Duration loss
        """
        # Convert target to log domain
        log_duration_target = torch.log(duration_target.float() + self.offset)

        if src_mask is not None:
            # Apply mask
            mask = ~src_mask
            log_duration_pred = log_duration_pred * mask
            log_duration_target = log_duration_target * mask

            loss = F.mse_loss(log_duration_pred, log_duration_target, reduction='none')
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = self.mse_loss(log_duration_pred, log_duration_target)

        return loss


class VarianceLoss(nn.Module):
    """Variance (pitch/energy) predictor loss."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute variance loss.

        Args:
            pred: Predicted values (B, T)
            target: Target values (B, T)
            mask: Optional mask (B, T)

        Returns:
            Variance loss
        """
        if mask is not None:
            mask = ~mask
            pred = pred * mask
            target = target * mask

            loss = F.mse_loss(pred, target, reduction='none')
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = self.mse_loss(pred, target)

        return loss


class ForwardSumLoss(nn.Module):
    """Forward-sum loss for alignment.

    Encourages monotonic alignment.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward-sum loss.

        Args:
            log_p_attn: Log attention probabilities (B, T_mel, T_text)
            text_lengths: Text lengths (B,)
            mel_lengths: Mel lengths (B,)

        Returns:
            Forward-sum loss
        """
        batch_size = log_p_attn.shape[0]
        device = log_p_attn.device

        loss = 0.0
        for b in range(batch_size):
            T_text = text_lengths[b].item()
            T_mel = mel_lengths[b].item()

            # Sum attention along text axis for each mel frame
            attn = log_p_attn[b, :T_mel, :T_text]

            # Forward sum: sum over all paths should be close to 1
            # This encourages each text token to be attended to
            attn_sum = torch.logsumexp(attn, dim=1).mean()
            loss -= attn_sum

        return loss / batch_size


class BinarizationLoss(nn.Module):
    """Binarization loss for alignment.

    Encourages sharp attention.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binarization loss.

        Args:
            log_p_attn: Log attention probabilities (B, T_mel, T_text)
            text_lengths: Text lengths (B,)
            mel_lengths: Mel lengths (B,)

        Returns:
            Binarization loss
        """
        batch_size = log_p_attn.shape[0]
        device = log_p_attn.device

        loss = 0.0
        for b in range(batch_size):
            T_text = text_lengths[b].item()
            T_mel = mel_lengths[b].item()

            attn = log_p_attn[b, :T_mel, :T_text]
            attn = torch.exp(attn)

            # Encourage sharp attention (entropy minimization)
            entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
            loss += entropy

        return loss / batch_size


class GeneratorAdversarialLoss(nn.Module):
    """Generator adversarial loss."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute generator adversarial loss.

        Args:
            outputs: List of discriminator outputs

        Returns:
            Adversarial loss
        """
        adv_loss = 0.0
        for output in outputs:
            if isinstance(output, (list, tuple)):
                output = output[-1]
            adv_loss += F.mse_loss(output, torch.ones_like(output))

        return adv_loss


class DiscriminatorAdversarialLoss(nn.Module):
    """Discriminator adversarial loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discriminator adversarial loss.

        Args:
            real_outputs: Discriminator outputs for real data
            fake_outputs: Discriminator outputs for fake data

        Returns:
            Tuple of (total_loss, real_loss, fake_loss)
        """
        real_loss = 0.0
        fake_loss = 0.0

        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if isinstance(real_out, (list, tuple)):
                real_out = real_out[-1]
            if isinstance(fake_out, (list, tuple)):
                fake_out = fake_out[-1]

            real_loss += F.mse_loss(real_out, torch.ones_like(real_out))
            fake_loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))

        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN training."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute feature matching loss.

        Args:
            fake_features: Feature maps from fake data
            real_features: Feature maps from real data

        Returns:
            Feature matching loss
        """
        loss = 0.0

        for fake_feats, real_feats in zip(fake_features, real_features):
            # Skip final output (discriminator score)
            fake_feats = fake_feats[:-1] if len(fake_feats) > 1 else fake_feats
            real_feats = real_feats[:-1] if len(real_feats) > 1 else real_feats

            for fake_f, real_f in zip(fake_feats, real_feats):
                loss += F.l1_loss(fake_f, real_f.detach())

        return loss


class JetsLoss(nn.Module):
    """Combined loss for Jets TTS model.

    Args:
        mel_loss_weight: Weight for mel loss
        duration_loss_weight: Weight for duration loss
        pitch_loss_weight: Weight for pitch loss
        energy_loss_weight: Weight for energy loss
        align_loss_weight: Weight for alignment loss
        adv_loss_weight: Weight for adversarial loss
        fm_loss_weight: Weight for feature matching loss
    """

    def __init__(
        self,
        mel_loss_weight: float = 45.0,
        duration_loss_weight: float = 1.0,
        pitch_loss_weight: float = 1.0,
        energy_loss_weight: float = 1.0,
        align_loss_weight: float = 2.0,
        adv_loss_weight: float = 1.0,
        fm_loss_weight: float = 2.0,
    ):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.align_loss_weight = align_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.fm_loss_weight = fm_loss_weight

        # Loss functions
        self.mel_loss = MelLoss()
        self.duration_loss = DurationLoss()
        self.pitch_loss = VarianceLoss()
        self.energy_loss = VarianceLoss()
        self.forward_sum_loss = ForwardSumLoss()
        self.binarization_loss = BinarizationLoss()
        self.gen_adv_loss = GeneratorAdversarialLoss()
        self.disc_adv_loss = DiscriminatorAdversarialLoss()
        self.fm_loss = FeatureMatchingLoss()

    def compute_generator_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator_outputs: Optional[List[List[torch.Tensor]]] = None,
        real_features: Optional[List[List[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total generator loss.

        Args:
            outputs: Model outputs
            targets: Target values
            discriminator_outputs: Discriminator outputs for generated data
            real_features: Feature maps from real data

        Returns:
            Dictionary with loss values
        """
        losses = {}

        # Mel loss
        if 'mel' in targets:
            mel_loss = self.mel_loss(
                outputs['mel_postnet'],
                targets['mel'],
                outputs.get('mel_lengths'),
            )
            losses['mel_loss'] = mel_loss * self.mel_loss_weight

        # Duration loss
        if 'duration' in targets:
            text_mask = targets.get('text_mask')
            dur_loss = self.duration_loss(
                outputs['log_duration_pred'],
                targets['duration'],
                text_mask,
            )
            losses['duration_loss'] = dur_loss * self.duration_loss_weight

        # Pitch loss
        if 'pitch' in targets:
            pitch_loss = self.pitch_loss(
                outputs['pitch_pred'],
                targets['pitch'],
                targets.get('text_mask'),
            )
            losses['pitch_loss'] = pitch_loss * self.pitch_loss_weight

        # Energy loss
        if 'energy' in targets:
            energy_loss = self.energy_loss(
                outputs['energy_pred'],
                targets['energy'],
                targets.get('text_mask'),
            )
            losses['energy_loss'] = energy_loss * self.energy_loss_weight

        # Alignment loss
        if 'log_p_attn' in outputs and 'text_lengths' in targets and 'mel_lengths' in targets:
            fs_loss = self.forward_sum_loss(
                outputs['log_p_attn'],
                targets['text_lengths'],
                targets['mel_lengths'],
            )
            bin_loss = self.binarization_loss(
                outputs['log_p_attn'],
                targets['text_lengths'],
                targets['mel_lengths'],
            )
            losses['align_loss'] = (fs_loss + bin_loss) * self.align_loss_weight

        # Adversarial loss
        if discriminator_outputs is not None:
            adv_loss = self.gen_adv_loss(discriminator_outputs)
            losses['adv_loss'] = adv_loss * self.adv_loss_weight

        # Feature matching loss
        if discriminator_outputs is not None and real_features is not None:
            fm_loss = self.fm_loss(discriminator_outputs, real_features)
            losses['fm_loss'] = fm_loss * self.fm_loss_weight

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses

    def compute_discriminator_loss(
        self,
        real_outputs: List[List[torch.Tensor]],
        fake_outputs: List[List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss.

        Args:
            real_outputs: Discriminator outputs for real data
            fake_outputs: Discriminator outputs for fake data

        Returns:
            Dictionary with loss values
        """
        total_loss, real_loss, fake_loss = self.disc_adv_loss(real_outputs, fake_outputs)

        return {
            'disc_loss': total_loss,
            'disc_real_loss': real_loss,
            'disc_fake_loss': fake_loss,
        }


__all__ = [
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