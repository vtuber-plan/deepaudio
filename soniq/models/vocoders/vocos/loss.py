# coding=utf-8
"""Loss functions for Vocos vocoder."""

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F

try:
    from torchaudio.transforms import MelSpectrogram
except ImportError:
    MelSpectrogram = None


class MelSpectrogramLoss(nn.Module):
    """Multi-resolution mel spectrogram loss.

    Args:
        sample_rate: Audio sample rate
        n_mels: List of number of mel bins
        window_lengths: List of window lengths
        clamp_eps: Clamp epsilon for log stability
        mag_weight: Weight for linear magnitude loss
        log_weight: Weight for log magnitude loss
        power: Power for magnitude
        mel_fmin: List of minimum mel frequencies
        mel_fmax: List of maximum mel frequencies
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [80, 128],
        window_lengths: List[int] = [512, 2048],
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        power: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[Optional[float]] = [None, None],
    ):
        super().__init__()
        if MelSpectrogram is None:
            raise ImportError("torchaudio is required for MelSpectrogramLoss")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_lengths = window_lengths
        self.clamp_eps = clamp_eps
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.power = power
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        # Build mel transforms
        self.mel_transforms = nn.ModuleList()
        for n_mel, win_len, fmin, fmax in zip(
            n_mels, window_lengths, mel_fmin, mel_fmax
        ):
            self.mel_transforms.append(
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=win_len,
                    hop_length=win_len // 4,
                    win_length=win_len,
                    n_mels=n_mel,
                    power=power,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                    f_min=fmin,
                    f_max=fmax,
                )
            )

        self.loss_fn = nn.L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute mel loss.

        Args:
            pred: Predicted audio (B, T) or (B, 1, T)
            target: Target audio (B, T) or (B, 1, T)

        Returns:
            Mel loss value
        """
        # Ensure 2D input
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        loss = 0.0
        for mel_fn in self.mel_transforms:
            pred_mel = mel_fn(pred)
            target_mel = mel_fn(target)

            # Log mel loss
            if self.log_weight > 0:
                log_pred = pred_mel.clamp(self.clamp_eps).pow(self.power).log10()
                log_target = target_mel.clamp(self.clamp_eps).pow(self.power).log10()
                loss += self.log_weight * self.loss_fn(log_pred, log_target)

            # Linear mel loss
            if self.mag_weight > 0:
                loss += self.mag_weight * self.loss_fn(pred_mel, target_mel)

        return loss


class GANLoss(nn.Module):
    """GAN loss for discriminator and generator.

    Args:
        mode: Loss mode ("lsgan", "hinge")
    """

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        assert mode in ["lsgan", "hinge"]
        self.mode = mode

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple:
        """Compute discriminator loss.

        Args:
            real: Real predictions
            fake: Fake predictions

        Returns:
            (real_loss, fake_loss, total_loss)
        """
        if self.mode == "lsgan":
            real_loss = F.mse_loss(real, torch.ones_like(real))
            fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        elif self.mode == "hinge":
            real_loss = torch.relu(1.0 - real).mean()
            fake_loss = torch.relu(1.0 + fake).mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        total_loss = real_loss + fake_loss
        return real_loss, fake_loss, total_loss

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Compute generator loss.

        Args:
            fake: Fake predictions

        Returns:
            Generator loss
        """
        if self.mode == "lsgan":
            return F.mse_loss(fake, torch.ones_like(fake))
        elif self.mode == "hinge":
            return -fake.mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN training.

    Computes L1 distance between intermediate features of real and generated.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(
        self, real_features: List[torch.Tensor], fake_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature matching loss.

        Args:
            real_features: List of real intermediate features
            fake_features: List of fake intermediate features

        Returns:
            Feature matching loss
        """
        loss = 0.0
        for real_f, fake_f in zip(real_features, fake_features):
            loss += self.loss_fn(real_f, fake_f)
        return loss


class VocosLoss(nn.Module):
    """Combined loss for Vocos vocoder.

    Args:
        sample_rate: Audio sample rate
        mel_loss_weight: Weight for mel loss
        adv_loss_weight: Weight for adversarial loss
        fm_loss_weight: Weight for feature matching loss
        gan_mode: GAN loss mode
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        mel_loss_weight: float = 10.0,
        adv_loss_weight: float = 2.0,
        fm_loss_weight: float = 2.0,
        gan_mode: str = "lsgan",
    ):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.fm_loss_weight = fm_loss_weight

        # Mel loss (multi-resolution)
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_mels=[80, 128],
            window_lengths=[512, 2048],
        )

        # GAN loss
        self.gan_loss = GANLoss(mode=gan_mode)

        # Feature matching loss
        self.fm_loss = FeatureMatchingLoss()

    def discriminator_loss(
        self,
        real_outputs: List[List[torch.Tensor]],
        fake_outputs: List[List[torch.Tensor]],
    ) -> dict:
        """Compute total discriminator loss.

        Args:
            real_outputs: Discriminator outputs for real audio
            fake_outputs: Discriminator outputs for fake audio

        Returns:
            Loss dict with 'loss', 'real_loss', 'fake_loss'
        """
        total_real_loss = 0.0
        total_fake_loss = 0.0

        for real_d, fake_d in zip(real_outputs, fake_outputs):
            # Last element is the final prediction
            real_pred = real_d[-1]
            fake_pred = fake_d[-1]

            _, _, loss = self.gan_loss.discriminator_loss(real_pred, fake_pred)
            total_real_loss += loss
            total_fake_loss += loss

        return {
            "loss": total_real_loss + total_fake_loss,
            "real_loss": total_real_loss,
            "fake_loss": total_fake_loss,
        }

    def generator_loss(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        real_outputs: List[List[torch.Tensor]],
        fake_outputs: List[List[torch.Tensor]],
    ) -> dict:
        """Compute total generator loss.

        Args:
            pred_audio: Generated audio
            target_audio: Target audio
            real_outputs: Discriminator outputs for real audio
            fake_outputs: Discriminator outputs for fake audio

        Returns:
            Loss dict with 'loss', 'mel_loss', 'adv_loss', 'fm_loss'
        """
        losses = {}

        # Mel loss
        mel_loss = self.mel_loss(pred_audio, target_audio)
        losses["mel_loss"] = mel_loss * self.mel_loss_weight

        # Adversarial loss
        adv_loss = 0.0
        for fake_d in fake_outputs:
            adv_loss += self.gan_loss.generator_loss(fake_d[-1])
        losses["adv_loss"] = adv_loss * self.adv_loss_weight

        # Feature matching loss
        fm_loss = 0.0
        for real_d, fake_d in zip(real_outputs, fake_outputs):
            # Use all intermediate features except the last
            real_features = real_d[:-1]
            fake_features = fake_d[:-1]
            fm_loss += self.fm_loss(real_features, fake_features)
        losses["fm_loss"] = fm_loss * self.fm_loss_weight

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses


__all__ = [
    "MelSpectrogramLoss",
    "GANLoss",
    "FeatureMatchingLoss",
    "VocosLoss",
]
