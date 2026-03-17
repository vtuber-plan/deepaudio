# coding=utf-8
"""Vocos vocoder training system."""

from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
from dataclasses import dataclass

from ..base.system import BaseTaskSystem, StepOutput
from soniq.models.vocoders.vocos.modeling_vocos import Vocos
from soniq.models.vocoders.vocos.discriminator import (
    HiFiGANMultiPeriodDiscriminator,
    SpecDiscriminator,
)
from soniq.models.vocoders.vocos.loss import VocosLoss


@dataclass
class VocosConfig:
    """Configuration for Vocos vocoder training.

    Attributes:
        learning_rate: Learning rate
        betas: Adam optimizer betas
        lr_decay: Learning rate decay gamma
        segment_size: Training segment size
        lambda_mel: Weight for mel loss
        lambda_adv: Weight for adversarial loss
        lambda_fm: Weight for feature matching loss
        n_mels: Number of mel bins
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_size: Hop size
    """
    learning_rate: float = 2e-4
    betas: tuple = (0.8, 0.99)
    lr_decay: float = 0.999
    segment_size: int = 8192
    lambda_mel: float = 10.0
    lambda_adv: float = 2.0
    lambda_fm: float = 2.0
    n_mels: int = 128
    sample_rate: int = 24000
    n_fft: int = 800
    hop_size: int = 200


class VocosTaskSystem(BaseTaskSystem):
    """
    Task system for training Vocos vocoder.

    This system handles:
    - Generator (Vocos) and discriminator training
    - Multi-period and spectrogram discriminators
    - Mel spectrogram reconstruction loss
    - Adversarial loss and feature matching loss

    Example:
        ```python
        config = VocosConfig()
        vocos = Vocos(input_channels=128, dim=384, num_layers=8)
        system = VocosTaskSystem(config, vocos)

        optimizer = system.configure_optimizers()
        # Training loop handles the rest
        ```
    """

    def __init__(
        self,
        config: Any,
        generator: Optional[nn.Module] = None,
        period_discriminator: Optional[nn.Module] = None,
        spec_discriminator: Optional[nn.Module] = None,
    ):
        """
        Initialize VocosTaskSystem.

        Args:
            config: Configuration object with training parameters
            generator: Vocos generator model
            period_discriminator: Multi-period discriminator
            spec_discriminator: Spectrogram discriminator
        """
        super().__init__(config)

        self.generator = generator
        self.period_discriminator = period_discriminator
        self.spec_discriminator = spec_discriminator

        if self.generator is None:
            raise ValueError("Generator must be provided")

        if self.period_discriminator is None:
            self.period_discriminator = HiFiGANMultiPeriodDiscriminator(
                periods=[2, 3, 5, 7, 11]
            )

        if self.spec_discriminator is None:
            self.spec_discriminator = SpecDiscriminator()

        # Loss function
        self.loss_fn = VocosLoss(
            sample_rate=getattr(config, "sample_rate", 24000),
            mel_loss_weight=getattr(config, "lambda_mel", 10.0),
            adv_loss_weight=getattr(config, "lambda_adv", 2.0),
            fm_loss_weight=getattr(config, "lambda_fm", 2.0),
        )

        # Training state
        self._train_discriminator = True

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> StepOutput:
        """
        Execute one training step.

        For GAN training, we alternate between generator and discriminator:
        - optimizer_idx=0: Train discriminator
        - optimizer_idx=1: Train generator

        Args:
            batch: Mini-batch with 'audio' and 'mel' keys
            batch_idx: Batch index
            optimizer_idx: 0 for discriminator, 1 for generator

        Returns:
            StepOutput with loss and metrics
        """
        audio = batch["audio"]  # (batch, 1, time) or (batch, time)
        mel = batch["mel"]      # (batch, n_mel, time)

        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Generate audio from mel
        audio_hat = self.generator(mel)  # (batch, 1, time)

        # Ensure audio_hat has same length as audio
        if audio_hat.shape[-1] != audio.shape[-1]:
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            audio = audio[:, :, :min_len]
            audio_hat = audio_hat[:, :, :min_len]

        metrics = {}

        if optimizer_idx == 0:
            # Train discriminator
            loss, metrics = self._train_discriminator_step(audio, audio_hat)
            self._train_discriminator = True
        else:
            # Train generator
            loss, metrics = self._train_generator_step(audio, audio_hat, mel)
            self._train_discriminator = False

        return StepOutput(loss=loss, metrics=metrics, logs={})

    def _train_discriminator_step(
        self,
        audio: torch.Tensor,
        audio_hat: torch.Tensor,
    ) -> tuple:
        """
        Train discriminators.

        Args:
            audio: Real audio
            audio_hat: Generated audio

        Returns:
            Tuple of (loss, metrics)
        """
        # Multi-period discriminator
        period_real_outputs = self.period_discriminator(audio)
        period_fake_outputs = self.period_discriminator(audio_hat.detach())

        # Spectrogram discriminator
        spec_real_outputs = self.spec_discriminator(audio)
        spec_fake_outputs = self.spec_discriminator(audio_hat.detach())

        # Compute discriminator loss
        disc_loss_dict = self.loss_fn.discriminator_loss(
            period_real_outputs + spec_real_outputs,
            period_fake_outputs + spec_fake_outputs,
        )

        loss = disc_loss_dict["loss"]

        metrics = {
            "loss_disc": loss.item(),
            "loss_disc_real": disc_loss_dict["real_loss"].item() if isinstance(disc_loss_dict["real_loss"], torch.Tensor) else disc_loss_dict["real_loss"],
            "loss_disc_fake": disc_loss_dict["fake_loss"].item() if isinstance(disc_loss_dict["fake_loss"], torch.Tensor) else disc_loss_dict["fake_loss"],
        }

        return loss, metrics

    def _train_generator_step(
        self,
        audio: torch.Tensor,
        audio_hat: torch.Tensor,
        mel: torch.Tensor,
    ) -> tuple:
        """
        Train generator.

        Args:
            audio: Real audio
            audio_hat: Generated audio
            mel: Mel spectrogram

        Returns:
            Tuple of (loss, metrics)
        """
        # Multi-period discriminator
        period_real_outputs = self.period_discriminator(audio)
        period_fake_outputs = self.period_discriminator(audio_hat)

        # Spectrogram discriminator
        spec_real_outputs = self.spec_discriminator(audio)
        spec_fake_outputs = self.spec_discriminator(audio_hat)

        # Compute generator loss
        gen_loss_dict = self.loss_fn.generator_loss(
            pred_audio=audio_hat,
            target_audio=audio,
            real_outputs=period_real_outputs + spec_real_outputs,
            fake_outputs=period_fake_outputs + spec_fake_outputs,
        )

        loss = gen_loss_dict["total_loss"]

        metrics = {
            "loss_gen": loss.item(),
            "loss_mel": gen_loss_dict["mel_loss"].item(),
            "loss_adv": gen_loss_dict["adv_loss"].item(),
            "loss_fm": gen_loss_dict["fm_loss"].item(),
        }

        return loss, metrics

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> StepOutput:
        """
        Execute one validation step.

        Only computes generator metrics (no discriminator training).
        """
        audio = batch["audio"]
        mel = batch["mel"]

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        with torch.no_grad():
            audio_hat = self.generator(mel)

            # Ensure same length
            if audio_hat.shape[-1] != audio.shape[-1]:
                min_len = min(audio.shape[-1], audio_hat.shape[-1])
                audio = audio[:, :, :min_len]
                audio_hat = audio_hat[:, :, :min_len]

            # Compute mel loss
            mel_loss = self.loss_fn.mel_loss(audio_hat, audio)

            # Get discriminator scores for evaluation
            with torch.no_grad():
                period_fake_outputs = self.period_discriminator(audio_hat)
                spec_fake_outputs = self.spec_discriminator(audio_hat)

        metrics = {
            "val_loss_mel": mel_loss.item(),
            "val_period_disc_score": torch.mean(period_fake_outputs[0][-1]).item(),
            "val_spec_disc_score": torch.mean(spec_fake_outputs[0][-1]).item(),
        }

        return StepOutput(loss=mel_loss, metrics=metrics, logs={})

    def inference_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one inference step.

        Returns:
            Dictionary with generated audio
        """
        mel = batch["mel"]

        with torch.no_grad():
            audio_hat = self.generator(mel)

        return {"audio_hat": audio_hat}

    def configure_optimizers(
        self
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Configure optimizers for generator and discriminator.

        Returns:
            Dictionary with 'generator' and 'discriminator' optimizers
        """
        opt_g = AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        # Combine discriminator parameters
        disc_params = list(self.period_discriminator.parameters()) + \
                      list(self.spec_discriminator.parameters())

        opt_d = AdamW(
            disc_params,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        return {
            "generator": opt_g,
            "discriminator": opt_d,
        }

    def configure_schedulers(
        self
    ) -> Dict[str, LRScheduler]:
        """Configure learning rate schedulers."""
        scheduler_g = ExponentialLR(
            self.optimizers["generator"],
            gamma=self.config.lr_decay,
        )
        scheduler_d = ExponentialLR(
            self.optimizers["discriminator"],
            gamma=self.config.lr_decay,
        )

        return {
            "generator": scheduler_g,
            "discriminator": scheduler_d,
        }

    def on_train_start(self) -> None:
        """Called when training starts."""
        self.generator.train()
        self.period_discriminator.train()
        self.spec_discriminator.train()

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(self) -> None:
        """Called when training ends."""
        self.generator.eval()
        self.period_discriminator.eval()
        self.spec_discriminator.eval()


__all__ = ["VocosTaskSystem", "VocosConfig"]
