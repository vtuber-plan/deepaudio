# coding=utf-8
"""Vocoder task system for training HiFiGAN and other vocoders."""

from typing import Any, Dict, Optional, Union, List
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass

from soniq.tasks.base.system import BaseTaskSystem, StepOutput
from soniq.tasks.vocoder.objectives.generator_loss import GeneratorLoss
from soniq.tasks.vocoder.objectives.discriminator_loss import DiscriminatorLoss
from soniq.models.vocoders.hifigan import HifiGAN
from soniq.models.vocoders.hifigan.discriminator import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)


@dataclass
class VocoderConfig:
    """Configuration for vocoder training."""
    learning_rate: float = 2e-4
    betas: tuple = (0.8, 0.99)
    lr_decay: float = 0.999
    segment_size: int = 8192
    lambda_mel: float = 45.0
    lambda_mel_dict: Optional[Dict[str, float]] = None
    lambda_adv: float = 1.0
    lambda_feat_match: float = 2.0


class VocoderTaskSystem(BaseTaskSystem):
    """
    Task system for training vocoder models (e.g., HiFiGAN).

    This system handles:
    - Generator and discriminator training
    - Adversarial loss computation
    - Feature matching loss
    - Mel spectrogram reconstruction loss

    Example:
        ```python
        config = VocoderConfig()
        generator = HifiGAN(config)
        system = VocoderTaskSystem(config, generator)

        optimizer = system.configure_optimizers()
        # Training loop handles the rest
        ```
    """

    def __init__(
        self,
        config: Any,
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        use_mel_loss: bool = True,
    ):
        """
        Initialize VocoderTaskSystem.

        Args:
            config: Configuration object with training parameters.
            generator: Generator model (vocoder).
            discriminator: Discriminator model. If None, creates default.
            use_mel_loss: Whether to use mel spectrogram reconstruction loss.
        """
        super().__init__(config)

        self.generator = generator
        self.discriminator = discriminator

        if self.generator is None:
            raise ValueError("Generator must be provided")

        if self.discriminator is None:
            # Create default discriminators
            self.discriminator_mp = HiFiGANMultiPeriodDiscriminator()
            self.discriminator_ms = HiFiGANMultiScaleDiscriminator()
        else:
            self.discriminator_mp = self.discriminator
            self.discriminator_ms = None

        # Loss functions
        self.use_mel_loss = use_mel_loss
        if use_mel_loss:
            self.mel_loss_fn = nn.L1Loss()

        self.generator_loss_fn = GeneratorLoss()
        self.discriminator_loss_fn = DiscriminatorLoss()

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
            batch: Mini-batch with 'audio' and 'mel' keys.
            batch_idx: Batch index.
            optimizer_idx: 0 for discriminator, 1 for generator.

        Returns:
            StepOutput with loss and metrics.
        """
        audio = batch["audio"]  # (batch, 1, time) or (batch, time)
        mel = batch["mel"]      # (batch, n_mel, time)

        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Generate audio from mel
        audio_hat = self.generator(mel)
        # Extract waveform from VocoderOutput
        audio_hat = audio_hat.waveform

        # Ensure audio_hat has same length as audio (handle padding mismatches)
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
        """Train discriminator."""
        # Multi-period discriminator
        y_df_hat_r, y_df_hat_g, _, _ = self.discriminator_mp(audio, audio_hat)
        loss_disc_periodic, losses_disc_periodic_r, losses_disc_periodic_g = \
            self.discriminator_loss_fn(y_df_hat_r, y_df_hat_g)

        # Multi-scale discriminator
        loss_disc_scale = 0
        losses_disc_scale_r = []
        losses_disc_scale_g = []

        if self.discriminator_ms is not None:
            y_ds_hat_r, y_ds_hat_g, _, _ = self.discriminator_ms(audio, audio_hat)
            loss_disc_scale, losses_disc_scale_r, losses_disc_scale_g = \
                self.discriminator_loss_fn(y_ds_hat_r, y_ds_hat_g)

        # Total discriminator loss
        loss_disc = loss_disc_periodic + loss_disc_scale

        metrics = {
            "loss_disc": loss_disc.item(),
            "loss_disc_periodic": loss_disc_periodic.item(),
            "loss_disc_scale": loss_disc_scale.item(),
        }

        return loss_disc, metrics

    def _train_generator_step(
        self,
        audio: torch.Tensor,
        audio_hat: torch.Tensor,
        mel: torch.Tensor,
    ) -> tuple:
        """Train generator."""
        # Generator adversarial loss (multi-period)
        _, y_df_hat_g, fmap_f_r, fmap_f_g = self.discriminator_mp(audio, audio_hat)
        loss_gen_periodic, losses_gen_periodic = self.generator_loss_fn(y_df_hat_g)

        # Generator adversarial loss (multi-scale)
        loss_gen_scale = 0
        losses_gen_scale = []
        fmap_s_r = []
        fmap_s_g = []

        if self.discriminator_ms is not None:
            _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.discriminator_ms(audio, audio_hat)
            loss_gen_scale, losses_gen_scale = self.generator_loss_fn(y_ds_hat_g)

        # Feature matching loss
        loss_feat_match = 0
        if self.use_mel_loss:
            loss_feat_match = self._feature_matching_loss(fmap_f_r, fmap_f_g)
            if self.discriminator_ms is not None:
                loss_feat_match += self._feature_matching_loss(fmap_s_r, fmap_s_g)
            loss_feat_match = loss_feat_match * self.config.lambda_feat_match

        # Mel spectrogram reconstruction loss
        loss_mel = 0
        if self.use_mel_loss:
            # Simple L1 loss on waveforms as proxy
            # In practice, you might want to compute mel from audio_hat
            loss_mel = nn.functional.l1_loss(audio, audio_hat) * self.config.lambda_mel

        # Total generator loss
        loss_gen = (
            loss_gen_periodic +
            loss_gen_scale +
            loss_feat_match +
            loss_mel
        )

        metrics = {
            "loss_gen": loss_gen.item(),
            "loss_gen_periodic": loss_gen_periodic.item(),
            "loss_gen_scale": loss_gen_scale.item(),
            "loss_feat_match": loss_feat_match.item(),
            "loss_mel": loss_mel.item(),
        }

        return loss_gen, metrics

    def _feature_matching_loss(
        self,
        fmap_r: List[List[torch.Tensor]],
        fmap_g: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute feature matching loss."""
        loss = 0
        num_items = 0
        # fmap_r and fmap_g are list of lists (one list per discriminator)
        # Each discriminator returns a list of feature maps from different layers
        for disc_r, disc_g in zip(fmap_r, fmap_g):
            for r, g in zip(disc_r, disc_g):
                # Ensure same shape by taking minimum length
                # This handles potential length mismatches from padding
                if r.shape != g.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(r.shape, g.shape))
                    r_slice = r[tuple(slice(0, s) for s in min_shape)]
                    g_slice = g[tuple(slice(0, s) for s in min_shape)]
                    loss += nn.functional.l1_loss(r_slice.detach(), g_slice) / len(disc_r)
                else:
                    loss += nn.functional.l1_loss(r.detach(), g) / len(disc_r)
                num_items += 1
        return loss / max(num_items, 1)

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
            # Extract waveform from VocoderOutput
            audio_hat = audio_hat.waveform

            # Ensure audio_hat has same length as audio (handle padding mismatches)
            if audio_hat.shape[-1] != audio.shape[-1]:
                min_len = min(audio.shape[-1], audio_hat.shape[-1])
                audio = audio[:, :, :min_len]
                audio_hat = audio_hat[:, :, :min_len]

            # Compute validation metrics
            loss_mel = nn.functional.l1_loss(audio, audio_hat)

            # Discriminator scores for evaluation
            y_df_hat_r, y_df_hat_g, _, _ = self.discriminator_mp(audio, audio_hat)

        metrics = {
            "val_loss_mel": loss_mel.item(),
            "val_discriminator_score": torch.mean(y_df_hat_g[-1]).item(),
        }

        return StepOutput(loss=loss_mel, metrics=metrics, logs={})

    def inference_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one inference step.

        Returns:
            Dictionary with generated audio.
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
            Dictionary with 'generator' and 'discriminator' optimizers.
        """
        opt_g = AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        opt_d = AdamW(
            self.discriminator_mp.parameters(),
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
        from torch.optim.lr_scheduler import ExponentialLR

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
        self.discriminator_mp.train()
        if self.discriminator_ms is not None:
            self.discriminator_ms.train()

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(self) -> None:
        """Called when training ends."""
        self.generator.eval()
