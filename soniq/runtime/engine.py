# coding=utf-8
"""Fabric runtime engine."""

from typing import Any, Dict, List, Optional, Union
import os
import torch
from torch.utils.data import DataLoader
from lightning import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from dataclasses import dataclass


@dataclass
class EngineState:
    epoch: int = 0
    global_step: int = 0


class FabricEngine:
    """Runtime engine based on Lightning Fabric."""

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Optional[Union[int, List[int]]] = None,
        precision: str = "32-true",
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        loggers: Optional[List] = None,
        callbacks: Optional[List] = None,
        default_root_dir: Optional[str] = None,
        seed: int = 42,
        use_tensorboard: bool = True,
    ):
        # Handle devices=None for CPU accelerator
        if devices is None:
            devices = 1  # Default to 1 CPU device

        # Setup TensorBoard logger
        logger_list = loggers or []
        if use_tensorboard and default_root_dir:
            try:
                from lightning.fabric.loggers import TensorBoardLogger
                tb_logger = TensorBoardLogger(
                    root_dir=os.path.join(default_root_dir, "logs"),
                    name="tensorboard",
                )
                logger_list.append(tb_logger)
            except (ModuleNotFoundError, ImportError):
                # TensorBoard not available, skip logging
                pass

        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            loggers=logger_list,
            callbacks=callbacks,
        )
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.state = EngineState()
        self.seed = seed
        self.default_root_dir = default_root_dir

    def fit(self, system, train_dataloader, val_dataloader=None,
            optimizers=None, resume_from_checkpoint: Optional[str] = None):
        """Fit the system using the provided dataloaders."""
        self.fabric.launch()

        # Set seed
        torch.manual_seed(self.seed)

        # Setup system and optimizers
        if optimizers is None:
            optimizer = system.configure_optimizers()
        else:
            optimizer = optimizers

        if isinstance(optimizer, dict):
            # Multi-optimizer setup (e.g., GAN training)
            system = self.fabric.setup_module(system)
            for name, opt in optimizer.items():
                optimizer[name] = opt  # Don't setup optimizer separately, fabric handles it
        else:
            system, optimizer = self.fabric.setup(system, optimizer)

        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        if resume_from_checkpoint:
            self._load_checkpoint(system, optimizer, resume_from_checkpoint)

        system.on_train_start()

        best_val_loss = float('inf')

        for epoch in range(self.state.epoch, self.max_epochs):
            self.state.epoch = epoch
            system.on_epoch_start(epoch)
            self._train_epoch(system, train_dataloader, optimizer)

            val_loss = None
            if val_dataloader:
                val_metrics = self._validate_epoch(system, val_dataloader)
                val_loss = val_metrics.get("val/loss", float('inf'))

            system.on_epoch_end(epoch)

            # Save checkpoint at the end of each epoch
            if self.default_root_dir:
                ckpt_dir = os.path.join(self.default_root_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)

                # Save full checkpoint (generator + discriminator)
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
                self.save_checkpoint(system, optimizer, ckpt_path,
                                     {"val_loss": val_loss} if val_loss else None)

                # Save generator-only checkpoint for inference
                gen_path = os.path.join(ckpt_dir, f"generator_epoch_{epoch}.pt")
                self.save_generator_checkpoint(system, gen_path,
                                               {"val_loss": val_loss, "epoch": epoch} if val_loss else {"epoch": epoch})

                # Save best model
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
                    self.save_checkpoint(system, optimizer, best_ckpt_path,
                                         {"val_loss": val_loss, "best_epoch": epoch})

                    best_gen_path = os.path.join(ckpt_dir, "generator_best.pt")
                    self.save_generator_checkpoint(system, best_gen_path,
                                                   {"val_loss": val_loss, "epoch": epoch})

            if self.max_steps and self.state.global_step >= self.max_steps:
                break

        system.on_train_end()

    def _train_epoch(self, system, dataloader, optimizer):
        """Train for one epoch."""
        system.train()
        for batch_idx, batch in enumerate(dataloader):
            self.state.global_step += 1

            # Handle multiple optimizers (e.g., GAN training)
            if isinstance(optimizer, dict):
                # For GAN training with AMP, we need to carefully handle gradient scaling
                # Zero all gradients first
                for opt_name, opt in optimizer.items():
                    opt.zero_grad()

                # Discriminator step (optimizer_idx=0)
                step_output_d = system.training_step(batch, batch_idx, optimizer_idx=0)
                self.fabric.backward(step_output_d.loss)
                # Clip gradients without calling unscale (already done by Fabric.backward)
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        system.parameters(),
                        self.gradient_clip_val
                    )
                optimizer["discriminator"].step()
                self._log_gan_metrics(step_output_d, "discriminator")

                # Generator step (optimizer_idx=1)
                step_output_g = system.training_step(batch, batch_idx, optimizer_idx=1)
                self.fabric.backward(step_output_g.loss)
                # Clip gradients without calling unscale (already done by Fabric.backward)
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        system.parameters(),
                        self.gradient_clip_val
                    )
                optimizer["generator"].step()
                self._log_gan_metrics(step_output_g, "generator")
            else:
                step_output = system.training_step(batch, batch_idx)
                self.fabric.backward(step_output.loss)
                if self.gradient_clip_val:
                    self.fabric.clip_gradients(system, optimizer, clip_val=self.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad()
                self._log_metrics(step_output)

    def _log_gan_metrics(self, step_output, prefix):
        """Log GAN-specific metrics."""
        self.fabric.log(f"{prefix}/loss", step_output.loss.item())
        for name, value in step_output.metrics.items():
            self.fabric.log(f"{prefix}/{name}", value)

    def _validate_epoch(self, system, dataloader):
        system.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                step_output = system.validation_step(batch, batch_idx)
                total_loss += step_output.loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        self.fabric.log("val/loss", avg_loss)
        return {"val/loss": avg_loss}

    def _log_metrics(self, step_output):
        self.fabric.log("train/loss", step_output.loss.item())
        for name, value in step_output.metrics.items():
            self.fabric.log(f"train/{name}", value)

    def _prepare_multi_optimizer(self, system, optimizers):
        """Prepare multiple optimizers for GAN training."""
        system = self.fabric.setup_module(system)
        for name, opt in optimizers.items():
            optimizers[name] = opt
        return system, optimizers

    def _load_checkpoint(self, system, optimizer, path):
        checkpoint = self.fabric.load(path)
        system.load_state_dict(checkpoint["system"])
        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                if name in checkpoint:
                    opt.load_state_dict(checkpoint[name])
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)

    def save_checkpoint(self, system, optimizer, path, extra=None):
        """Save checkpoint to file."""
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "system": system.state_dict(),
            "optimizer": optimizer.state_dict() if not isinstance(optimizer, dict)
                         else {k: v.state_dict() for k, v in optimizer.items()},
        }
        if extra:
            checkpoint["extra"] = extra

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use torch.save directly for compatibility
        torch.save(checkpoint, path)

    def save_generator_checkpoint(self, system, path, extra=None):
        """Save generator-only checkpoint for inference."""
        # Extract generator state dict
        system_state = system.state_dict()
        generator_state = {}

        for key, value in system_state.items():
            if key.startswith('generator.'):
                generator_state[key.replace('generator.', '')] = value
            else:
                # Keep other keys that might be needed (e.g., for non-GAN models)
                if not any(k in key for k in ['discriminator', 'discriminator_mp', 'discriminator_ms']):
                    generator_state[key] = value

        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "generator": generator_state,
        }
        if extra:
            checkpoint["extra"] = extra

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(checkpoint, path)
