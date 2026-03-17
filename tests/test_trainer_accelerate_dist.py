#!/usr/bin/env python
# coding=utf-8
"""
Test Trainer with Accelerate distributed backend.

Usage:
    accelerate launch --num_processes 2 tests/test_trainer_accelerate_dist.py
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.models.vocoders.hifigan.discriminator import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)
from soniq.training import Trainer, StepOutput
from soniq.training.base.callback import Callback


# Simple config
@dataclass
class SimpleConfig:
    learning_rate: float = 2e-4
    betas: tuple = (0.8, 0.99)


class SimpleVocoderSystem(nn.Module):
    """Simple vocoder system for testing."""

    def __init__(self, config: SimpleConfig, generator: nn.Module):
        super().__init__()
        self.config = config
        self.generator = generator
        self.discriminator = HiFiGANMultiPeriodDiscriminator()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel):
        return self.generator(mel).waveform

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        mel = batch["mel"]

        audio_hat = self.forward(mel)
        min_len = min(audio.shape[-1], audio_hat.shape[-1])
        audio = audio[:, :, :min_len]
        audio_hat = audio_hat[:, :, :min_len]

        l1_loss = self.l1_loss(audio, audio_hat)
        loss = l1_loss * 45.0

        return StepOutput(loss=loss, metrics={"loss": loss.item()}, logs={})

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        mel = batch["mel"]

        with torch.no_grad():
            audio_hat = self.forward(mel)
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            loss = self.l1_loss(audio[:, :, :min_len], audio_hat[:, :, :min_len])

        return StepOutput(loss=loss, metrics={"val_loss": loss.item()}, logs={})

    def inference_step(self, batch):
        with torch.no_grad():
            return {"audio_hat": self.forward(batch["mel"])}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.generator.parameters(), lr=self.config.learning_rate)

    def on_train_start(self):
        self.generator.train()

    def on_train_end(self):
        self.generator.eval()


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    def __init__(self, num_samples=100, n_mel=80, segment_size=8192, hop_length=256):
        self.num_samples = num_samples
        self.n_mel = n_mel
        self.segment_size = segment_size
        self.mel_length = segment_size // hop_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "audio": torch.randn(1, self.segment_size),
            "mel": torch.randn(self.n_mel, self.mel_length),
        }


class ProgressCallback(Callback):
    def on_train_batch_end(self, trainer, ctx, batch, batch_idx, outputs):
        if ctx.need_to_log:
            trainer.engine.print(f"[Rank {trainer.engine.ctx.rank}] Step {ctx.iteration}: loss={outputs.loss.item():.4f}")


def main():
    # Create output directory
    output_dir = "/tmp/trainer_accelerate_dist_test"
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    config = HifiGANConfig(
        inter_channels=80,
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=(1, 3),
    )
    generator = HifiGAN(config)

    # Create system
    train_config = SimpleConfig()
    system = SimpleVocoderSystem(train_config, generator)

    # Create datasets
    train_dataset = DummyDataset(num_samples=50)
    val_dataset = DummyDataset(num_samples=10)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Create trainer with accelerate engine
    trainer = Trainer(
        engine="accelerate",
        run_path=output_dir,
        max_steps=20,
        gradient_accumulation_steps=1,
        log_interval_steps=5,
        val_interval_steps=10,
        save_interval_steps=20,
        seed=42,
        callbacks=[ProgressCallback()],
    )

    # Print backend info
    trainer.engine.print("=" * 60)
    trainer.engine.print("Testing Trainer with Accelerate Distributed Backend")
    trainer.engine.print("=" * 60)

    # Fit
    trainer.fit(system, train_loader, val_loader)

    trainer.engine.print("\n" + "=" * 60)
    trainer.engine.print("Trainer + Accelerate distributed test PASSED!")
    trainer.engine.print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()