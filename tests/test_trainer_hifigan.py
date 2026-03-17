#!/usr/bin/env python
# coding=utf-8
"""
Test script for the unified Trainer system with HiFiGAN vocoder.

Tests both Fabric and Accelerate engines using synthetic data.

Usage:
    python tests/test_trainer_hifigan.py --engine fabric
    python tests/test_trainer_hifigan.py --engine accelerate
    python tests/test_trainer_hifigan.py --engine both
"""

import argparse
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


# ============================================================================
# Synthetic Dataset
# ============================================================================

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic dataset."""
    sample_rate: int = 24000
    n_mel: int = 80
    segment_size: int = 8192
    hop_length: int = 256
    n_fft: int = 1024


class SyntheticVocoderDataset(Dataset):
    """Synthetic dataset for testing vocoder training."""

    def __init__(
        self,
        num_samples: int = 100,
        config: Optional[SyntheticDataConfig] = None,
    ):
        self.num_samples = num_samples
        self.config = config or SyntheticDataConfig()

        # Calculate mel length from audio segment size
        self.mel_length = self.config.segment_size // self.config.hop_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random audio (batch, 1, time)
        audio = torch.randn(1, self.config.segment_size)

        # Generate random mel spectrogram (n_mel, time)
        mel = torch.randn(self.config.n_mel, self.mel_length)

        return {
            "id": f"sample_{idx}",
            "audio": audio,
            "mel": mel,
            "audio_length": self.config.segment_size,
            "mel_length": self.mel_length,
        }


class SyntheticVocoderCollator:
    """Collator for synthetic vocoder dataset."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [s["audio"] for s in batch]
        mels = [s["mel"] for s in batch]

        return {
            "audio": torch.stack(audios),  # (batch, 1, time)
            "mel": torch.stack(mels),      # (batch, n_mel, time)
            "audio_lengths": torch.tensor([s["audio_length"] for s in batch]),
            "mel_lengths": torch.tensor([s["mel_length"] for s in batch]),
            "ids": [s["id"] for s in batch],
        }


# ============================================================================
# Vocoder Task System for Testing
# ============================================================================

@dataclass
class TestVocoderConfig:
    """Configuration for test vocoder training."""
    learning_rate: float = 2e-4
    betas: tuple = (0.8, 0.99)
    segment_size: int = 8192
    lambda_mel: float = 45.0
    lambda_adv: float = 1.0
    lambda_feat_match: float = 2.0


class TestVocoderTaskSystem(nn.Module):
    """
    Simplified vocoder task system for testing.

    This is a minimal implementation to test the Trainer without
    depending on the full VocoderTaskSystem.
    """

    def __init__(
        self,
        config: TestVocoderConfig,
        generator: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.generator = generator

        # Create discriminators
        self.discriminator_mp = HiFiGANMultiPeriodDiscriminator()
        self.discriminator_ms = HiFiGANMultiScaleDiscriminator()

        # Loss functions
        self.l1_loss = nn.L1Loss()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        output = self.generator(mel)
        return output.waveform

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> StepOutput:
        """Execute one training step."""
        audio = batch["audio"]  # (batch, 1, time)
        mel = batch["mel"]      # (batch, n_mel, time)

        # Generate audio
        audio_hat = self.forward(mel)

        # Ensure same length
        min_len = min(audio.shape[-1], audio_hat.shape[-1])
        audio = audio[:, :, :min_len]
        audio_hat = audio_hat[:, :, :min_len]

        # Simple loss: L1 + adversarial
        l1_loss = self.l1_loss(audio, audio_hat) * self.config.lambda_mel

        # Discriminator outputs for adversarial loss
        y_df_hat_r, y_df_hat_g, _, _ = self.discriminator_mp(audio, audio_hat)

        # Generator adversarial loss
        adv_loss = 0
        for dr in y_df_hat_g:
            adv_loss += torch.mean((1 - dr) ** 2)
        adv_loss = adv_loss * self.config.lambda_adv

        loss = l1_loss + adv_loss

        metrics = {
            "loss": loss.item(),
            "l1_loss": l1_loss.item(),
            "adv_loss": adv_loss.item(),
        }

        return StepOutput(loss=loss, metrics=metrics, logs={})

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> StepOutput:
        """Execute one validation step."""
        audio = batch["audio"]
        mel = batch["mel"]

        with torch.no_grad():
            audio_hat = self.forward(mel)

            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            audio = audio[:, :, :min_len]
            audio_hat = audio_hat[:, :, :min_len]

            loss = self.l1_loss(audio, audio_hat)

        return StepOutput(loss=loss, metrics={"val_loss": loss.item()}, logs={})

    def inference_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one inference step."""
        mel = batch["mel"]
        with torch.no_grad():
            audio_hat = self.forward(mel)
        return {"audio_hat": audio_hat}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers."""
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        opt_d = torch.optim.AdamW(
            list(self.discriminator_mp.parameters()) +
            list(self.discriminator_ms.parameters()),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        return {"generator": opt_g, "discriminator": opt_d}

    def on_train_start(self) -> None:
        """Called when training starts."""
        self.generator.train()
        self.discriminator_mp.train()
        self.discriminator_ms.train()

    def on_train_end(self) -> None:
        """Called when training ends."""
        self.generator.eval()


# ============================================================================
# Test Callback
# ============================================================================

class TestProgressCallback(Callback):
    """Callback for printing progress during test."""

    def on_train_batch_end(self, trainer, ctx, batch, batch_idx, outputs):
        if ctx.need_to_log:
            print(f"  [Step {ctx.iteration}] loss: {outputs.loss.item():.4f}")

    def on_validation_end(self, trainer, ctx, outputs, metrics):
        if trainer.engine.is_main_process():
            print(f"  [Validation] loss: {metrics.get('loss', 0):.4f}")


# ============================================================================
# Test Functions
# ============================================================================

def test_engine(engine_type: str, output_dir: str, debug: bool = False):
    """Test a specific training engine."""
    print(f"\n{'='*60}")
    print(f"Testing {engine_type.upper()} engine")
    print(f"{'='*60}\n")

    # Create output directory
    engine_output_dir = os.path.join(output_dir, engine_type)
    os.makedirs(engine_output_dir, exist_ok=True)

    # Configuration
    data_config = SyntheticDataConfig()
    train_config = TestVocoderConfig()

    # Create model
    print("Creating HiFiGAN model...")
    hifi_config = HifiGANConfig(
        inter_channels=80,  # Match n_mel
        upsample_rates=(8, 8, 4),  # 256 = 8 * 8 * 4
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=(1, 3),
    )
    generator = HifiGAN(hifi_config)

    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {num_params / 1e6:.2f}M")

    # Create task system
    system = TestVocoderTaskSystem(train_config, generator)

    # Create synthetic datasets
    print("Creating synthetic datasets...")
    train_dataset = SyntheticVocoderDataset(
        num_samples=50 if debug else 200,
        config=data_config,
    )
    val_dataset = SyntheticVocoderDataset(
        num_samples=10 if debug else 20,
        config=data_config,
    )

    # Create dataloaders
    collator = SyntheticVocoderCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=4 if debug else 8,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create trainer
    print(f"\nInitializing {engine_type} trainer...")
    trainer = Trainer(
        engine=engine_type,
        run_path=engine_output_dir,
        max_steps=10 if debug else 50,
        gradient_accumulation_steps=1,
        gradient_clip_val=1.0,
        log_interval_steps=5,
        val_interval_steps=10 if debug else 20,
        save_interval_steps=20,
        seed=42,
        debug=debug,
        callbacks=[TestProgressCallback()],
        precision="32-true",  # Use FP32 for testing stability
    )

    # Start training
    print(f"\nStarting training with {engine_type} engine...")
    try:
        trainer.fit(
            system=system,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
        print(f"\n[SUCCESS] {engine_type.upper()} engine test passed!")
        return True
    except Exception as e:
        print(f"\n[FAILED] {engine_type.upper()} engine test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Trainer with HiFiGAN")
    parser.add_argument(
        "--engine",
        type=str,
        default="both",
        choices=["fabric", "accelerate", "both"],
        help="Engine to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test artifacts",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (fewer steps)",
    )
    args = parser.parse_args()

    # Create temporary output directory if not specified
    if args.output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="trainer_test_")
        print(f"Using temporary output directory: {output_dir}")
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Test engines
    if args.engine in ["fabric", "both"]:
        try:
            results["fabric"] = test_engine("fabric", output_dir, args.debug)
        except ImportError as e:
            print(f"\n[SKIPPED] Fabric engine not available: {e}")
            results["fabric"] = None

    if args.engine in ["accelerate", "both"]:
        try:
            results["accelerate"] = test_engine("accelerate", output_dir, args.debug)
        except ImportError as e:
            print(f"\n[SKIPPED] Accelerate engine not available: {e}")
            results["accelerate"] = None

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for engine, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {engine.upper()}: {status}")

    # Cleanup temporary directory if we created it
    if args.output_dir is None and all(p is not False for p in results.values()):
        print(f"\nCleaning up temporary directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    # Return exit code
    if all(p in [True, None] for p in results.values()):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())