#!/usr/bin/env python3
# coding=utf-8
"""
Training script for HiFiGAN vocoder using the new Soniq architecture.

Example usage:
    python -m soniq.bins.train_vocoder \
        --config config/hifigan.json \
        --output_dir ./outputs/hifigan
"""

import argparse
import os
import json
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from soniq.config.experiment import ExperimentConfig
from soniq.runtime.engine import FabricEngine
from soniq.tasks.vocoder.system import VocoderTaskSystem, VocoderConfig
from soniq.tasks.vocoder.datasets import VocoderDataset
from soniq.tasks.vocoder.collators import VocoderCollator
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HiFiGAN vocoder")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Override arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Max epochs override",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices to use (e.g., '1' for GPU 0, '0,1' for multi-GPU)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration."""
    return ExperimentConfig.from_json(config_path)


def build_vocoder_config(experiment_config: ExperimentConfig) -> VocoderConfig:
    """Build vocoder-specific config from experiment config."""
    return VocoderConfig(
        learning_rate=experiment_config.train.learning_rate,
        segment_size=getattr(experiment_config.data, 'segment_size', 8192),
        lambda_mel=getattr(experiment_config.model.model_args, 'lambda_mel', 45.0),
        lambda_feat_match=getattr(experiment_config.model.model_args, 'lambda_feat_match', 2.0),
    )


def build_generator(config: HifiGANConfig) -> HifiGAN:
    """Build generator model."""
    return HifiGAN(config)


def build_dataset(
    manifest_path: str,
    config: Any,
    split: str = "train",
) -> VocoderDataset:
    """Build dataset for training or validation."""
    return VocoderDataset(
        manifest_path=manifest_path,
        config=config,
        sample_rate=getattr(config, 'sample_rate', 24000),
    )


def build_dataloader(
    dataset: VocoderDataset,
    config: Any,
    split: str = "train",
) -> DataLoader:
    """Build dataloader."""
    collator = VocoderCollator(config)

    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=config.data.num_workers,
        persistent_workers=getattr(config.data, 'persistent_workers', True),
        pin_memory=getattr(config.data, 'pin_memory', True),
    )


def main():
    """Main training loop."""
    args = parse_args()

    # Load configuration
    experiment_config = load_config(args.config)

    # Override arguments if provided
    if args.learning_rate:
        experiment_config.train.learning_rate = args.learning_rate
    if args.max_epochs:
        experiment_config.train.max_epochs = args.max_epochs
    if args.batch_size:
        experiment_config.train.batch_size = args.batch_size

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config to output directory
    config_save_path = os.path.join(args.output_dir, "config.json")
    experiment_config.to_json(config_save_path)

    # Build vocoder config
    vocoder_config = build_vocoder_config(experiment_config)

    # Build generator
    model_config = HifiGANConfig(**experiment_config.model.model_args)
    generator = build_generator(model_config)

    # Print model info
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {num_params:,}")

    # Build task system
    system = VocoderTaskSystem(
        config=vocoder_config,
        generator=generator,
    )

    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(
        experiment_config.data.train_manifest,
        experiment_config,
        split="train",
    )
    val_dataset = None
    if experiment_config.data.val_manifest:
        val_dataset = build_dataset(
            experiment_config.data.val_manifest,
            experiment_config,
            split="val",
        )

    # Build dataloaders
    train_dataloader = build_dataloader(
        train_dataset,
        experiment_config,
        split="train",
    )
    val_dataloader = None
    if val_dataset:
        val_dataloader = build_dataloader(
            val_dataset,
            experiment_config,
            split="val",
        )

    # Parse devices
    devices = args.devices
    if devices == "auto":
        devices = 1 if torch.cuda.is_available() else None
    elif devices.isdigit():
        devices = int(devices)
    else:
        devices = [int(d) for d in devices.split(",")]

    # Build engine
    print("Initializing training engine...")
    engine = FabricEngine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        devices=devices,
        precision=experiment_config.precision,
        gradient_accumulation_steps=getattr(experiment_config.train, 'gradient_accumulation_steps', 1),
        gradient_clip_val=getattr(experiment_config.train, 'gradient_clip_val', 1.0),
    )

    # Train
    print("Starting training...")
    engine.fit(
        system=system,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_epochs=experiment_config.train.max_epochs,
        max_steps=getattr(experiment_config.train, 'max_steps', None),
        resume_from_checkpoint=args.resume,
    )

    print("Training complete!")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "model_final.pt")
    optimizers = system.configure_optimizers()
    engine.save_checkpoint(
        system=system,
        optimizer=optimizers,
        path=final_checkpoint_path,
        extra={"config": experiment_config.__dict__},
    )
    print(f"Final checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
