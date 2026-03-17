#!/usr/bin/env python
# coding=utf-8
"""
Training script for Vocoder models.

Usage:
    python -m bins.train_vocoder --config config/experiments/hifigan_train.json
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add soniq to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soniq.config import ExperimentConfig
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.models.vocoders.vocos import Vocos, VocosConfig as VocosModelConfig
from soniq.tasks.vocoder.datasets import VocoderDataset, VocoderCollator
from soniq.tasks.vocoder.system import VocoderTaskSystem, VocoderConfig
from soniq.tasks.vocoder.vocos_system import VocosTaskSystem, VocosConfig as VocosTrainConfig
from soniq.runtime import FabricTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vocoder model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    experiment_config = ExperimentConfig.from_json(args.config)

    # Override output dir if provided
    if args.output_dir:
        experiment_config.output_dir = args.output_dir
    if args.exp_name:
        experiment_config.name = args.exp_name

    # Create output directory
    os.makedirs(experiment_config.output_dir, exist_ok=True)
    exp_dir = os.path.join(experiment_config.output_dir, experiment_config.name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    experiment_config.to_json(os.path.join(exp_dir, "config.json"))

    # Create model based on model_type
    model_config = experiment_config.model
    model_type = getattr(model_config, 'model_type', 'HiFiGAN')

    if model_type == 'Vocos':
        # Create Vocos model
        vocos_config = VocosModelConfig(**model_config.model_args)
        generator = Vocos(
            input_channels=vocos_config.input_channels,
            dim=vocos_config.dim,
            intermediate_dim=vocos_config.intermediate_dim,
            num_layers=vocos_config.num_layers,
            n_fft=vocos_config.n_fft,
            hop_size=vocos_config.hop_size,
            padding=vocos_config.padding,
        )

        # Create Vocos task system
        train_config = VocosTrainConfig(
            learning_rate=experiment_config.train.learning_rate,
            betas=tuple(experiment_config.train.betas),
            lr_decay=experiment_config.train.lr_decay,
            segment_size=experiment_config.data.segment_size,
            lambda_mel=experiment_config.loss.get('mel_loss_weight', 10.0) if hasattr(experiment_config, 'loss') else 10.0,
            lambda_adv=experiment_config.loss.get('adv_loss_weight', 2.0) if hasattr(experiment_config, 'loss') else 2.0,
            lambda_fm=experiment_config.loss.get('fm_loss_weight', 2.0) if hasattr(experiment_config, 'loss') else 2.0,
            sample_rate=experiment_config.data.sample_rate,
            n_fft=experiment_config.data.n_fft,
            hop_size=experiment_config.data.hop_length,
        )

        system = VocosTaskSystem(
            config=train_config,
            generator=generator,
        )
    else:
        # Create HiFiGAN model (default)
        hifi_config = HifiGANConfig(**model_config.model_args)
        generator = HifiGAN(hifi_config)

        # Create vocoder config
        vocoder_config = VocoderConfig(
            learning_rate=experiment_config.train.learning_rate,
            segment_size=experiment_config.data.segment_size,
        )

        # Create system (will create discriminators internally)
        system = VocoderTaskSystem(
            config=vocoder_config,
            generator=generator,
        )

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M")
    print(f"Total parameters: {sum(p.numel() for p in system.parameters()) / 1e6:.2f}M")

    # Create datasets
    train_dataset = VocoderDataset(
        manifest_path=experiment_config.data.train_manifest,
        config=experiment_config.data,
        sample_rate=experiment_config.data.sample_rate,
    )

    val_dataset = VocoderDataset(
        manifest_path=experiment_config.data.val_manifest,
        config=experiment_config.data,
        sample_rate=experiment_config.data.sample_rate,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_collator = VocoderCollator(config=experiment_config.data)
    val_collator = VocoderCollator(config=experiment_config.data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment_config.train.batch_size,
        shuffle=True,
        num_workers=getattr(experiment_config.data, 'num_workers', 0),
        collate_fn=train_collator,
        persistent_workers=getattr(experiment_config.data, 'persistent_workers', False),
        pin_memory=getattr(experiment_config.data, 'pin_memory', False),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=experiment_config.train.batch_size,
        shuffle=False,
        num_workers=getattr(experiment_config.data, 'num_workers', 0),
        collate_fn=val_collator,
        persistent_workers=getattr(experiment_config.data, 'persistent_workers', False),
        pin_memory=getattr(experiment_config.data, 'pin_memory', False),
    )

    # Create trainer
    trainer = FabricTrainer(
        accelerator=experiment_config.accelerator,
        devices=experiment_config.devices,
        precision=experiment_config.precision,
        max_epochs=experiment_config.train.max_epochs,
        gradient_accumulation_steps=getattr(experiment_config.train, 'gradient_accumulation_steps', 1),
        gradient_clip_val=experiment_config.train.gradient_clip_val,
        default_root_dir=exp_dir,
        seed=experiment_config.seed,
    )

    # Create optimizers
    optimizers = system.configure_optimizers()

    print(f"Starting training...")
    print(f"  Accelerator: {experiment_config.accelerator}")
    print(f"  Devices: {experiment_config.devices}")
    print(f"  Precision: {experiment_config.precision}")
    print(f"  Batch size: {experiment_config.train.batch_size}")
    print(f"  Max epochs: {experiment_config.train.max_epochs}")

    # Train
    trainer.fit(
        system=system,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizers=optimizers,
        resume_from_checkpoint=args.resume,
    )

    print(f"Training completed. Checkpoint saved to {exp_dir}/checkpoints/")


if __name__ == "__main__":
    main()
