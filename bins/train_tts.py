#!/usr/bin/env python
# coding=utf-8
"""
Training script for TTS models.

Usage:
    python -m bins.train_tts --config config/tts.json
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add soniq to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soniq.training import FabricTrainer
from soniq.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TTS model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--exp-name", type=str, default="tts_exp", help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(exp_dir, "config.json"))

    # TODO: Implement TTS model training
    print("TTS training is not yet implemented.")
    print("Please check back later for updates.")


if __name__ == "__main__":
    main()
