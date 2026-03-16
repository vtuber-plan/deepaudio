#!/usr/bin/env python
# coding=utf-8
"""
Inference script for trained HiFiGAN vocoder.

Usage:
    python -m bins/infer_vocoder --config ./outputs/hifigan/hifigan_vocoder/config.json \\
                                 --checkpoint ./outputs/hifigan/hifigan_vocoder/checkpoints/epoch_9.pt \\
                                 --input ./data/test_audio/audio_000.wav \\
                                 --output ./outputs/hifigan/generated.wav
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soniq.config import ExperimentConfig
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.processing.audio.io import load_audio, save_audio
from soniq.processing.features.mel import MelSpectrogramExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained vocoder")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="./output.wav", help="Path to output audio file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = ExperimentConfig.from_json(args.config)

    # Create model
    model_config = config.model
    hifi_config = HifiGANConfig(**model_config.model_args)
    model = HifiGAN(hifi_config)

    # Load checkpoint - support both formats
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Check checkpoint format
    if 'generator' in checkpoint:
        # New format: generator-only checkpoint
        model.load_state_dict(checkpoint['generator'])
    elif 'system' in checkpoint:
        # Old format: full system checkpoint
        system_state = checkpoint['system']
        generator_state = {}
        for key, value in system_state.items():
            if key.startswith('generator.'):
                generator_state[key.replace('generator.', '')] = value
        if not generator_state:
            generator_state = system_state
        model.load_state_dict(generator_state)
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'extra' in checkpoint:
        print(f"Checkpoint info: {checkpoint['extra']}")

    # Load input audio
    audio = load_audio(args.input, sample_rate=config.data.sample_rate)
    print(f"Loaded input audio: {audio.shape}")

    # Extract mel spectrogram
    mel_extractor = MelSpectrogramExtractor(config.data)
    sample = {'audio': audio}
    sample = mel_extractor(sample)
    mel = sample['mel'].unsqueeze(0)  # Add batch dimension
    print(f"Mel spectrogram shape: {mel.shape}")

    # Generate audio
    with torch.no_grad():
        output = model(mel)
        generated = output.waveform

    print(f"Generated audio shape: {generated.shape}")

    # Save output
    save_audio(generated.squeeze(0), args.output, sample_rate=config.data.sample_rate)
    print(f"Saved generated audio to {args.output}")


if __name__ == "__main__":
    main()
