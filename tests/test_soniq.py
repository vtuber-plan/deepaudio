#!/usr/bin/env python
# coding=utf-8
"""Test script for Soniq package."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Adding to path: {project_root}")
print(f"Python path: {sys.path[:3]}...")


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing Soniq package imports...")

    # Test main package
    import soniq
    print(f"  ✓ soniq v{soniq.__version__}")

    # Test models
    from soniq.models import SoniqPreTrainedModel, SoniqConfig
    print("  ✓ soniq.models")

    from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
    print("  ✓ soniq.models.vocoders.hifigan")

    # Test pipelines
    from soniq.pipelines import AudioPipeline, MelPipeline
    print("  ✓ soniq.pipelines")

    # Test training
    from soniq.training import Trainer
    print("  ✓ soniq.training")

    # Test datasets
    from soniq.datasets import BaseDataset, BaseCollator
    print("  ✓ soniq.datasets")

    # Test processors
    from soniq.processors import AcousticExtractor, PhoneExtractor
    print("  ✓ soniq.processors")

    # Test features
    from soniq.features import MelFeatures, F0Features
    print("  ✓ soniq.features")

    # Test utils
    from soniq.utils import load_audio, save_audio, init_weights
    print("  ✓ soniq.utils")

    # Test config
    from soniq.config import BaseConfig, load_config
    print("  ✓ soniq.config")

    # Test modules
    from soniq.modules.commons import ResBlock1d, LayerNorm
    print("  ✓ soniq.modules.commons")

    from soniq.modules.transformer import MultiHeadAttention, TransformerEncoder
    print("  ✓ soniq.modules.transformer")

    # Test text processing
    from soniq.text import SYMBOLS, GraphemeToPhoneme
    print("  ✓ soniq.text")

    print("\n✓ All imports successful!")
    return True


def test_model_creation():
    """Test creating a model."""
    print("\nTesting model creation...")

    from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig

    config = HifiGANConfig(
        inter_channels=128,
        upsample_rates=[8, 8, 4, 2],
        upsample_initial_channel=512,
    )

    model = HifiGAN(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Created HifiGAN with {num_params / 1e6:.2f}M parameters")

    # Test forward pass - use correct input channels (inter_channels=128)
    import torch
    mel = torch.randn(1, 128, 100)  # n_mel should match inter_channels
    with torch.no_grad():
        output = model(mel)
    print(f"  ✓ Forward pass: {mel.shape} -> {output.waveform.shape}")

    return True


def test_pipelines():
    """Test audio pipelines."""
    print("\nTesting pipelines...")

    from soniq.pipelines import AudioPipeline, MelPipeline
    import torch

    # Test AudioPipeline
    audio_pipeline = AudioPipeline(sample_rate=24000)
    dummy_audio = torch.randn(1, 24000)
    processed = audio_pipeline(dummy_audio, src_sample_rate=24000)
    print(f"  ✓ AudioPipeline: {dummy_audio.shape} -> {processed.shape}")

    # Test MelPipeline
    mel_pipeline = MelPipeline(sample_rate=24000, n_mel=80)
    mel = mel_pipeline(dummy_audio)
    print(f"  ✓ MelPipeline: {dummy_audio.shape} -> {mel.shape}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Soniq Package Test Suite")
    print("=" * 50)

    all_passed = True

    try:
        all_passed &= test_imports()
        all_passed &= test_model_creation()
        all_passed &= test_pipelines()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
