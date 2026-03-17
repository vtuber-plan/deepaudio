# coding=utf-8
"""Example script for Vocos vocoder training and inference."""

import torch
from soniq.models.vocoders.vocos import (
    Vocos,
    HiFiGANMultiPeriodDiscriminator,
    SpecDiscriminator,
    VocosLoss,
)


def test_vocos_inference():
    """Test Vocos inference."""
    print("=== Testing Vocos Inference ===")

    # Create model
    vocos = Vocos(
        input_channels=128,
        dim=384,
        intermediate_dim=1152,
        num_layers=8,
        n_fft=800,
        hop_size=200,
    )
    vocos.eval()

    print(f"Vocos parameters: {sum(p.numel() for p in vocos.parameters()) / 1e6:.2f}M")

    # Create random mel spectrogram
    batch_size = 2
    mel_bins = 128
    mel_frames = 50
    mel = torch.randn(batch_size, mel_bins, mel_frames)

    # Generate audio
    with torch.no_grad():
        audio = vocos(mel)

    print(f"Input mel shape: {mel.shape}")
    print(f"Output audio shape: {audio.shape}")

    # Calculate expected audio length
    expected_length = mel_frames * 200  # hop_size
    print(f"Expected audio length: ~{expected_length} samples")
    print(f"Actual audio length: {audio.shape[-1]} samples")

    print("Vocos inference test passed!\n")


def test_vocos_discriminators():
    """Test Vocos discriminators."""
    print("=== Testing Vocos Discriminators ===")

    # Create discriminators
    period_disc = HiFiGANMultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11])
    spec_disc = SpecDiscriminator()

    print(f"Period discriminator parameters: {sum(p.numel() for p in period_disc.parameters()) / 1e6:.2f}M")
    print(f"Spec discriminator parameters: {sum(p.numel() for p in spec_disc.parameters()) / 1e6:.2f}M")

    # Create random audio
    batch_size = 2
    audio_length = 8192
    audio = torch.randn(batch_size, 1, audio_length)

    # Forward pass
    period_outs = period_disc(audio)
    spec_outs = spec_disc(audio)

    print(f"Period discriminator: {len(period_outs)} sub-outputs")
    for i, out in enumerate(period_outs):
        print(f"  Disc {i}: {[o.shape for o in out]}")

    print(f"Spec discriminator: {len(spec_outs)} sub-outputs")
    for i, out in enumerate(spec_outs):
        print(f"  Disc {i}: {[o.shape for o in out]}")

    print("Vocos discriminators test passed!\n")


def test_vocos_loss():
    """Test Vocos loss functions."""
    print("=== Testing Vocos Loss ===")

    # Create loss function
    loss_fn = VocosLoss(
        sample_rate=24000,
        mel_loss_weight=10.0,
        adv_loss_weight=2.0,
        fm_loss_weight=2.0,
    )

    # Create random audio
    batch_size = 2
    audio_length = 8192
    real_audio = torch.randn(batch_size, 1, audio_length)
    fake_audio = torch.randn(batch_size, 1, audio_length)

    # Create discriminators for loss computation
    period_disc = HiFiGANMultiPeriodDiscriminator()
    spec_disc = SpecDiscriminator()

    # Get discriminator outputs
    period_real_outs = period_disc(real_audio)
    period_fake_outs = period_disc(fake_audio)
    spec_real_outs = spec_disc(real_audio)
    spec_fake_outs = spec_disc(fake_audio)

    # Compute discriminator loss
    disc_loss_dict = loss_fn.discriminator_loss(
        period_real_outs + spec_real_outs,
        period_fake_outs + spec_fake_outs,
    )
    print(f"Discriminator loss: {disc_loss_dict['loss'].item():.4f}")

    # Compute generator loss
    gen_loss_dict = loss_fn.generator_loss(
        pred_audio=fake_audio,
        target_audio=real_audio,
        real_outputs=period_real_outs + spec_real_outs,
        fake_outputs=period_fake_outs + spec_fake_outs,
    )
    print(f"Generator loss: {gen_loss_dict['total_loss'].item():.4f}")
    print(f"  Mel loss: {gen_loss_dict['mel_loss'].item():.4f}")
    print(f"  Adv loss: {gen_loss_dict['adv_loss'].item():.4f}")
    print(f"  FM loss: {gen_loss_dict['fm_loss'].item():.4f}")

    print("Vocos loss test passed!\n")


def test_vocos_with_different_mel_lengths():
    """Test Vocos with different mel lengths."""
    print("=== Testing Vocos with Different Mel Lengths ===")

    vocos = Vocos(input_channels=128, dim=384, num_layers=8)
    vocos.eval()

    # Test with different mel lengths
    mel_lengths = [10, 25, 50, 100, 200]

    for mel_frames in mel_lengths:
        mel = torch.randn(1, 128, mel_frames)
        with torch.no_grad():
            audio = vocos(mel)
        print(f"Mel frames: {mel_frames:3d} -> Audio samples: {audio.shape[-1]:5d}")

    print("Vocos variable length test passed!\n")


if __name__ == "__main__":
    test_vocos_inference()
    test_vocos_discriminators()
    test_vocos_loss()
    test_vocos_with_different_mel_lengths()

    print("All Vocos tests passed!")
