"""Example: Using HiFi-GAN vocoder."""

import torch
import torchaudio

from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.pipelines import MelPipeline


def main():
    # Load configuration and model
    config = HifiGANConfig.from_pretrained("soniq/hifigan-24k")
    model = HifiGAN.from_pretrained("soniq/hifigan-24k")
    model.eval()

    # Create mel pipeline
    mel_pipeline = MelPipeline(
        sample_rate=24000,
        n_fft=1024,
        n_mel=80,
        hop_length=256,
    )

    # Load reference audio
    audio, sr = torchaudio.load("reference.wav")

    # Extract mel spectrogram
    mel = mel_pipeline(audio)

    # Generate waveform
    with torch.no_grad():
        output = model(mel.unsqueeze(0))

    # Save output
    torchaudio.save("output.wav", output.cpu(), sample_rate=24000)
    print("Generated output.wav")


if __name__ == "__main__":
    main()
