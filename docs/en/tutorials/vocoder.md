# Vocoder Tutorial

This tutorial shows you how to train and use neural vocoders with Soniq.

## What is a Neural Vocoder?

A neural vocoder converts mel spectrograms (or other acoustic features) into waveforms. It's a crucial component in text-to-speech systems.

## Supported Vocoders

| Model | Description | Sample Rate |
|-------|-------------|-------------|
| HiFi-GAN | High-fidelity GAN vocoder | 16k, 24k, 44k, 48k |
| MelGAN | Fast GAN vocoder | 16k, 24k |
| BigVGAN | Universal neural vocoder | 24k, 44k |
| WaveNet | Autoregressive vocoder | 16k, 24k |

## Using Pre-trained Vocoders

### Load HiFi-GAN

```python
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.pipelines import MelPipeline
import torchaudio

# Load model
config = HifiGANConfig.from_pretrained("soniq/hifigan-24k")
model = HifiGAN.from_pretrained("soniq/hifigan-24k")
model.eval()

# Prepare input
audio, sr = torchaudio.load("reference.wav")
mel_pipeline = MelPipeline(sample_rate=24000, n_mel=80)
mel = mel_pipeline(audio)

# Generate waveform
with torch.no_grad():
    output = model(mel.unsqueeze(0))

# Save output
torchaudio.save("output.wav", output.cpu(), sample_rate=24000)
```

## Training a Vocoder

### Prepare Dataset

Create a metadata JSON file:

```json
[
    {"id": "utterance_001", "duration": 2.5, "speaker": "spk1"},
    {"id": "utterance_002", "duration": 3.1, "speaker": "spk2"}
]
```

### Training Script

```python
import torch
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader

# Configuration
config = HifiGANConfig(
    inter_channels=128,
    upsample_rates=[8, 8, 4, 2],
    upsample_initial_channel=512,
)

# Model
model = HifiGAN(config)

# Dataset
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
    sample_rate=24000,
)

# DataLoader
dataloader = build_dataloader(
    dataset,
    collator=BaseCollator(),
    batch_size=16,
    num_workers=4,
)

# Trainer
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
    gradient_accumulation_steps=1,
)

# Training step function
def train_step(model, batch):
    mel = batch["mel"]
    audio = batch["wav"]

    # Generate audio
    generated = model(mel)

    # Compute loss (simplified)
    loss = torch.nn.functional.l1_loss(generated, audio)
    return loss

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(
    model,
    dataloader,
    optimizer=optimizer,
    train_step_fn=train_step,
)
```

### GAN Training (Advanced)

For GAN-based vocoders like HiFi-GAN, you need to train both generator and discriminator:

```python
from soniq.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

# Discriminators
mpd = MultiPeriodDiscriminator()
msd = MultiScaleDiscriminator()

# Training step for GAN
def gan_train_step(model, mpd, msd, batch):
    mel = batch["mel"]
    audio = batch["wav"]

    # Generate
    generated = model(mel)

    # Discriminator loss
    loss_d = compute_discriminator_loss(mpd, msd, audio, generated)

    # Generator loss
    loss_g = compute_generator_loss(mpd, msd, generated)

    return loss_g + loss_d
```

## Exporting Models

### Export to ONNX

```python
# Dummy input
dummy_mel = torch.randn(1, 80, 100)

# Export
torch.onnx.export(
    model,
    dummy_mel,
    "hifigan.onnx",
    input_names=["mel"],
    output_names=["audio"],
    dynamic_axes={"mel": {2: "time"}, "audio": {2: "time"}},
)
```

## Tips for Better Results

1. **Use high-quality training data** - Clean audio with minimal noise
2. **Train long enough** - At least 100k steps for HiFi-GAN
3. **Use mixed precision** - Faster training with `precision="16-mixed"`
4. **Monitor validation loss** - Stop training when loss plateaus
