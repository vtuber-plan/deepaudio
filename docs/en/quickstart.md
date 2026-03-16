# Quick Start

This guide shows you how to get started with Soniq in minutes.

## Load Pre-trained Model

### Load HiFi-GAN Vocoder

```python
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.pipelines import MelPipeline
import torchaudio

# Load configuration and model
config = HifiGANConfig.from_pretrained("soniq/hifigan-24k")
model = HifiGAN.from_pretrained("soniq/hifigan-24k")

# Create mel pipeline
mel_pipeline = MelPipeline(
    sample_rate=24000,
    n_fft=1024,
    n_mel=80,
    hop_length=256,
)

# Load audio and extract mel spectrogram
audio, sr = torchaudio.load("audio.wav")
mel = mel_pipeline(audio)

# Generate waveform
output = model(mel.unsqueeze(0))
```

### Use AutoModel

```python
from soniq.models import AutoModel

# Automatically load the correct model class
model = AutoModel.from_pretrained("soniq/hifigan-24k")
```

## Training Your First Model

### Basic Training Setup

```python
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import build_dataloader, BaseDataset, BaseCollator

# Create model
config = HifiGANConfig()
model = HifiGAN(config)

# Create trainer
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
)

# Prepare dataset and dataloader
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
)
dataloader = build_dataloader(
    dataset,
    collator=BaseCollator(),
    batch_size=32,
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(model, dataloader, optimizer=optimizer)
```

## Audio Processing Pipelines

### Audio Pipeline

```python
from soniq.pipelines import AudioPipeline

# Create pipeline
pipeline = AudioPipeline(sample_rate=24000, mono=True)

# Load and preprocess audio
audio = pipeline("path/to/audio.wav")
print(audio.shape)  # (1, num_samples)
```

### Mel Pipeline

```python
from soniq.pipelines import MelPipeline

# Create pipeline
pipeline = MelPipeline(
    sample_rate=24000,
    n_fft=1024,
    n_mel=80,
    hop_length=256,
)

# Extract mel spectrogram
mel = pipeline(audio)
print(mel.shape)  # (80, num_frames)
```

## Feature Extraction

### Acoustic Features

```python
from soniq.processors import AcousticExtractor

# Create extractor
extractor = AcousticExtractor(sample_rate=24000)

# Extract features
mel = extractor.extract_mel(audio)
spectrogram = extractor.extract_spectrogram(audio)
mfcc = extractor.extract_mfcc(audio)
f0 = extractor.extract_f0(audio)
energy = extractor.extract_energy(audio)
```

### Phone Extraction

```python
from soniq.processors import PhoneExtractor

# Create extractor
extractor = PhoneExtractor()

# Extract phonemes
result = extractor("hello world", return_ids=True)
print(result["phonemes"])  # "h ə l oʊ w ɝ l d"
print(result["ids"])  # [20, 35, 15, ...]
```

## What's Next?

- Check out the [Tutorials](tutorials/) for more detailed guides
- Read the [API Reference](api/) for complete documentation
- Explore the [Model Zoo](models/) for all supported models
