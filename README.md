# Soniq

**Soniq** is a PyTorch-based audio machine learning library that provides state-of-the-art speech and audio models with a Transformers-style API.

[![PyPI](https://img.shields.io/pypi/v/soniq.svg)](https://pypi.org/project/soniq/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Transformers-style API** - Familiar interface for Hugging Face users
- **Lightning Fabric** - Flexible and efficient training with distributed support
- **Rich Model Zoo** - Vocoders, TTS, ASR, SVC, and more
- **Multi-language Documentation** - Documentation in English and Chinese

## Installation

### From PyPI

```bash
pip install soniq
```

### From Source

```bash
git clone https://github.com/vtuber-plan/soniq.git
cd soniq
pip install -e .
```

## Quick Start

### Load Pre-trained Model

```python
from soniq import AutoModel, MelPipeline

# Load model
model = AutoModel.from_pretrained("soniq/hifigan-24k")

# Create pipeline
pipeline = MelPipeline(sample_rate=24000, n_mel=80)

# Run inference
import torchaudio
audio, sr = torchaudio.load("audio.wav")
mel = pipeline(audio)
output = model(mel.unsqueeze(0))
```

### Training

```python
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader

# Model
config = HifiGANConfig()
model = HifiGAN(config)

# Dataset
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
)
dataloader = build_dataloader(dataset, BaseCollator(), batch_size=32)

# Trainer
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(model, dataloader, optimizer=optimizer)
```

## Supported Models

### Vocoder

| Model | Description | Sample Rate |
|-------|-------------|-------------|
| HiFi-GAN | High-fidelity GAN vocoder | 16k, 24k, 44k, 48k |
| MelGAN | Fast GAN vocoder | 16k, 24k |
| BigVGAN | Universal neural vocoder | 24k, 44k |
| WaveNet | Autoregressive vocoder | 16k, 24k |

### Text-to-Speech

| Model | Description |
|-------|-------------|
| VITS | Conditional Variational Autoencoder |
| FastSpeech2 | Fast and high-quality TTS |
| VALLE | VALL-E neural codec language model |

### Other

| Task | Models |
|------|--------|
| ASR | Whisper |
| F0 Detection | CREPE, RMVPE, Harvest |
| Content Encoder | HubERT, ContentVec |

## Documentation

- [English Documentation](docs/en/README.md)
- [中文文档](docs/zh/README.md)

### Quick Links

- [Installation Guide](docs/en/installation.md)
- [Quick Start](docs/en/quickstart.md)
- [Tutorials](docs/en/tutorials/)
- [API Reference](docs/en/api/)

## Project Structure

```
soniq/
├── soniq/                  # Main package
│   ├── models/             # Model definitions
│   ├── pipelines/          # Audio processing pipelines
│   ├── features/           # Feature extraction
│   ├── datasets/           # Dataset classes
│   ├── training/           # Training utilities
│   ├── processors/         # Data preprocessing
│   ├── modules/            # Reusable neural modules
│   ├── text/               # Text processing
│   └── utils/              # Utility functions
├── docs/                   # Documentation
│   ├── en/                 # English docs
│   └── zh/                 # Chinese docs
├── config/                 # Configuration files
├── bins/                   # Training scripts
└── examples/               # Example code
```

## License

Soniq is released under the MIT License.

## Acknowledgements

This project draws inspiration and ideas from the following excellent projects:

- [Amphion](https://github.com/open-mmlab/Amphion) - Audio, Music, and Speech Generation Toolkit
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art Machine Learning library
- [Lightning AI](https://lightning.ai/) - PyTorch Lightning and Lightning Fabric

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/vtuber-plan/soniq/issues)
