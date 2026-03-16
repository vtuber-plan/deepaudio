# Soniq Documentation

Welcome to the Soniq documentation.

Soniq is a PyTorch-based audio machine learning library that provides state-of-the-art speech and audio models with a Transformers-style API.

## Contents

### Getting Started

- [Installation](installation.md) - How to install Soniq
- [Quick Start](quickstart.md) - Get started with Soniq in minutes

### Tutorials

- [Vocoder Tutorial](tutorials/vocoder.md) - Train and use neural vocoders
- [TTS Tutorial](tutorials/tts.md) - Text-to-Speech synthesis
- [ASR Tutorial](tutorials/asr.md) - Automatic Speech Recognition

### Models

- [Vocoders](models/vocoders.md) - Neural vocoder models (HiFi-GAN, MelGAN, BigVGAN)
- [TTS Models](models/tts.md) - Text-to-Speech models (VITS, FastSpeech2)
- [ASR Models](models/asr.md) - Automatic Speech Recognition models

### API Reference

- [Models API](api/models.md) - Model classes and configurations
- [Pipelines API](api/pipelines.md) - Audio processing pipelines
- [Training API](api/training.md) - Training utilities with Lightning Fabric
- [Datasets API](api/datasets.md) - Dataset classes and collators

### Guides

- [Configuration Guide](guides/configuration.md) - How to configure models and training
- [Custom Models](guides/custom_models.md) - Create your own models
- [Distributed Training](guides/distributed_training.md) - Multi-GPU training guide

---

## Quick Example

```python
from soniq import AutoModel, MelPipeline

# Load pre-trained model
model = AutoModel.from_pretrained("soniq/hifigan-24k")

# Create pipeline
pipeline = MelPipeline(sample_rate=24000, n_mel=80)

# Run inference
import torchaudio
audio, sr = torchaudio.load("audio.wav")
mel = pipeline(audio)
output = model(mel.unsqueeze(0))
```

## Supported Models

| Task | Models |
|------|--------|
| Vocoder | HiFi-GAN, MelGAN, BigVGAN, WaveNet, DiffWave |
| TTS | VITS, FastSpeech2, VALLE, NaturalSpeech2 |
| ASR | Whisper |
| F0 | CREPE, RMVPE, Harvest |

## License

Soniq is released under the MIT License.

## Acknowledgements

- [Amphion](https://github.com/open-mmlab/Amphion)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Lightning AI](https://lightning.ai/)
