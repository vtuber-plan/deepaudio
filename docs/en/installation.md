# Installation

This guide explains how to install Soniq.

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- torchaudio >= 0.14

## Installation Methods

### Method 1: Install from PyPI

```bash
pip install soniq
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/vtuber-plan/soniq.git
cd soniq

# Install the package
pip install -e .
```

### Method 3: Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

## Optional Dependencies

### For High-Quality Audio Processing

```bash
pip install librosa soundfile
```

### For G2P (Grapheme-to-Phoneme)

```bash
pip install g2p_en  # English
pip install pypinyin  # Chinese
```

### For Training

```bash
pip install lightning tensorboard wandb
```

## Verify Installation

```python
import soniq
print(soniq.__version__)
# Should print: 0.1.0
```

## Platform Support

| Platform | Supported |
|----------|-----------|
| Linux | ✅ |
| macOS | ✅ |
| Windows | ✅ |
| GPU (CUDA) | ✅ |
| CPU | ✅ |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use gradient accumulation:

```python
trainer = FabricTrainer(
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
)
```

### Import Errors

Make sure you have installed all required dependencies:

```bash
pip install torch torchaudio transformers lightning
```

### Audio Loading Issues

Ensure your audio files are in a supported format (WAV, FLAC, MP3).
