# Pipelines API

This page documents the audio processing pipelines in Soniq.

## AudioPipeline

Audio loading and preprocessing pipeline.

### Usage

```python
from soniq.pipelines import AudioPipeline

# Create pipeline
pipeline = AudioPipeline(
    sample_rate=24000,  # Target sample rate
    normalize=True,     # Normalize to [-1, 1]
    mono=True,          # Convert to mono
)

# Load from file
audio = pipeline("path/to/audio.wav")

# Process tensor
audio = pipeline(audio_tensor, src_sample_rate=16000)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | int | 24000 | Target sample rate |
| `normalize` | bool | True | Normalize audio |
| `mono` | bool | True | Convert to mono |

## MelPipeline

Mel spectrogram extraction pipeline.

### Usage

```python
from soniq.pipelines import MelPipeline

# Create pipeline
pipeline = MelPipeline(
    sample_rate=24000,
    n_fft=1024,
    n_mel=80,
    hop_length=256,
    win_length=1024,
    f_min=0.0,
    f_max=12000.0,
    log_scale=True,
)

# Extract mel spectrogram
mel = pipeline(audio)  # (n_mel, time)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | int | 24000 | Audio sample rate |
| `n_fft` | int | 1024 | FFT window size |
| `n_mel` | int | 80 | Number of mel bins |
| `hop_length` | int | 256 | Hop length |
| `win_length` | int | None | Window length (default: n_fft) |
| `f_min` | float | 0.0 | Minimum frequency |
| `f_max` | float | None | Maximum frequency |
| `log_scale` | bool | True | Apply log scaling |

## Common Patterns

### Pipeline Chaining

```python
# Chain pipelines
audio_pipeline = AudioPipeline(sample_rate=24000)
mel_pipeline = MelPipeline(sample_rate=24000)

# Process
audio = audio_pipeline("input.wav")
mel = mel_pipeline(audio)
```

### Batch Processing

```python
# Process multiple files
files = ["a.wav", "b.wav", "c.wav"]
audios = [audio_pipeline(f) for f in files]

# Stack for batch
batch = torch.stack(audios)
```

## Custom Pipelines

Create a custom pipeline by subclassing:

```python
from soniq.pipelines import AudioPipeline

class CustomPipeline(AudioPipeline):
    def __call__(self, audio, **kwargs):
        # Custom preprocessing
        audio = super().__call__(audio, **kwargs)
        # Additional processing
        audio = self.custom_process(audio)
        return audio

    def custom_process(self, audio):
        # Your implementation
        return audio
```
