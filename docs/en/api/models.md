# Models API

This page documents the model classes and configurations in Soniq.

## Base Classes

### SoniqPreTrainedModel

```python
from soniq.models import SoniqPreTrainedModel

class MyModel(SoniqPreTrainedModel):
    config_class = MyConfig
    base_model_prefix = "my_model"

    def __init__(self, config):
        super().__init__(config)
        # Your implementation
```

### SoniqConfig

```python
from soniq.models import SoniqConfig

class MyConfig(SoniqConfig):
    model_type = "my_model"

    def __init__(self, param1=128, **kwargs):
        self.param1 = param1
        super().__init__(**kwargs)
```

## Auto Classes

### AutoModel

```python
from soniq.models import AutoModel

# Load from Hugging Face Hub
model = AutoModel.from_pretrained("soniq/hifigan-24k")

# Load from local path
model = AutoModel.from_pretrained("./checkpoints/hifigan")
```

### AutoConfig

```python
from soniq.models import AutoConfig

# Load config
config = AutoConfig.from_pretrained("soniq/hifigan-24k")
```

## Model Architecture

### HiFi-GAN

```python
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig

# Configuration
config = HifiGANConfig(
    inter_channels=128,
    resblock_kernel_sizes=[3, 7, 11, 13],
    resblock_dilation_sizes=[1, 3, 5],
    upsample_rates=[8, 8, 4, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 8, 4],
)

# Model
model = HifiGAN(config)

# Forward pass
mel = torch.randn(1, 80, 100)  # (batch, n_mel, time)
audio = model(mel)  # (batch, 1, time * hop_ratio)
```

### Model Methods

| Method | Description |
|--------|-------------|
| `forward(x)` | Forward pass |
| `infer(x)` | Inference (alias for forward) |
| `from_pretrained(path)` | Load pre-trained weights |
| `save_pretrained(path)` | Save model weights |
| `remove_weight_norm()` | Remove weight normalization |

## Configuration Parameters

### HiFiGANConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inter_channels` | int | 128 | Intermediate channel size |
| `resblock_kernel_sizes` | tuple | (3,7,11,13) | Kernel sizes for residual blocks |
| `resblock_dilation_sizes` | tuple | (1,3,5) | Dilation sizes for residual blocks |
| `upsample_rates` | tuple | (8,8,4,2) | Upsampling rates |
| `upsample_initial_channel` | int | 512 | Initial upsampling channels |
| `upsample_kernel_sizes` | tuple | (16,16,8,4) | Upsampling kernel sizes |
| `use_spectral_norm` | bool | False | Use spectral normalization |
| `lrelu_slope` | float | 0.1 | LeakyReLU slope |

## Saving and Loading

### Save Model

```python
# Save to local path
model.save_pretrained("./my_model")

# Save with config
config.save_pretrained("./my_model")
```

### Load Model

```python
# Load from local path
model = HifiGAN.from_pretrained("./my_model")

# Load from Hugging Face
model = HifiGAN.from_pretrained("soniq/hifigan-24k")
```

## Model Registry

To register a new model:

```python
from soniq.models import AutoModel

# Register your model class
AutoModel.register(MyConfig, MyModel)
```
