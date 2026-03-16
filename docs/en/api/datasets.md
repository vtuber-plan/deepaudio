# Datasets API

This page documents the dataset classes and data loading utilities in Soniq.

## BaseDataset

Base class for audio datasets.

### Usage

```python
from soniq.datasets import BaseDataset

# Create dataset
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={
        "mel": "data/mels",
        "wav": "data/wavs",
    },
    sample_rate=24000,
    use_spkid=True,
    spk2id_path="data/spk2id.json",
)

# Access item
item = dataset[0]
print(item.keys())  # ['id', 'mel', 'wav', 'spk_id', ...]
```

### Metadata Format

The metadata JSON file should contain a list of dictionaries:

```json
[
    {
        "id": "utterance_001",
        "duration": 2.5,
        "speaker": "spk1",
        "text": "Hello world"
    },
    {
        "id": "utterance_002",
        "duration": 3.1,
        "speaker": "spk2",
        "text": "Goodbye"
    }
]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata_path` | str | - | Path to metadata JSON file |
| `feature_dirs` | dict | - | Map of feature type to directory |
| `sample_rate` | int | 24000 | Audio sample rate |
| `use_spkid` | bool | False | Use speaker IDs |
| `spk2id_path` | str | None | Path to speaker mapping |

## BaseCollator

Collator for batching variable-length sequences.

### Usage

```python
from soniq.datasets import BaseCollator

# Create collator
collator = BaseCollator(
    padding_value=0.0,
    batch_first=True,
)

# Collate batch
batch = collator([dataset[0], dataset[1], dataset[2]])
```

### Output Format

The collator returns a dictionary with:
- Padded tensors for each feature type
- Length tensors for each feature

```python
{
    "mel": tensor of shape (batch, time, n_mel),
    "mel_lengths": tensor of shape (batch,),
    "wav": tensor of shape (batch, time),
    "wav_lengths": tensor of shape (batch,),
    "id": ["utterance_001", "utterance_002", ...],
}
```

## DataLoader

### build_dataloader Function

```python
from soniq.datasets import build_dataloader

# Create dataloader
dataloader = build_dataloader(
    dataset,
    collator=BaseCollator(),
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | Dataset | - | PyTorch dataset |
| `collator` | Collator | - | Collator function |
| `batch_size` | int | 32 | Batch size |
| `num_workers` | int | 4 | Number of data loading workers |
| `shuffle` | bool | True | Shuffle data |
| `pin_memory` | bool | True | Pin memory for faster loading |
| `drop_last` | bool | False | Drop last incomplete batch |

## Custom Datasets

### Create Custom Dataset

```python
from soniq.datasets import BaseDataset
import torch
import torchaudio

class CustomDataset(BaseDataset):
    def __getitem__(self, index):
        # Get base item
        item = super().__getitem__(index)

        # Add custom features
        item["custom_feature"] = self.extract_custom(item["wav"])

        return item

    def extract_custom(self, audio):
        # Your feature extraction
        return torch.randn(100)
```

### Custom Collator

```python
from soniq.datasets import BaseCollator

class CustomCollator(BaseCollator):
    def __call__(self, batch):
        # Get base collation
        batch_dict = super().__call__(batch)

        # Add custom processing
        batch_dict["custom_batch"] = self.process_custom(batch_dict)

        return batch_dict

    def process_custom(self, batch_dict):
        # Your custom batch processing
        return batch_dict["mel"].mean(dim=1)
```

## Variable Length Batching

For efficient training with variable-length audio:

```python
from torch.utils.data import DataLoader
from soniq.datasets import BaseDataset, BaseCollator

# Use a sampler that groups similar-length sequences
dataset = BaseDataset(...)

# Dynamic batch size based on frames
def num_tokens_fn(idx):
    return dataset.get_num_frames(idx)

# Create sampler (implementation depends on your needs)
sampler = DynamicBatchSampler(
    dataset,
    max_tokens=10000,  # Max frames per batch
    max_sentences=32,  # Max sentences per batch
)

dataloader = DataLoader(
    dataset,
    collate_fn=BaseCollator(),
    batch_sampler=sampler,
)
```
