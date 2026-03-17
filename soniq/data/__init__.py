# coding=utf-8
"""
Data utilities for Soniq.

This module provides:
- BaseDataset: 通用数据集基类
- ManifestDataset: 基于 manifest 文件的数据集
- BaseCollator: 数据整理器
- DataLoader utilities
- Manifest utilities
"""

from typing import Any, Callable, Dict, List, Optional
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# Manifest Utilities
# ============================================================================

def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Load manifest from JSON or JSONL file.

    Args:
        path: Path to manifest file

    Returns:
        List of manifest entries
    """
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        return json.load(f)


def save_manifest(data: List[Dict[str, Any]], path: str) -> None:
    """Save manifest to file.

    Args:
        data: List of manifest entries
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# Dataset Classes
# ============================================================================

class BaseDataset(Dataset):
    """
    Base dataset class for Soniq.

    This class provides common functionality for audio datasets including:
    - Metadata loading from JSON/JSONL
    - Feature file path management
    - Speaker ID mapping
    - Audio resampling

    Args:
        metadata_path: Path to metadata JSON file
        feature_dirs: Dictionary of feature type to directory path
        sample_rate: Audio sample rate
        use_spkid: Whether to use speaker IDs
        spk2id_path: Path to speaker-to-ID mapping file

    Example:
        ```python
        dataset = BaseDataset(
            metadata_path="data/train.json",
            feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
        )
        item = dataset[0]
        ```
    """

    def __init__(
        self,
        metadata_path: str,
        feature_dirs: Optional[Dict[str, str]] = None,
        sample_rate: int = 24000,
        use_spkid: bool = False,
        spk2id_path: Optional[str] = None,
    ):
        self.metadata_path = metadata_path
        self.feature_dirs = feature_dirs or {}
        self.sample_rate = sample_rate
        self.use_spkid = use_spkid

        # Load metadata
        self.metadata = load_manifest(metadata_path)

        # Load speaker mapping if needed
        if use_spkid and spk2id_path is not None:
            with open(spk2id_path, "r", encoding="utf-8") as f:
                self.spk2id = json.load(f)
        else:
            self.spk2id = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.metadata[index].copy()

        # Load features
        for feature_type, feature_dir in self.feature_dirs.items():
            feature_path = os.path.join(
                feature_dir,
                f"{item['id']}.{self._get_extension(feature_type)}"
            )
            if os.path.exists(feature_path):
                item[feature_type] = self._load_feature(feature_path, feature_type)

        # Add speaker ID if needed
        if self.use_spkid and self.spk2id is not None:
            item["spk_id"] = self.spk2id.get(item.get("speaker", "unknown"), 0)

        return item

    def _get_extension(self, feature_type: str) -> str:
        """Get file extension for feature type."""
        extensions = {
            "wav": "wav",
            "audio": "wav",
            "mel": "npy",
            "spectrogram": "npy",
            "f0": "npy",
            "energy": "npy",
            "phone": "txt",
        }
        return extensions.get(feature_type, "npy")

    def _load_feature(self, path: str, feature_type: str) -> torch.Tensor:
        """Load feature from file."""
        if feature_type in ["wav", "audio"]:
            import torchaudio
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            return waveform
        else:
            return torch.from_numpy(np.load(path))


class ManifestDataset(Dataset):
    """
    Dataset that reads from a manifest file with custom readers.

    Args:
        manifest_path: Path to manifest file
        readers: Dictionary mapping keys to reader functions
        processors: Optional list of sample processors
        filter_fn: Optional function to filter samples

    Example:
        ```python
        import soundfile as sf

        readers = {
            "audio": lambda p: torch.from_numpy(sf.read(p)[0]),
        }
        dataset = ManifestDataset("manifest.json", readers)
        ```
    """

    def __init__(
        self,
        manifest_path: str,
        readers: Dict[str, Callable],
        processors: Optional[List[Callable]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ):
        self.manifest_path = manifest_path
        self.readers = readers
        self.processors = processors or []
        self.filter_fn = filter_fn
        self.metadata = load_manifest(manifest_path)

        if self.filter_fn:
            self.metadata = [m for m in self.metadata if self.filter_fn(m)]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.metadata[index].copy()
        sample = self._apply_readers(entry, index)
        for processor in self.processors:
            sample = processor(sample)
        return sample

    def _apply_readers(self, entry: Dict[str, Any], index: int) -> Dict[str, Any]:
        sample = {"id": entry.get("id", str(index))}
        for key, reader in self.readers.items():
            if key in entry:
                sample[key] = reader(entry[key])
        for key in ["speaker", "duration", "text", "language"]:
            if key in entry:
                sample[key] = entry[key]
        return sample


# ============================================================================
# Collator Classes
# ============================================================================

class BaseCollator:
    """
    Base collator for batching dataset items.

    Handles padding and batching of variable-length sequences.

    Args:
        padding_value: Value for padding
        batch_first: If True, output has batch as first dimension
        pad_keys: Specific keys to pad (if None, auto-detect sequences)

    Example:
        ```python
        collator = BaseCollator(pad_keys=["audio", "mel"])
        batch = collator([dataset[0], dataset[1]])
        ```
    """

    def __init__(
        self,
        padding_value: float = 0.0,
        batch_first: bool = True,
        pad_keys: Optional[List[str]] = None,
    ):
        self.padding_value = padding_value
        self.batch_first = batch_first
        self.pad_keys = pad_keys or []

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}

        # Collect all keys
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        result = {}
        for key in all_keys:
            values = [s[key] for s in batch if key in s]
            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                if key in self.pad_keys or self._needs_padding(values):
                    result[key] = pad_sequence(
                        values,
                        batch_first=self.batch_first,
                        padding_value=self.padding_value
                    )
                    result[f"{key}_lengths"] = torch.tensor([
                        v.shape[0] if self.batch_first else v.shape[1]
                        for v in values
                    ])
                else:
                    result[key] = torch.stack(values)
            else:
                result[key] = values

        return result

    def _needs_padding(self, values: List[torch.Tensor]) -> bool:
        """Check if tensors need padding (different shapes)."""
        if len(values) < 2:
            return False
        shapes = set(v.shape for v in values)
        return len(shapes) > 1


# ============================================================================
# DataLoader Utilities
# ============================================================================

def build_dataloader(
    dataset: Dataset,
    collator: BaseCollator,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Build DataLoader for dataset.

    Args:
        dataset: PyTorch dataset
        collator: Collator function
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


__all__ = [
    # Manifest utilities
    "load_manifest",
    "save_manifest",
    # Datasets
    "BaseDataset",
    "ManifestDataset",
    # Collators
    "BaseCollator",
    # DataLoader
    "build_dataloader",
]