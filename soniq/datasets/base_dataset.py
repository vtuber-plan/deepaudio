# coding=utf-8
"""
Base dataset classes for Soniq.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BaseDataset(Dataset):
    """
    Base dataset class for Soniq.

    This class provides common functionality for audio datasets including:
        - Metadata loading
        - Feature file path management
        - Utterance indexing

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
        feature_dirs: Dict[str, str],
        sample_rate: int = 24000,
        use_spkid: bool = False,
        spk2id_path: Optional[str] = None,
    ):
        """
        Initialize BaseDataset.

        Args:
            metadata_path: Path to metadata JSON file.
            feature_dirs: Dictionary of feature type to directory path.
            sample_rate: Audio sample rate.
            use_spkid: Whether to use speaker IDs.
            spk2id_path: Path to speaker-to-ID mapping file.
        """
        self.metadata_path = metadata_path
        self.feature_dirs = feature_dirs
        self.sample_rate = sample_rate
        self.use_spkid = use_spkid

        # Load metadata
        self.metadata = self._load_metadata()

        # Load speaker mapping if needed
        if use_spkid and spk2id_path is not None:
            self.spk2id = self._load_spk2id(spk2id_path)
        else:
            self.spk2id = None

    def _load_metadata(self) -> List[Dict]:
        """Load metadata from JSON file."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def _load_spk2id(self, path: str) -> Dict[str, int]:
        """Load speaker-to-ID mapping."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Args:
            index: Item index.

        Returns:
            Dictionary containing features and metadata.
        """
        item = self.metadata[index].copy()

        # Load features
        features = {}
        for feature_type, feature_dir in self.feature_dirs.items():
            feature_path = os.path.join(feature_dir, f"{item['id']}.{self._get_extension(feature_type)}")
            if os.path.exists(feature_path):
                features[feature_type] = self._load_feature(feature_path, feature_type)
            else:
                features[feature_type] = None

        item.update(features)

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
            # Load numpy array
            return torch.from_numpy(np.load(path))

    def get_num_frames(self, index: int) -> int:
        """
        Get number of frames for item at index.

        Args:
            index: Item index.

        Returns:
            Number of frames.
        """
        item = self.metadata[index]
        if "duration" in item:
            # Estimate frames from duration
            return int(item["duration"] * self.sample_rate / 256)
        return 100  # Default

    @property
    def num_frame_indices(self) -> List[int]:
        """Return list of indices for frame-based sampling."""
        return list(range(len(self)))


class BaseCollator:
    """
    Base collator for batching dataset items.

    This class handles padding and batching of variable-length sequences.

    Example:
        ```python
        collator = BaseCollator()
        batch = collator([dataset[0], dataset[1]])
        ```
    """

    def __init__(
        self,
        padding_value: float = 0.0,
        batch_first: bool = True,
    ):
        """
        Initialize BaseCollator.

        Args:
            padding_value: Value for padding.
            batch_first: If True, output has batch as first dimension.
        """
        self.padding_value = padding_value
        self.batch_first = batch_first

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch items.

        Args:
            batch: List of batch items.

        Returns:
            Collated batch dictionary.
        """
        if len(batch) == 0:
            return {}

        # Collect keys
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())

        # Group tensor and non-tensor items
        batch_dict = {}
        for key in all_keys:
            values = [item[key] for item in batch if key in item]

            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                # Pad tensors
                batch_dict[key] = self._pad_tensors(values)
                # Add length tensor for this key
                batch_dict[f"{key}_lengths"] = torch.tensor([v.shape[0] for v in values])
            else:
                # Keep non-tensors as list
                batch_dict[key] = values

        return batch_dict

    def _pad_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad tensors to same length.

        Args:
            tensors: List of tensors with varying lengths.

        Returns:
            Padded tensor.
        """
        max_length = max(t.shape[0] for t in tensors)

        # Handle different tensor dimensions
        if tensors[0].dim() == 1:
            output = tensors[0].new_full((len(tensors), max_length), self.padding_value)
            for i, t in enumerate(tensors):
                output[i, : t.shape[0]] = t
        elif tensors[0].dim() == 2:
            max_dim1 = max(t.shape[1] for t in tensors)
            output = tensors[0].new_full(
                (len(tensors), max_length, max_dim1), self.padding_value
            )
            for i, t in enumerate(tensors):
                output[i, : t.shape[0], : t.shape[1]] = t
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensors[0].dim()}")

        return output


def build_dataloader(
    dataset: Dataset,
    collator: BaseCollator,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build DataLoader for dataset.

    Args:
        dataset: PyTorch dataset.
        collator: Collator function.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
