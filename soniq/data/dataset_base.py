# coding=utf-8
"""Base dataset class."""

from typing import Any, Callable, Dict, List, Optional
import json
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Dataset that reads from a manifest file."""

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
        self.metadata = self._load_manifest()

        if self.filter_fn:
            self.metadata = [m for m in self.metadata if self.filter_fn(m)]

    def _load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            if self.manifest_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            return json.load(f)

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
