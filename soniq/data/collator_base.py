# coding=utf-8
"""Base collator class."""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence


class BaseCollator:
    """Base class for collating samples."""

    def __init__(self, padding_value: float = 0.0, batch_first: bool = True,
                 pad_keys: Optional[List[str]] = None):
        self.padding_value = padding_value
        self.batch_first = batch_first
        self.pad_keys = pad_keys or []

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        result = {}
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        for key in all_keys:
            values = [s[key] for s in batch if key in s]
            if len(values) == 0:
                continue
            if isinstance(values[0], torch.Tensor):
                if key in self.pad_keys or self._is_sequence(values):
                    result[key] = pad_sequence(values, batch_first=self.batch_first,
                                               padding_value=self.padding_value)
                    result[f"{key}_lengths"] = torch.tensor(
                        [v.shape[0] if self.batch_first else v.shape[1] for v in values]
                    )
                else:
                    result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    def _is_sequence(self, values: List[torch.Tensor]) -> bool:
        if len(values) < 2:
            return False
        return len(set(v.shape for v in values)) > 1
