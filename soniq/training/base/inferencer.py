# coding=utf-8
"""Base inferencer class."""

from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader


class BaseInferencer:
    """Base class for task inferencers."""

    def __init__(self, config, system):
        self.config = config
        self.system = system
        self.system.eval()

    @torch.no_grad()
    def infer(self, dataloader: DataLoader, **kwargs) -> List[Dict[str, Any]]:
        results = []
        for batch in dataloader:
            batch_results = self.infer_batch(batch, **kwargs)
            results.extend(batch_results)
        return results

    def infer_batch(self, batch, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError
