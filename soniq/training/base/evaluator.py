# coding=utf-8
"""Base evaluator class."""

from typing import Dict, List
from torch.utils.data import DataLoader


class BaseEvaluator:
    """Base class for task evaluators."""

    def __init__(self, config):
        self.config = config
        self.metrics_history: List[Dict[str, float]] = []

    def evaluate(self, system, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        self.metrics_history = []

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        aggregated = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        return aggregated
