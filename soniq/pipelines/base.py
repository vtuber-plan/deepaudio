# coding=utf-8
"""Base pipeline class."""

from typing import Any, Dict
from abc import ABC, abstractmethod
import torch


class BasePipeline(ABC):
    """Base class for inference pipelines."""

    def __init__(self, model, processor=None, config=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.model.eval()

    @abstractmethod
    def preprocess(self, inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError

    def __call__(self, inputs, **kwargs) -> Any:
        processed = self.preprocess(inputs)
        outputs = self.forward(processed)
        return self.postprocess(outputs)
