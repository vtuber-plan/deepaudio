# coding=utf-8
"""Output dataclasses for tasks."""

from dataclasses import dataclass
from typing import Any, Dict
import torch


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    logs: Dict[str, Any]


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    results: Dict[str, Any]


@dataclass
class InferOutput:
    results: Dict[str, Any]
