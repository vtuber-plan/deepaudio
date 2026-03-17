# coding=utf-8
"""Output dataclasses for tasks."""

from dataclasses import dataclass, field
from typing import Any, Dict
import torch


@dataclass
class StepOutput:
    """Training/validation step output."""

    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    results: Dict[str, Any]


@dataclass
class InferOutput:
    results: Dict[str, Any]
