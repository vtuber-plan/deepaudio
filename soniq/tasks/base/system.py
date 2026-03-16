# coding=utf-8
"""Base task system class."""

from typing import Any, Dict, List, Union
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    logs: Dict[str, Any]


class BaseTaskSystem(nn.Module):
    """Base class for task systems."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def training_step(self, batch, batch_idx: int) -> StepOutput:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> StepOutput:
        raise NotImplementedError

    def inference_step(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        raise NotImplementedError

    def on_train_start(self) -> None: pass
    def on_train_end(self) -> None: pass
    def on_epoch_start(self, epoch: int) -> None: pass
    def on_epoch_end(self, epoch: int) -> None: pass
