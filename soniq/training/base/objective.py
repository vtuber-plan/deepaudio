# coding=utf-8
"""Base objective class."""

from typing import Dict, Any
import torch
from torch import nn


class BaseObjective(nn.Module):
    """Base class for task objectives."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions, targets, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
