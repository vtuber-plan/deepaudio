# coding=utf-8
"""Base processor class."""

from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base class for processors."""

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
