# coding=utf-8
"""Base configuration class for Soniq models."""

from transformers import PretrainedConfig
from typing import Any, Dict


class SoniqModelConfig(PretrainedConfig):
    """Base configuration for all Soniq models."""

    model_type = "soniq"

    def __init__(self, initializer_range: float = 0.02, **kwargs):
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def model_config(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k not in ['return_dict']}
