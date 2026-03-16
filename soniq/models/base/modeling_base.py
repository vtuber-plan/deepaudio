# coding=utf-8
"""Base model class for Soniq models."""

from typing import Dict, Optional
import torch
from torch import nn
from transformers import PreTrainedModel
from .configuration_base import SoniqModelConfig


class SoniqModel(PreTrainedModel):
    """Abstract base class for all Soniq models."""

    config_class = SoniqModelConfig
    base_model_prefix = "soniq"
    supports_gradient_checkpointing = True

    def __init__(self, config: SoniqModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def infer(self, **inputs):
        self.eval()
        with torch.no_grad():
            return self.forward(**inputs)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, 'initializer_range', 0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
