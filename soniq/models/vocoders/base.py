# coding=utf-8
"""Base class for vocoder models."""

from typing import Optional
import torch
from ..base.modeling_base import SoniqModel
from ..base.outputs import VocoderOutput


class BaseVocoderModel(SoniqModel):
    """Base class for all vocoder models."""

    def synthesize(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
        raise NotImplementedError("Subclasses must implement synthesize()")

    def forward(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
        return self.synthesize(acoustic_features, **kwargs)
