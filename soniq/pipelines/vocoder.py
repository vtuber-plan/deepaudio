# coding=utf-8
"""Vocoder pipeline."""

from typing import Any, Dict
import torch
from .base import BasePipeline


class VocoderPipeline(BasePipeline):
    """Pipeline for vocoder inference."""

    def preprocess(self, inputs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            return {"acoustic_features": inputs}
        return {"acoustic_features": torch.tensor(inputs)}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.model.synthesize(inputs["acoustic_features"])
        return {"waveform": output.waveform}

    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["waveform"].squeeze(0).cpu()
