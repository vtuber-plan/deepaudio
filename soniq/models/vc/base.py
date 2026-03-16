# coding=utf-8
"""Base model class for voice conversion models."""

from abc import abstractmethod
from typing import Dict, Any, Optional
import torch
from torch import nn

from transformers import PreTrainedModel
from soniq.models.base.configuration_base import SoniqModelConfig
from soniq.models.base.outputs import VCOutput


class BaseVCModel(PreTrainedModel):
    """
    Base class for voice conversion models.

    All VC models should inherit from this class and implement the required methods.

    Example:
        ```python
        class MyVC(BaseVCModel):
            config_class = MyVCConfig

            def voice_conversion(self, source_audio, target_speaker):
                # Convert source to target speaker voice
                ...
        ```
    """

    config_class = SoniqModelConfig
    base_model_prefix = "vc"
    supports_gradient_checkpointing = True

    def __init__(self, config: SoniqModelConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing source and target audio/features.

        Returns:
            Dictionary containing converted features and loss.
        """
        pass

    @abstractmethod
    def voice_conversion(
        self,
        source: torch.Tensor,
        target_speaker: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert source to target speaker voice.

        Args:
            source: Source audio or features.
            target_speaker: Target speaker embedding or audio.
            **kwargs: Additional arguments.

        Returns:
            Converted features or audio.
        """
        pass

    def synthesize(
        self,
        source: torch.Tensor,
        target_speaker: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Synthesize converted speech.

        Args:
            source: Source audio or features.
            target_speaker: Target speaker embedding or audio.
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted audio.
        """
        converted = self.voice_conversion(source, target_speaker, **kwargs)
        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )

    def reconstruct(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Reconstruct/convert audio.

        Args:
            source_audio: Source audio waveform.
            target_audio: Target audio waveform.
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted audio.
        """
        converted = self.voice_conversion(source_audio, target_audio, **kwargs)
        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return getattr(self.config, "sample_rate", 24000)
