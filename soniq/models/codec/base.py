# coding=utf-8
"""Base model class for neural audio codecs."""

from abc import abstractmethod
from typing import Dict, Any, Optional
import torch
from torch import nn

from transformers import PreTrainedModel
from soniq.models.base.configuration_base import SoniqModelConfig
from soniq.models.base.outputs import CodecOutput


class BaseCodecModel(PreTrainedModel):
    """
    Base class for neural audio codec models.

    All codec models (SpeechTokenizer, FACodec, etc.) should inherit from this class
    and implement the required methods.

    Example:
        ```python
        class MyCodec(BaseCodecModel):
            config_class = MyCodecConfig

            def encode(self, audio):
                # Encode audio to discrete codes
                ...

            def decode(self, codes):
                # Decode codes to audio
                ...
        ```
    """

    config_class = SoniqModelConfig
    base_model_prefix = "codec"
    supports_gradient_checkpointing = True

    def __init__(self, config: SoniqModelConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def encode(self, audio: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Encode audio to discrete codes.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing quantization codes.
        """
        pass

    @abstractmethod
    def decode(self, codes: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Decode codes to audio.

        Args:
            codes: Dictionary containing quantization codes.
            **kwargs: Additional arguments.

        Returns:
            Reconstructed audio waveform (batch, 1, seq_len).
        """
        pass

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - audio: Input audio waveform (batch, 1, seq_len)
                - audio_lengths: Audio lengths (batch,)

        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed audio
                - codes: Quantization codes
                - loss: Reconstruction/commitment losses
        """
        pass

    def reconstruct(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> CodecOutput:
        """
        Reconstruct audio from input audio.

        Args:
            audio: Input audio waveform (batch, seq_len) or (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with reconstructed audio.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        output = self({"audio": audio})
        return CodecOutput(
            reconstructed=output.get("reconstructed"),
            codes=output.get("codes"),
        )

    def synthesize(
        self,
        codes: Dict[str, torch.Tensor],
        **kwargs,
    ) -> CodecOutput:
        """
        Synthesize audio from codes.

        Args:
            codes: Dictionary containing quantization codes.
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with synthesized audio.
        """
        waveform = self.decode(codes, **kwargs)
        return CodecOutput(
            waveform=waveform,
            codes=codes,
        )

