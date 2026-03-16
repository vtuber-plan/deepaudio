# coding=utf-8
"""Base TTS model class for Soniq."""

from typing import Dict, Any, Optional
import torch
from torch import nn

from transformers.utils import logging
from soniq.models.base.modeling_base import SoniqModel
from soniq.models.base.outputs import TTSOutput
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class BaseTTSModel(SoniqModel):
    """
    Abstract base class for all TTS models in Soniq.

    This class provides common functionality for text-to-speech models including:
    - Text/phoneme input processing
    - Mel spectrogram or waveform generation
    - Speaker conditioning (optional)
    """

    config_class = SoniqModelConfig
    base_model_prefix = "tts"
    supports_gradient_checkpointing = False

    def __init__(self, config: SoniqModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def synthesize(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        **kwargs,
    ) -> TTSOutput:
        """
        Synthesize speech from text.

        Args:
            text_ids: Text token IDs of shape (batch, seq_len).
            text_lengths: Text sequence lengths of shape (batch,).
            speaker_id: Optional speaker ID for multi-speaker models.
            **kwargs: Additional arguments for model-specific inference.

        Returns:
            TTSOutput containing generated audio or mel spectrogram.
        """
        raise NotImplementedError("synthesize method must be implemented by subclass")

    def infer(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference method for TTS.

        Args:
            text_ids: Text token IDs of shape (batch, seq_len).
            text_lengths: Text sequence lengths of shape (batch,).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing model outputs.
        """
        raise NotImplementedError("infer method must be implemented by subclass")

    def forward(
        self,
        data: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing model inputs.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing model outputs and losses.
        """
        raise NotImplementedError("forward method must be implemented by subclass")

    def compute_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Dictionary containing loss terms.
        """
        raise NotImplementedError("compute_loss method must be implemented by subclass")
