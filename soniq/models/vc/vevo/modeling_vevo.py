# coding=utf-8
"""
Vevo: Voice Conversion with Semantic Tokens and Flow Matching

Vevo uses:
- Semantic tokens for content representation
- Flow matching for high-quality mel spectrogram generation
- Speaker embeddings for timbre control
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple
import torchaudio

from transformers.utils import logging
from soniq.models.base.outputs import VCOutput
from soniq.models.vc.base import BaseVCModel
from soniq.models.vc.vevo.configuration_vevo import VevoConfig
from soniq.models.vc.vevo.vevo_components import (
    SemanticEncoder,
    FlowMatchingDecoder,
    MelDecoder,
)


logger = logging.get_logger(__name__)


class Vevo(BaseVCModel):
    """
    Vevo: Voice Conversion with Semantic Tokens and Flow Matching.

    This model converts speech from one speaker to another while preserving
    the linguistic content.

    Example:
        ```python
        config = VevoConfig()
        model = Vevo(config)

        # Training
        batch = {
            "source_audio": source_audio,
            "target_audio": target_audio,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
        }
        output = model(batch)

        # Inference
        converted = model.voice_conversion(source_audio, target_speaker_embedding)
        ```
    """

    config_class = VevoConfig
    base_model_prefix = "vevo"
    supports_gradient_checkpointing = True

    def __init__(self, config: VevoConfig):
        super().__init__(config)
        self.config = config

        # Semantic encoder
        self.semantic_encoder = SemanticEncoder(
            vocab_size=config.codebook_size,
            dim=config.codebook_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )

        # Flow matching decoder
        self.flow_decoder = FlowMatchingDecoder(
            in_dim=config.codebook_dim + config.hidden_dim,  # semantic + speaker
            out_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )

        # Mel decoder
        self.mel_decoder = MelDecoder(
            dim=config.hidden_dim,
            out_dim=config.n_mel,
        )

        # Speaker embedding projection (from n_mel to hidden_dim)
        self.speaker_proj = nn.Linear(config.n_mel, config.hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def _encode_speaker(self, target_audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from target audio."""
        # Simple speaker embedding via mean pooling of mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
        )(target_audio.squeeze(1))
        mel = (mel + 1e-9).log()
        speaker_emb = mel.mean(dim=2)  # (batch, n_mel)
        speaker_emb = self.speaker_proj(speaker_emb)  # (batch, hidden_dim)
        return speaker_emb

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - source_codes: Source semantic codes (batch, seq_len)
                - target_audio: Target audio waveform (batch, 1, seq_len)
                - source_lengths: Source code lengths (batch,)
                - target_lengths: Target audio lengths (batch,)

        Returns:
            Dictionary containing:
                - mel_pred: Predicted mel spectrogram
                - speaker_pred: Predicted speaker embedding
                - loss: Total loss
        """
        source_codes = data["source_codes"]
        target_audio = data["target_audio"]
        source_lengths = data.get("source_lengths", None)
        target_lengths = data.get("target_lengths", None)

        batch_size, seq_len = source_codes.shape

        # Create mask
        if source_lengths is not None:
            mask = torch.arange(seq_len, device=source_codes.device).unsqueeze(0) < source_lengths.unsqueeze(1)
        else:
            mask = torch.ones_like(source_codes, dtype=torch.bool)

        # Extract speaker embedding from target
        speaker_emb = self._encode_speaker(target_audio)  # (batch, hidden_dim)

        # Encode semantic content
        semantic_out = self.semantic_encoder(source_codes, mask)  # (batch, seq_len, codebook_dim)

        # Add speaker embedding
        speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([semantic_out, speaker_emb_expanded], dim=-1)

        # Decode with flow matching
        flow_out = self.flow_decoder(combined, mask)  # (batch, seq_len, hidden_dim)

        # Predict mel spectrogram
        mel_pred = self.mel_decoder(flow_out)  # (batch, seq_len, n_mel)

        # Get target mel for loss computation
        target_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
        )(target_audio.squeeze(1))
        target_mel = (target_mel + 1e-9).log().transpose(1, 2)  # (batch, time, n_mel)

        # Crop to match length
        max_len = mel_pred.shape[1]
        if target_mel.shape[1] > max_len:
            target_mel = target_mel[:, :max_len, :]
        elif target_mel.shape[1] < max_len:
            # Pad if target is shorter
            pad_len = max_len - target_mel.shape[1]
            target_mel = F.pad(target_mel, (0, 0, 0, pad_len))

        # Compute mel reconstruction loss
        mel_loss = F.l1_loss(mel_pred, target_mel)

        return {
            "mel_pred": mel_pred,
            "mel_target": target_mel,
            "loss": mel_loss,
        }

    @torch.no_grad()
    def voice_conversion(
        self,
        source_codes: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source semantic codes to target speaker voice.

        Args:
            source_codes: Source semantic codes (batch, seq_len).
            target_speaker_embedding: Target speaker embedding (batch, hidden_dim).

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        batch_size, seq_len = source_codes.shape

        # Encode semantic content
        semantic_out = self.semantic_encoder(source_codes)

        # Add speaker embedding
        speaker_emb_expanded = target_speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([semantic_out, speaker_emb_expanded], dim=-1)

        # Decode
        flow_out = self.flow_decoder(combined)
        mel_pred = self.mel_decoder(flow_out)

        return mel_pred

    @torch.no_grad()
    def convert_audio(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source audio to target speaker voice.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_audio: Target audio waveform (batch, 1, seq_len).

        Returns:
            Converted mel spectrogram.
        """
        # Extract speaker embedding from target
        speaker_emb = self._encode_speaker(target_audio)

        # For now, use a simple mel conversion (full implementation would need semantic encoder)
        source_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
        )(source_audio.squeeze(1))
        source_mel = (source_mel + 1e-9).log().transpose(1, 2)

        # Apply speaker conversion via feature statistics matching
        target_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
        )(target_audio.squeeze(1))
        target_mel = (target_mel + 1e-9).log()

        # Simple statistics matching (placeholder for full VC)
        source_mean = source_mel.mean(dim=1, keepdim=True)
        source_std = source_mel.std(dim=1, keepdim=True) + 1e-9
        target_mean = target_mel.mean(dim=2, keepdim=True)
        target_std = target_mel.std(dim=2, keepdim=True) + 1e-9

        converted = (source_mel - source_mean) / source_std * target_std + target_mean

        return converted.transpose(1, 2)

    def synthesize(
        self,
        source_codes: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Synthesize converted speech from codes.

        Args:
            source_codes: Source semantic codes (batch, seq_len).
            target_speaker_embedding: Target speaker embedding (batch, hidden_dim).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted features.
        """
        converted = self.voice_conversion(source_codes, target_speaker_embedding)
        return VCOutput(
            waveform=converted,  # In practice, need vocoder to convert mel to waveform
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
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_audio: Target audio waveform (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted audio.
        """
        converted = self.convert_audio(source_audio, target_audio)
        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )
