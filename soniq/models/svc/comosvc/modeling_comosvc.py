# coding=utf-8
"""
ComoSVC: Singing Voice Conversion with Content-Rich Features

ComoSVC uses content-rich acoustic features and adaptive layer normalization
for high-quality singing voice conversion.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple
import torchaudio

from transformers.utils import logging
from soniq.models.base.outputs import VCOutput
from soniq.models.svc.base import BaseSVCModel
from soniq.models.svc.comosvc.configuration_comosvc import ComoSVCConfig
from soniq.models.svc.comosvc.comosvc_components import (
    F0Encoder,
    ContentEncoder,
    Decoder,
    SpeakerEncoder,
)


logger = logging.get_logger(__name__)


class ComoSVC(BaseSVCModel):
    """
    ComoSVC: Singing Voice Conversion with Content-Rich Features.

    This model converts singing voice from one singer to another while
    preserving the linguistic content and musical expression.

    Example:
        ```python
        config = ComoSVCConfig()
        model = ComoSVC(config)

        # Training
        batch = {
            "source_audio": source_audio,
            "target_audio": target_audio,
            "source_f0": source_f0,
            "target_f0": target_f0,
        }
        output = model(batch)

        # Inference
        converted = model.voice_conversion(source_mel, target_f0, target_speaker_emb)
        ```
    """

    config_class = ComoSVCConfig
    base_model_prefix = "comosvc"
    supports_gradient_checkpointing = True

    def __init__(self, config: ComoSVCConfig):
        super().__init__(config)
        self.config = config

        # F0 encoder
        self.f0_encoder = F0Encoder(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
        )

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            in_dim=config.n_mel,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
        )

        # Content encoder
        self.content_encoder = ContentEncoder(
            in_dim=config.n_mel,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            cond_dim=config.hidden_dim,
        )

        # Decoder
        self.decoder = Decoder(
            in_dim=config.hidden_dim * 2,  # content + f0
            out_dim=config.n_mel,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    @property
    def hop_length(self) -> int:
        """Get the hop length."""
        return self.config.hop_length

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def _extract_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract F0 (pitch) from audio.

        In practice, this would use a proper F0 extraction algorithm
        like CREPE, RMVPE, or Harvest. For now, we use a simple approximation.
        """
        # Placeholder: In real implementation, use CREPE/RMVPE
        # This is just for testing - replace with actual F0 extraction
        batch_size, _, seq_len = audio.shape
        device = audio.device

        # Simulate F0 with random values (placeholder)
        # Real implementation would use: f0 = crepe.predict(audio, sr=self.sample_rate)
        f0 = torch.randn(batch_size, seq_len // self.hop_length, 1, device=device) * 50 + 200
        return f0

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
            hop_length=self.hop_length,
            win_length=self.hop_length * 4,
            n_fft=self.n_fft,
        )
        mel = mel_fn(audio.squeeze(1))
        mel = (mel + 1e-9).log().transpose(1, 2)  # (batch, time, n_mel)
        return mel

    @property
    def n_fft(self) -> int:
        """Get the FFT size."""
        return self.config.n_fft

    def _extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        mel = self._audio_to_mel(audio)
        # Mean pooling over time
        speaker_emb = mel.mean(dim=1)  # (batch, n_mel)
        speaker_emb = self.speaker_encoder(speaker_emb)  # (batch, hidden_dim)
        return speaker_emb

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - source_audio: Source audio waveform (batch, 1, seq_len)
                - target_audio: Target audio waveform (batch, 1, seq_len)
                - source_f0: Source F0 contour (batch, seq_len, 1)
                - target_f0: Target F0 contour (batch, seq_len, 1)

        Returns:
            Dictionary containing:
                - mel_pred: Predicted mel spectrogram
                - mel_target: Target mel spectrogram
                - loss: Total loss
        """
        source_audio = data["source_audio"]
        target_audio = data["target_audio"]

        # Extract or use provided F0
        if "source_f0" in data:
            source_f0 = data["source_f0"]
        else:
            source_f0 = self._extract_f0(source_audio)

        if "target_f0" in data:
            target_f0 = data["target_f0"]
        else:
            target_f0 = self._extract_f0(target_audio)

        # Convert audio to mel
        source_mel = self._audio_to_mel(source_audio)
        target_mel = self._audio_to_mel(target_audio)

        batch_size, mel_seq_len, _ = source_mel.shape

        # Create mask
        mask = torch.ones(batch_size, mel_seq_len, device=source_mel.device, dtype=torch.bool)

        # Extract speaker embedding from target
        speaker_emb = self._extract_speaker_embedding(target_audio)  # (batch, hidden_dim)

        # Encode F0 - crop or pad to match mel length
        if target_f0.shape[1] != mel_seq_len:
            if target_f0.shape[1] > mel_seq_len:
                target_f0 = target_f0[:, :mel_seq_len, :]
            else:
                pad_len = mel_seq_len - target_f0.shape[1]
                target_f0 = F.pad(target_f0, (0, 0, 0, pad_len))

        # Encode F0
        f0_emb = self.f0_encoder(target_f0)  # (batch, seq_len, hidden_dim)

        # Encode content
        content_out = self.content_encoder(source_mel, speaker_emb, mask)  # (batch, seq_len, hidden_dim)

        # Combine content and F0
        combined = torch.cat([content_out, f0_emb], dim=-1)

        # Decode
        mel_pred = self.decoder(combined)  # (batch, seq_len, n_mel)

        # Crop target to match prediction length
        max_len = mel_pred.shape[1]
        if target_mel.shape[1] > max_len:
            target_mel = target_mel[:, :max_len, :]
        elif target_mel.shape[1] < max_len:
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
        source_mel: torch.Tensor,
        target_f0: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source mel spectrogram to target singer voice.

        Args:
            source_mel: Source mel spectrogram (batch, seq_len, n_mel).
            target_f0: Target F0 contour (batch, seq_len, 1).
            target_speaker_embedding: Target speaker embedding (batch, hidden_dim).

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        batch_size, mel_seq_len, _ = source_mel.shape

        # Ensure F0 matches mel length
        if target_f0.shape[1] != mel_seq_len:
            if target_f0.shape[1] > mel_seq_len:
                target_f0 = target_f0[:, :mel_seq_len, :]
            else:
                pad_len = mel_seq_len - target_f0.shape[1]
                target_f0 = F.pad(target_f0, (0, 0, 0, pad_len))

        # Encode F0
        f0_emb = self.f0_encoder(target_f0)

        # Encode content
        mask = torch.ones(batch_size, mel_seq_len, device=source_mel.device, dtype=torch.bool)
        content_out = self.content_encoder(source_mel, target_speaker_embedding, mask)

        # Combine content and F0
        combined = torch.cat([content_out, f0_emb], dim=-1)

        # Decode
        mel_pred = self.decoder(combined)

        return mel_pred

    @torch.no_grad()
    def convert_audio(
        self,
        source_audio: torch.Tensor,
        target_f0: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source audio to target singer voice.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_f0: Target F0 contour (batch, seq_len, 1).
            target_audio: Target audio waveform (batch, 1, seq_len).

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        # Extract speaker embedding from target
        speaker_emb = self._extract_speaker_embedding(target_audio)

        # Convert source to mel
        source_mel = self._audio_to_mel(source_audio)

        # Convert
        converted = self.voice_conversion(source_mel, target_f0, speaker_emb)

        return converted

    def synthesize(
        self,
        source_mel: torch.Tensor,
        target_f0: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Synthesize converted singing voice.

        Args:
            source_mel: Source mel spectrogram (batch, seq_len, n_mel).
            target_f0: Target F0 contour (batch, seq_len, 1).
            target_speaker_embedding: Target speaker embedding (batch, hidden_dim).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted features.
        """
        converted = self.voice_conversion(source_mel, target_f0, target_speaker_embedding)
        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )

    def reconstruct(
        self,
        source_audio: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Reconstruct audio.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with reconstructed audio.
        """
        source_mel = self._audio_to_mel(source_audio)
        f0 = self._extract_f0(source_audio)
        speaker_emb = self._extract_speaker_embedding(source_audio)

        converted = self.voice_conversion(source_mel, f0, speaker_emb)

        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )
