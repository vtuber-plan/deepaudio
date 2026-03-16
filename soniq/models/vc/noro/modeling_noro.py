# coding=utf-8
"""
Noro: Voice Conversion with Neural Audio Codecs

Noro uses:
- Residual vector quantization for discrete representation learning
- Transformer-based encoder and decoder
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
from soniq.models.vc.noro.configuration_noro import NoroConfig
from soniq.models.vc.noro.noro_components import (
    CodecEncoder,
    SpeakerEncoder,
    Decoder,
)


logger = logging.get_logger(__name__)


class Noro(BaseVCModel):
    """
    Noro: Voice Conversion with Neural Audio Codecs.

    This model converts speech from one speaker to another using discrete
    representations learned through residual vector quantization.

    Example:
        ```python
        config = NoroConfig()
        model = Noro(config)

        # Training
        batch = {
            "source_audio": source_audio,
            "target_audio": target_audio,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
        }
        output = model(batch)

        # Inference
        converted = model.voice_conversion(source_mel, target_speaker_embedding)
        ```
    """

    config_class = NoroConfig
    base_model_prefix = "noro"
    supports_gradient_checkpointing = True

    def __init__(self, config: NoroConfig):
        super().__init__(config)
        self.config = config

        # Codec encoder
        self.encoder = CodecEncoder(
            dim=config.hidden_dim,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            in_dim=config.n_mel,
            hidden_dim=config.hidden_dim,
            speaker_dim=config.speaker_dim,
        )

        # Decoder
        self.decoder = Decoder(
            in_dim=config.codebook_dim + config.speaker_dim,
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

    def _extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        # Compute mel spectrogram
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
            hop_length=self.hop_length,
            win_length=self.hop_length * 4,
            n_fft=self.hop_length * 4,
        )
        mel = mel_fn(audio.squeeze(1))  # (batch, n_mel, time)
        mel = (mel + 1e-9).log()

        # Mean pooling over time
        speaker_emb = mel.mean(dim=2)  # (batch, n_mel)
        speaker_emb = self.speaker_encoder(speaker_emb)  # (batch, speaker_dim)
        return speaker_emb

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mel,
            hop_length=self.hop_length,
            win_length=self.hop_length * 4,
            n_fft=self.hop_length * 4,
        )
        mel = mel_fn(audio.squeeze(1))
        mel = (mel + 1e-9).log().transpose(1, 2)  # (batch, time, n_mel)
        return mel

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - source_audio: Source audio waveform (batch, 1, seq_len)
                - target_audio: Target audio waveform (batch, 1, seq_len)
                - source_lengths: Source audio lengths (batch,)
                - target_lengths: Target audio lengths (batch,)

        Returns:
            Dictionary containing:
                - mel_pred: Predicted mel spectrogram
                - mel_target: Target mel spectrogram
                - codes: Encoded codes
                - loss: Total loss
        """
        source_audio = data["source_audio"]
        target_audio = data["target_audio"]
        source_lengths = data.get("source_lengths", None)
        target_lengths = data.get("target_lengths", None)

        # Convert audio to mel
        source_mel = self._audio_to_mel(source_audio)
        target_mel = self._audio_to_mel(target_audio)

        batch_size, seq_len, _ = source_mel.shape

        # Create mask
        if source_lengths is not None:
            mel_seq_len = source_mel.shape[1]
            mask = torch.arange(mel_seq_len, device=source_mel.device).unsqueeze(0) < source_lengths.unsqueeze(1)
        else:
            mask = torch.ones_like(source_mel[:, :, 0], dtype=torch.bool)

        # Extract speaker embedding from target
        speaker_emb = self._extract_speaker_embedding(target_audio)  # (batch, speaker_dim)

        # Encode source
        codes, quantized = self.encoder(source_mel, mask)  # codes: (batch, seq_len, n_codebooks)

        # Add speaker embedding
        speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([quantized, speaker_emb_expanded], dim=-1)

        # Decode
        mel_pred = self.decoder(combined, mask)

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
            "codes": codes,
            "loss": mel_loss,
        }

    @torch.no_grad()
    def voice_conversion(
        self,
        source_mel: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source mel spectrogram to target speaker voice.

        Args:
            source_mel: Source mel spectrogram (batch, seq_len, n_mel).
            target_speaker_embedding: Target speaker embedding (batch, speaker_dim).

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        batch_size, seq_len, _ = source_mel.shape

        # Encode source
        codes, quantized = self.encoder(source_mel)

        # Add speaker embedding
        speaker_emb_expanded = target_speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([quantized, speaker_emb_expanded], dim=-1)

        # Decode
        mel_pred = self.decoder(combined)

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
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        # Extract speaker embedding from target
        speaker_emb = self._extract_speaker_embedding(target_audio)

        # Convert source to mel
        source_mel = self._audio_to_mel(source_audio)

        # Convert
        converted = self.voice_conversion(source_mel, speaker_emb)

        return converted

    def synthesize(
        self,
        source_codes: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Synthesize converted speech from codes.

        Args:
            source_codes: Source codes (batch, seq_len, n_codebooks).
            target_speaker_embedding: Target speaker embedding (batch, speaker_dim).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted features.
        """
        # Get quantized features from codes
        quantized = self.encoder.quantizer.get_codes_from_indices(source_codes)

        batch_size, seq_len, _ = quantized.shape

        # Add speaker embedding
        speaker_emb_expanded = target_speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([quantized, speaker_emb_expanded], dim=-1)

        # Decode
        mel_pred = self.decoder(combined)

        return VCOutput(
            waveform=mel_pred,
            converted_features=mel_pred,
        )

    def reconstruct(
        self,
        source_audio: torch.Tensor,
        target_audio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VCOutput:
        """
        Reconstruct audio.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_audio: Optional target audio for speaker (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with reconstructed audio.
        """
        source_mel = self._audio_to_mel(source_audio)

        if target_audio is not None:
            speaker_emb = self._extract_speaker_embedding(target_audio)
            converted = self.voice_conversion(source_mel, speaker_emb)
        else:
            # Self-reconstruction
            codes, quantized = self.encoder(source_mel)
            speaker_emb = self._extract_speaker_embedding(source_audio)
            speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, source_mel.shape[1], -1)
            combined = torch.cat([quantized, speaker_emb_expanded], dim=-1)
            converted = self.decoder(combined)

        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )
