# coding=utf-8
"""
VitsSVC: Singing Voice Conversion with VAE + Flow

VitsSVC uses a VAE-Flow architecture similar to VITS but with
content features (from content encoders) instead of text input.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional
import torchaudio

from transformers.utils import logging
from soniq.models.base.outputs import VCOutput
from soniq.models.svc.base import BaseSVCModel
from soniq.models.svc.vitsvc.configuration_vitsvc import VitsSVCConfig
from soniq.models.svc.vitsvc.vitsvc_components import (
    ContentEncoder,
    PosteriorEncoder,
    PriorEncoder,
    ResidualCouplingFlow,
    Generator,
    SpeakerEncoder,
)


logger = logging.get_logger(__name__)


class VitsSVC(BaseSVCModel):
    """
    VitsSVC: Singing Voice Conversion with VAE + Flow.

    This model converts singing voice using content features extracted
    from pretrained content encoders (like ContentVec, Hubert, etc.).

    Example:
        ```python
        config = VitsSVCConfig()
        model = VitsSVC(config)

        # Training
        batch = {
            "source_audio": source_audio,
            "target_audio": target_audio,
            "content_features": content_features,
        }
        output = model(batch)

        # Inference
        converted = model.voice_conversion(content_features, target_speaker_emb)
        ```
    """

    config_class = VitsSVCConfig
    base_model_prefix = "vitsvc"
    supports_gradient_checkpointing = True

    def __init__(self, config: VitsSVCConfig):
        super().__init__(config)
        self.config = config

        # Content encoder (transforms content features to hidden representation)
        self.content_encoder = ContentEncoder(
            in_channels=config.n_mel,  # Use n_mel as default content feature dim
            hidden_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        # Posterior encoder (VAE encoder)
        self.posterior_encoder = PosteriorEncoder(
            in_channels=config.n_mel,
            out_channels=config.hidden_dim,
            hidden_channels=config.inter_channels,
            kernel_size=config.kernel_size,
            n_layers=config.n_flow_layers,
            dropout=config.dropout,
        )

        # Prior encoder (VAE prior)
        self.prior_encoder = PriorEncoder(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            hidden_channels=config.inter_channels,
            kernel_size=config.kernel_size,
            n_layers=config.n_prior_layers,
            dropout=config.dropout,
        )

        # Flow-based decoder
        self.decoder_flow = ResidualCouplingFlow(
            channels=config.hidden_dim,
            hidden_channels=config.inter_channels,
            kernel_size=config.kernel_size,
            n_layers=config.n_flow_layers,
            dropout=config.dropout,
        )

        # Generator (waveform synthesis)
        self.generator = Generator(
            in_channels=config.hidden_dim,
            out_channels=1,
            hidden_channels=config.hidden_dim,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
        )

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            in_channels=config.n_mel,
            hidden_channels=config.inter_channels,
            out_channels=config.gin_channels,
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

    @property
    def n_fft(self) -> int:
        """Get the FFT size."""
        return self.config.n_fft

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

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
        mel = (mel + 1e-9).log()
        return mel  # (batch, n_mel, time)

    def _extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        mel = self._audio_to_mel(audio)
        speaker_emb = self.speaker_encoder(mel)
        return speaker_emb

    def _extract_content_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from audio.

        In practice, this would use a pretrained content encoder
        like ContentVec, Hubert, or Whisper. For now, we use mel
        as a placeholder.
        """
        mel = self._audio_to_mel(audio)
        # Placeholder: use mel as content features
        # Real implementation: content_features = content_encoder(audio)
        return mel

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - source_audio: Source audio waveform (batch, 1, seq_len)
                - target_audio: Target audio waveform (batch, 1, seq_len)
                - content_features: Content features (batch, hidden_dim, seq_len)

        Returns:
            Dictionary containing:
                - audio_pred: Predicted audio waveform
                - loss: Total loss (reconstruction + KL)
        """
        source_audio = data["source_audio"]
        target_audio = data["target_audio"]

        # Extract or use provided content features
        if "content_features" in data:
            content_features = data["content_features"]
        else:
            content_features = self._extract_content_features(source_audio)

        # Get target mel
        target_mel = self._audio_to_mel(target_audio)  # (batch, n_mel, time)

        # Encode content
        content_emb = self.content_encoder(content_features)  # (batch, hidden_dim, time)

        # Posterior encoding (VAE)
        z_posterior, m_q, logs_q = self.posterior_encoder(target_mel)

        # Prior encoding
        z_prior, m_p = self.prior_encoder(content_emb)

        # Flow-based decoding
        z_decoded = self.decoder_flow(z_posterior)

        # Generate waveform
        audio_pred = self.generator(z_decoded)

        # Crop target audio to match prediction length
        if audio_pred.shape[2] < target_audio.shape[2]:
            audio_target = target_audio[:, :, :audio_pred.shape[2]]
        elif audio_pred.shape[2] > target_audio.shape[2]:
            audio_target = F.pad(target_audio, (0, audio_pred.shape[2] - target_audio.shape[2]))
        else:
            audio_target = target_audio

        # Compute losses
        # Reconstruction loss
        recon_loss = F.l1_loss(audio_pred, audio_target)

        # KL divergence loss
        kl_loss = torch.mean(0.5 * (m_q ** 2 + logs_q.exp() - logs_q - 1))

        # Total loss
        loss = recon_loss + kl_loss

        return {
            "audio_pred": audio_pred,
            "audio_target": target_audio,
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def voice_conversion(
        self,
        content_features: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert content features to singing voice.

        Args:
            content_features: Content features (batch, hidden_dim, seq_len).
            target_speaker_embedding: Target speaker embedding (batch, gin_channels).

        Returns:
            Converted audio waveform (batch, 1, seq_len * hop_factor).
        """
        # Encode content
        content_emb = self.content_encoder(content_features)

        # Sample from prior
        z, _ = self.prior_encoder(content_emb)

        # Flow-based decoding
        z_decoded = self.decoder_flow(z, reverse=True)

        # Generate waveform
        audio_pred = self.generator(z_decoded)

        return audio_pred

    @torch.no_grad()
    def convert_audio(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert source audio to target singer voice.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_audio: Target audio waveform (batch, 1, seq_len).

        Returns:
            Converted audio waveform (batch, 1, seq_len * hop_factor).
        """
        # Extract content features from source
        content_features = self._extract_content_features(source_audio)

        # Extract speaker embedding from target
        speaker_emb = self._extract_speaker_embedding(target_audio)

        # Convert
        audio_pred = self.voice_conversion(content_features, speaker_emb)

        return audio_pred

    def synthesize(
        self,
        content_features: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        **kwargs,
    ) -> VCOutput:
        """
        Synthesize converted singing voice.

        Args:
            content_features: Content features (batch, hidden_dim, seq_len).
            target_speaker_embedding: Target speaker embedding (batch, gin_channels).
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted audio.
        """
        audio_pred = self.voice_conversion(content_features, target_speaker_embedding)
        return VCOutput(
            waveform=audio_pred,
            converted_features=audio_pred,
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
        content_features = self._extract_content_features(source_audio)
        speaker_emb = self._extract_speaker_embedding(source_audio)
        audio_pred = self.voice_conversion(content_features, speaker_emb)
        return VCOutput(
            waveform=audio_pred,
            converted_features=audio_pred,
        )
