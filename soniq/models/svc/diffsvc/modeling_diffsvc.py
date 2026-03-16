# coding=utf-8
"""
DiffSVC: Singing Voice Conversion with Diffusion Models

DiffSVC uses diffusion models for high-quality singing voice conversion.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple
import torchaudio

from transformers.utils import logging
from soniq.models.base.outputs import VCOutput
from soniq.models.svc.base import BaseSVCModel
from soniq.models.svc.diffsvc.configuration_diffsvc import DiffSVCConfig
from soniq.models.svc.diffsvc.diffsvc_components import (
    DiffusionEmbedding,
    UNet1D,
    SpeakerEncoder,
    F0Encoder,
    get_beta_schedule,
)


logger = logging.get_logger(__name__)


class DiffSVC(BaseSVCModel):
    """
    DiffSVC: Singing Voice Conversion with Diffusion Models.

    This model uses diffusion probabilistic models for high-quality
    singing voice conversion.

    Example:
        ```python
        config = DiffSVCConfig()
        model = DiffSVC(config)

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

    config_class = DiffSVCConfig
    base_model_prefix = "diffsvc"
    supports_gradient_checkpointing = True

    def __init__(self, config: DiffSVCConfig):
        super().__init__(config)
        self.config = config

        # Register noise schedule
        betas = get_beta_schedule(
            config.beta_start,
            config.beta_end,
            config.diffusion_steps,
            config.beta_schedule,
        )
        self.register_buffer("betas", betas)

        # Precompute diffusion coefficients
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Diffusion timestep embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config.diffusion_steps,
            dim=config.hidden_dim,
        )

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
            out_dim=config.hidden_dim,  # Match hidden_dim for easier combination
        )

        # UNet for diffusion
        self.unet = UNet1D(
            in_dim=config.n_mel,
            out_dim=config.n_mel,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
            cond_dim=config.hidden_dim,  # Just use hidden_dim
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

    @property
    def num_diffusion_timesteps(self) -> int:
        """Get the number of diffusion timesteps."""
        return self.config.diffusion_steps

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

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

    def _extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        mel = self._audio_to_mel(audio)
        # Mean pooling over time
        speaker_emb = mel.mean(dim=1)  # (batch, n_mel)
        speaker_emb = self.speaker_encoder(speaker_emb)  # (batch, spk_emb_dim)
        return speaker_emb

    def _extract_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract F0 (pitch) from audio.

        In practice, this would use a proper F0 extraction algorithm
        like CREPE, RMVPE, or Harvest. For now, we use a simple approximation.
        """
        batch_size, _, seq_len = audio.shape
        device = audio.device
        # Placeholder F0
        f0 = torch.randn(batch_size, seq_len // self.hop_length, 1, device=device) * 50 + 200
        return f0

    def _q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(x_t | x_0).

        Args:
            x_0: Clean data (batch, seq_len, dim).
            t: Timestep (batch,).

        Returns:
            Noisy data x_t (batch, seq_len, dim).
        """
        alphas_cumprod = self.alphas_cumprod.to(x_0.device)
        alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        return alpha_cumprod_t.sqrt() * x_0 + (1 - alpha_cumprod_t).sqrt() * noise

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - target_audio: Target audio waveform (batch, 1, seq_len)
                - target_f0: Target F0 contour (batch, seq_len, 1)
                - source_mel: Source mel spectrogram (batch, seq_len, n_mel)

        Returns:
            Dictionary containing:
                - loss: Diffusion loss
        """
        target_audio = data["target_audio"]
        target_mel = self._audio_to_mel(target_audio)

        batch_size, seq_len, _ = target_mel.shape
        device = target_mel.device

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_timesteps, (batch_size,), device=device).long()

        # Get conditions
        if "target_f0" in data:
            target_f0 = data["target_f0"]
        else:
            target_f0 = self._extract_f0(target_audio)

        if "source_mel" in data:
            source_mel = data["source_mel"]
        else:
            source_mel = target_mel  # Reconstruction if no source

        # Ensure F0 matches mel length
        if target_f0.shape[1] != seq_len:
            if target_f0.shape[1] > seq_len:
                target_f0 = target_f0[:, :seq_len, :]
            else:
                pad_len = seq_len - target_f0.shape[1]
                target_f0 = F.pad(target_f0, (0, 0, 0, pad_len))

        speaker_emb = self._extract_speaker_embedding(target_audio)

        # Encode conditions
        f0_emb = self.f0_encoder(target_f0)  # (batch, seq_len, hidden_dim)

        # Combine conditions
        diff_emb = self.diffusion_embedding(t)  # (batch, hidden_dim)
        # Combine diff_emb, speaker_emb for global condition
        cond_global = diff_emb + speaker_emb  # (batch, hidden_dim)

        # Expand condition for sequence and add F0
        cond_expanded = cond_global.unsqueeze(1) + f0_emb  # (batch, seq_len, hidden_dim)

        # Sample noisy data
        x_t = self._q_sample(target_mel, t)

        # Predict noise
        noise_pred = self.unet(x_t, cond_expanded)

        # Compute loss
        loss = F.mse_loss(noise_pred, target_mel - x_t)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
        }

    @torch.no_grad()
    def voice_conversion(
        self,
        source_mel: torch.Tensor,
        target_f0: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convert source mel spectrogram to target singer voice using diffusion.

        Args:
            source_mel: Source mel spectrogram (batch, seq_len, n_mel).
            target_f0: Target F0 contour (batch, seq_len, 1).
            target_speaker_embedding: Target speaker embedding (batch, spk_emb_dim).
            num_steps: Number of diffusion steps (default: config.diffusion_steps).

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        batch_size, seq_len, _ = source_mel.shape
        device = source_mel.device

        num_steps = num_steps or self.num_diffusion_timesteps

        # Ensure F0 matches mel length
        if target_f0.shape[1] != seq_len:
            if target_f0.shape[1] > seq_len:
                target_f0 = target_f0[:, :seq_len, :]
            else:
                pad_len = seq_len - target_f0.shape[1]
                target_f0 = F.pad(target_f0, (0, 0, 0, pad_len))

        # Encode conditions
        f0_emb = self.f0_encoder(target_f0)

        # Start from noise
        x = torch.randn_like(source_mel)

        # Reverse diffusion
        betas = self.betas.to(device)
        alphas = 1.0 - betas
        alphas_cumprod = self.alphas_cumprod.to(device)

        for t in reversed(range(num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get condition embedding
            diff_emb = self.diffusion_embedding(t_batch)  # (batch, hidden_dim)
            # Combine with speaker embedding using addition (both are hidden_dim)
            cond = diff_emb + target_speaker_embedding  # (batch, hidden_dim)
            cond_expanded = cond.unsqueeze(1) + f0_emb  # (batch, seq_len, hidden_dim)

            # Predict
            noise_pred = self.unet(x, cond_expanded)

            # Compute x_{t-1}
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            x_0_pred = (x - noise_pred * (1 - alpha_cumprod).sqrt()) / alpha_cumprod.sqrt()
            x = (1 - beta).sqrt() * x_0_pred + beta.sqrt() * torch.randn_like(x)

        return x

    @torch.no_grad()
    def convert_audio(
        self,
        source_audio: torch.Tensor,
        target_f0: torch.Tensor,
        target_audio: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convert source audio to target singer voice.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_f0: Target F0 contour (batch, seq_len, 1).
            target_audio: Target audio waveform (batch, 1, seq_len).
            num_steps: Number of diffusion steps.

        Returns:
            Converted mel spectrogram (batch, seq_len, n_mel).
        """
        source_mel = self._audio_to_mel(source_audio)
        speaker_emb = self._extract_speaker_embedding(target_audio)
        converted = self.voice_conversion(source_mel, target_f0, speaker_emb, num_steps)
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
            target_speaker_embedding: Target speaker embedding (batch, spk_emb_dim).
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
