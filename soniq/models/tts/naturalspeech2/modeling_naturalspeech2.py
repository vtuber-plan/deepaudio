# coding=utf-8
"""
NaturalSpeech2: Latent Diffusion Neural Networks for Text-to-Speech Synthesis

NaturalSpeech2 uses:
- Prior encoder with duration and pitch predictors
- Diffusion decoder for high-quality latent synthesis
- Query-based speaker embedding from reference audio
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional

from transformers.utils import logging
from soniq.models.base.outputs import TTSOutput
from soniq.models.tts.base import BaseTTSModel
from soniq.models.tts.naturalspeech2.configuration_naturalspeech2 import NaturalSpeech2Config
from soniq.models.tts.naturalspeech2.naturalspeech2_components import (
    PriorEncoder,
    WaveNet,
    sequence_mask,
)


logger = logging.get_logger(__name__)


class NaturalSpeech2(BaseTTSModel):
    """
    NaturalSpeech2: Latent Diffusion Neural Networks for Text-to-Speech Synthesis.

    This model uses a two-stage approach:
    1. Prior encoder generates conditions from text (duration, pitch)
    2. Diffusion model synthesizes latent representations

    Example:
        ```python
        config = NaturalSpeech2Config()
        model = NaturalSpeech2(config)

        # Training
        batch = {"phone_ids": phone_ids, "durations": durations,
                 "pitch": pitch, "latents": latents}
        output = model(batch)

        # Inference
        output = model.inference(phone_ids, phone_lengths, ref_latents)
        ```
    """

    config_class = NaturalSpeech2Config
    base_model_prefix = "naturalspeech2"
    supports_gradient_checkpointing = True

    def __init__(self, config: NaturalSpeech2Config):
        super().__init__(config)
        self.config = config

        # Prior encoder
        self.prior_encoder = PriorEncoder(config)

        # Prompt encoder for speaker embedding
        prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.query_hidden,
            nhead=config.encoder_heads,
            dim_feedforward=config.query_hidden * 4,
            dropout=config.encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(prompt_encoder_layer, num_layers=2)

        # Project latent dim if needed
        if config.latent_dim != config.query_hidden:
            self.prompt_lin = nn.Linear(config.latent_dim, config.query_hidden)
        else:
            self.prompt_lin = None

        # Speaker query embeddings
        self.query_emb = nn.Embedding(config.query_token_num, config.query_hidden)

        # Query attention for speaker embedding
        self.query_attn = nn.MultiheadAttention(
            config.query_hidden,
            config.encoder_heads,
            batch_first=True,
        )

        # Diffusion model
        self.diff_estimator = WaveNet(config)

        # Diffusion parameters
        self.register_buffer("beta_min", torch.tensor(config.beta_min))
        self.register_buffer("beta_max", torch.tensor(config.beta_max))
        self.sigma = config.sigma
        self.noise_factor = config.noise_factor

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def _compute_diffusion_coefficients(self, t: torch.Tensor) -> tuple:
        """Compute diffusion coefficients for forward process."""
        # Cumulative beta
        cum_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

        # Mean and variance coefficients
        mean_coeff = torch.exp(-0.5 * cum_beta / (self.sigma ** 2))
        variance = 1 - torch.exp(-cum_beta / (self.sigma ** 2))

        return mean_coeff, variance

    def _forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Forward diffusion: add noise to x0.

        Args:
            x0: Clean latent (batch, dim, seq_len).
            t: Diffusion step (batch,).

        Returns:
            Noisy xt, mean coefficient, variance.
        """
        mean_coeff, variance = self._compute_diffusion_coefficients(t)

        # Add noise
        noise = torch.randn_like(x0) * self.noise_factor
        xt = x0 * mean_coeff[:, None, None] + noise * torch.sqrt(variance[:, None, None])

        return xt, mean_coeff, variance, noise

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - phone_ids: Phone token IDs (batch, seq_len)
                - phone_lengths: Phone sequence lengths (batch,)
                - durations: Ground truth durations (batch, seq_len)
                - pitch: Ground truth pitch (batch, output_len)
                - latents: Target latent features (batch, latent_dim, output_len)
                - ref_latents: Reference speaker latents (batch, latent_dim, ref_len)

        Returns:
            Dictionary containing predictions and losses.
        """
        phone_ids = data["phone_ids"]
        phone_lengths = data["phone_lengths"]
        latents = data["latents"]  # (batch, latent_dim, output_len)
        ref_latents = data.get("ref_latents", latents)  # Use target as ref if not provided

        # Get masks
        phone_mask = ~sequence_mask(phone_lengths, phone_ids.shape[1])
        output_len = latents.shape[2]
        output_mask = ~sequence_mask(torch.tensor([output_len] * latents.shape[0], device=latents.device))
        output_mask_2d = output_mask.unsqueeze(1)  # (batch, 1, output_len)

        # 1. Encode reference for speaker embedding
        # Create mask for ref_latents (batch, latent_dim, ref_len)
        ref_len = ref_latents.shape[2]
        ref_mask = ~sequence_mask(torch.tensor([ref_len] * ref_latents.shape[0], device=ref_latents.device))
        spk_query_emb = self._encode_speaker(ref_latents, ref_mask)

        # 2. Prior encoding
        duration = data.get("durations")
        pitch = data.get("pitch")

        dur_pred_round, dur_pred_log, dur_pred, pitch_pred_log, pitch_bucket, mel_len, prior_out = self.prior_encoder(
            phone_ids,
            duration,
            pitch,
            phone_mask,
            is_inference=False,
            ref_emb=spk_query_emb.mean(dim=1) if spk_query_emb is not None else None,
        )

        # Create output mask based on actual output length from prior encoder
        max_mel_len = mel_len.max().item()
        output_mask = ~sequence_mask(mel_len, max_mel_len)
        output_mask_2d = output_mask.unsqueeze(1)  # (batch, 1, max_mel_len)

        # Crop latents to match prior_out length
        latents = latents[:, :, :max_mel_len]

        # 3. Sample random diffusion step
        batch_size = phone_ids.shape[0]
        t = torch.rand(batch_size, device=latents.device)
        t = torch.clamp(t, 0.001, 0.999)

        # 4. Forward diffusion (latents is already (batch, latent_dim, output_len))
        xt, mean_coeff, variance, noise = self._forward_diffusion(latents, t)

        # 5. Predict x0 with WaveNet
        x0_pred = self.diff_estimator(
            xt,
            output_mask_2d,
            prior_out,
            t,
            spk_query_emb,
        )

        # 6. Compute noise prediction
        eps_pred = (xt - x0_pred * mean_coeff[:, None, None]) / torch.sqrt(variance[:, None, None] + 1e-8)

        # 7. Compute losses
        # Diffusion loss (predict x0)
        diff_loss = F.l1_loss(x0_pred, latents, reduction="none")
        diff_loss = (diff_loss * output_mask_2d).sum() / output_mask_2d.sum()

        # Prior losses
        # Duration loss
        if duration is not None:
            dur_loss = F.l1_loss(
                dur_pred_log[~phone_mask],
                torch.log(duration[~phone_mask].float() + 1),
                reduction="mean",
            )
        else:
            dur_loss = torch.tensor(0.0, device=latents.device)

        # Pitch loss
        if pitch is not None and pitch_bucket is not None:
            pitch_loss = F.l1_loss(
                pitch_pred_log[output_mask],
                torch.log(pitch[output_mask].float() + 1e-6),
                reduction="mean",
            )
        else:
            pitch_loss = torch.tensor(0.0, device=latents.device)

        # SSIM loss (optional, simplified)
        ssim_loss = torch.tensor(0.0, device=latents.device)

        total_loss = diff_loss + dur_loss * 0.1 + pitch_loss * 0.1

        return {
            "loss": total_loss,
            "diff_loss": diff_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "ssim_loss": ssim_loss,
            "x0_pred": x0_pred,
            "dur_pred": dur_pred,
            "pitch_pred": torch.exp(pitch_pred_log),
        }

    def _encode_speaker(
        self,
        ref_latents: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode reference audio to speaker embedding.

        Args:
            ref_latents: Reference latents (batch, latent_dim, ref_len).
            ref_mask: Reference mask.

        Returns:
            Speaker query embedding (batch, query_token_num, query_hidden).
        """
        # Transpose to (batch, seq_len, latent_dim)
        ref_latents = ref_latents.transpose(1, 2)

        # Project if needed
        if self.prompt_lin is not None:
            ref_latents = self.prompt_lin(ref_latents)

        # Encode with transformer
        ref_emb = self.prompt_encoder(ref_latents, src_key_padding_mask=ref_mask)

        # Cross-attention with learnable queries
        query_emb = self.query_emb.weight.unsqueeze(0).expand(ref_latents.shape[0], -1, -1)
        spk_query_emb, _ = self.query_attn(
            query_emb,
            ref_emb,
            ref_emb,
            key_padding_mask=ref_mask,
        )

        return spk_query_emb

    @torch.no_grad()
    def inference(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: torch.Tensor,
        ref_latents: Optional[torch.Tensor] = None,
        inference_steps: int = 1000,
        temperature: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Inference for speech synthesis.

        Args:
            phone_ids: Phone token IDs (batch, seq_len).
            phone_lengths: Phone sequence lengths (batch,).
            ref_latents: Optional reference latents for speaker (batch, latent_dim, ref_len).
            inference_steps: Number of reverse diffusion steps.
            temperature: Sampling temperature.

        Returns:
            Dictionary containing generated latents.
        """
        batch_size = phone_ids.shape[0]
        device = phone_ids.device

        # 1. Encode speaker from reference
        if ref_latents is None:
            # Use a dummy reference
            ref_latents = torch.randn(batch_size, self.config.latent_dim, 50, device=device)

        ref_mask = torch.zeros(batch_size, ref_latents.shape[2], dtype=torch.bool, device=device)
        spk_query_emb = self._encode_speaker(ref_latents, ref_mask)

        # 2. Prior encoding (inference mode)
        phone_mask = ~sequence_mask(phone_lengths, phone_ids.shape[1])
        dur_pred_round, dur_pred_log, dur_pred, pitch_pred_log, pitch_bucket, mel_len, prior_out = self.prior_encoder(
            phone_ids,
            duration=None,
            pitch=None,
            phone_mask=phone_mask,
            is_inference=True,
            ref_emb=spk_query_emb.mean(dim=1),
        )

        # Get output length from predicted duration
        output_len = mel_len.max().item()
        output_mask = ~sequence_mask(torch.tensor([output_len] * batch_size, device=device))
        output_mask_2d = output_mask.unsqueeze(1)

        # 3. Sample initial noise
        z = torch.randn(batch_size, output_len, self.config.latent_dim, device=device) / temperature

        # 4. Reverse diffusion
        step_size = 1.0 / inference_steps
        for i in range(inference_steps):
            t = torch.ones(batch_size, device=device) * (1.0 - i * step_size)
            t = torch.clamp(t, 0.001, 0.999)

            # Compute coefficients
            mean_coeff, variance = self._compute_diffusion_coefficients(t)

            # Predict x0
            x0_pred = self.diff_estimator(
                z.transpose(1, 2),
                output_mask_2d,
                prior_out,
                t,
                spk_query_emb,
            )

            # Compute noise
            eps_pred = (z - x0_pred.transpose(1, 2) * mean_coeff[:, None, None]) / torch.sqrt(variance[:, None, None] + 1e-8)

            # Update z (Euler solver)
            cum_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)
            h = step_size

            dz = -0.5 * h * (self.beta_min + (self.beta_max - self.beta_min) * t)[:, None, None] * (
                x0_pred.transpose(1, 2) / (self.sigma ** 2) + eps_pred / (variance[:, None, None] + 1e-8)
            )

            z = z - dz

        # Final output
        generated_latents = z.transpose(1, 2) * output_mask_2d

        return {
            "latents": generated_latents,
            "durations": dur_pred,
            "pitch": torch.exp(pitch_pred_log),
        }

    def synthesize(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        ref_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TTSOutput:
        """
        Synthesize speech from phone IDs.

        Args:
            phone_ids: Phone token IDs (batch, seq_len).
            phone_lengths: Phone sequence lengths (batch,).
            speaker_id: Optional speaker ID (not used directly).
            ref_latents: Optional reference latents for zero-shot synthesis.
            **kwargs: Additional arguments for inference.

        Returns:
            TTSOutput with generated latents.
        """
        output = self.inference(
            phone_ids, phone_lengths, ref_latents=ref_latents, **kwargs
        )
        return TTSOutput(latents=output["latents"])
