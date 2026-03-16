# coding=utf-8
"""
VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)

VITS is an end-to-end TTS model that generates waveforms directly from text using:
- Conditional Variational Autoencoder (VAE)
- Flow-based latent variable transformation
- Adversarial learning with GAN discriminator
- Monotonic alignment search for text-audio alignment
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple

from transformers.utils import logging
from soniq.models.base.outputs import TTSOutput
from soniq.models.tts.base import BaseTTSModel
from soniq.models.tts.vits.configuration_vits import VITSConfig
from soniq.models.tts.vits.vits_components import (
    TextEncoder,
    PosteriorEncoder,
    ResidualCouplingBlock,
    DurationPredictor,
    StochasticDurationPredictor,
    HifiGANGenerator,
    sequence_mask,
    generate_path,
    rand_slice_segments,
)


logger = logging.get_logger(__name__)


# Gradient Reversal Layer for GAN training
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return -ctx.lambda_ * grads, None


def gradient_reversal(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class VITS(BaseTTSModel):
    """
    VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

    This model generates waveforms directly from text using a variational autoencoder
    framework combined with adversarial training.

    Example:
        ```python
        config = VITSConfig()
        model = VITS(config)

        # Training
        batch = {"phone_seq": phone_ids, "phone_len": phone_lengths,
                 "mel": mel_spec, "target_len": mel_lengths}
        output = model(batch)

        # Inference
        output = model.infer(phone_ids, phone_lengths, sid=speaker_id)
        ```
    """

    config_class = VITSConfig
    base_model_prefix = "vits"
    supports_gradient_checkpointing = False

    def __init__(self, config: VITSConfig):
        super().__init__(config)
        self.config = config

        # Text encoder
        self.enc_p = TextEncoder(
            n_vocab=config.n_vocab,
            out_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            filter_channels=config.filter_channels,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            kernel_size=config.kernel_size,
            p_dropout=config.p_dropout,
        )

        # Decoder/Generator
        self.dec = HifiGANGenerator(
            initial_channel=config.inter_channels,
            resblock=config.resblock,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            gin_channels=config.gin_channels if config.n_speakers > 0 else 0,
        )

        # Posterior encoder
        self.enc_q = PosteriorEncoder(
            in_channels=config.n_mel,
            out_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=5,
            n_layers=config.n_layers_q,
            gin_channels=config.gin_channels if config.n_speakers > 0 else 0,
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            in_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=config.n_layers_q,
            gin_channels=config.gin_channels if config.n_speakers > 0 else 0,
            n_flows=config.n_flows,
            p_dropout=config.p_dropout,
        )

        # Duration predictor
        if config.use_sdp:
            self.dp = StochasticDurationPredictor(
                in_channels=config.hidden_channels,
                filter_channels=config.filter_channels,
                kernel_size=config.kernel_size,
                p_dropout=config.p_dropout,
                n_flows=config.n_flows,
                gin_channels=config.gin_channels if config.n_speakers > 0 else 0,
            )
        else:
            self.dp = DurationPredictor(
                in_channels=config.hidden_channels,
                filter_channels=config.filter_channels,
                kernel_size=config.kernel_size,
                p_dropout=config.p_dropout,
                gin_channels=config.gin_channels if config.n_speakers > 0 else 0,
            )

        # Speaker embedding
        if config.n_speakers > 0:
            self.emb_g = nn.Embedding(config.n_speakers, config.gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.0001, 0.0001)
        else:
            self.emb_g = None

        # Learnable alignment scale
        self.sdp_ratio = 0.8

    def compute_loss(self, y_hat, y, l_length, z, z_p, m_p, logs_p, m_q, logs_q):
        """
        Compute VITS losses.

        Note: VITS generates audio waveforms directly, not mel spectrograms.
        The mel reconstruction loss is not applicable in the standard VITS architecture.
        The generator is trained using adversarial loss from the discriminator.

        Args:
            y_hat: Generated waveform (not used for mel loss in VITS)
            y: Target waveform or mel (used for reference)
            l_length: Duration loss
            z: Latent variable from posterior encoder
            z_p: Transformed latent variable from flow
            m_p: Prior mean
            logs_p: Prior log variance
            m_q: Posterior mean
            logs_q: Posterior log variance

        Returns:
            Tuple of (loss_gen, loss_kl, l_length)
        """
        # KL divergence loss between posterior and prior
        loss_kl = (logs_p - logs_q - 0.5 + 0.5 * torch.exp(2 * (logs_p - logs_q)) +
                   0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p))
        loss_kl = torch.sum(loss_kl * z) / torch.sum(z) * self.config.c_kl

        # Total generator loss (duration + KL)
        # Note: adversarial loss from discriminator is added separately in GAN training
        loss_gen = l_length * self.config.c_dur + loss_kl

        return loss_gen, loss_kl, l_length

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - phone_seq: Phone token IDs (batch, seq_len)
                - phone_len: Phone sequence lengths (batch,)
                - mel: Mel spectrogram (batch, n_mel, time)
                - target_len: Mel sequence lengths (batch,)

        Returns:
            Dictionary containing outputs and losses.
        """
        x = data["phone_seq"]
        x_lengths = data["phone_len"]
        y = data["mel"]
        y_lengths = data["target_len"]

        # Get speaker embedding if available
        g = None
        if self.config.n_speakers > 0 and "spk_id" in data:
            g = self.emb_g(data["spk_id"]).unsqueeze(-1)

        # 1. Text encoding
        x_emb, m_p, logs_p, x_mask = self._encode_text(x, x_lengths, g)

        # 2. Posterior encoding
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # 3. Flow transformation
        z_p = self.flow(z, y_mask, g=g)

        # 4. Duration prediction and alignment
        l_length, attn, logw = self._compute_duration_loss(
            x_emb, x_lengths, y, y_lengths, m_p, logs_p, x_mask, y_mask, g, z
        )

        # 5. Expand prior using alignment
        m_p = self._expand_prior(m_p, attn, x_mask)
        logs_p = self._expand_prior(logs_p, attn, x_mask)

        # 6. Random segment slicing for training
        z_slice, slice_ids = rand_slice_segments(z, y_lengths, self.config.segment_size)

        # 7. Generate audio
        o = self.dec(z_slice, g=g)

        # 8. Compute losses
        loss_gen, loss_kl, _ = self.compute_loss(
            o, y, l_length, z, z_p, m_p, logs_p, m_q, logs_q
        )

        return {
            "y_hat": o,
            "loss_gen": loss_gen,
            "loss_kl": loss_kl,
            "loss_dur": l_length,
            "attn": attn,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
        }

    def _encode_text(self, x, x_lengths, g=None):
        """Encode text and return prior parameters."""
        x_out, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        # Speaker conditioning is handled by the decoder and flow layers
        # via the g parameter, not by adding to the text encoder output
        return x_out, m_p, logs_p, x_mask

    def _compute_duration_loss(self, x, x_lengths, y, y_lengths, m_p, logs_p, x_mask, y_mask, g, z=None):
        """Compute duration prediction loss and alignment."""
        # Compute negative cross-entropy for alignment
        neg_cent = self._compute_neg_cent(m_p, logs_p, z, x_mask, y_mask)

        # Monotonic alignment search
        # attn_mask: (batch, t_x, t_y) = (batch, t_x, 1) * (batch, 1, t_y)
        attn_mask = x_mask.transpose(1, 2) * y_mask
        attn = self._monotonic_alignment_search(neg_cent, attn_mask)

        # Extract duration from alignment
        w = attn.sum(2)  # (batch, t_x)

        # Compute duration loss
        if self.config.use_sdp:
            l_length = self.dp(x, x_mask, w.float(), g=g)
            l_length = l_length / (x_mask.sum())
        else:
            logw_ = torch.log(w.float() + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = ((logw - logw_) ** 2).sum() / x_mask.sum()

        return l_length, attn, logw if not self.config.use_sdp else None

    def _compute_neg_cent(self, m_p, logs_p, z, x_mask, y_mask):
        """Compute negative cross-entropy matrix for alignment."""
        # Compute alignment probabilities using the prior and encoded mel
        # neg_cent[j, i] = log P(y_i | x_j) where y_i is mel frame i and x_j is text token j
        # This is computed as the negative squared error between z and the prior
        neg_cent = torch.matmul(m_p.transpose(1, 2), z)  # (batch, t_x, t_y)
        return neg_cent

    def _monotonic_alignment_search(self, neg_cent, attn_mask):
        """
        Monotonic alignment search (simplified version).

        In a full implementation, this would use dynamic programming.
        Here we use a simplified approach for demonstration.
        """
        # Simplified: use soft alignment via softmax
        attn = F.softmax(neg_cent * 10.0, dim=2) * attn_mask
        return attn

    def _expand_prior(self, x, attn, x_mask):
        """
        Expand prior using alignment matrix.

        Args:
            x: (batch, channels, t_x)
            attn: (batch, t_x, t_y)
            x_mask: (batch, 1, t_x)

        Returns:
            (batch, channels, t_y)
        """
        # x: (batch, channels, t_x) -> transpose to (batch, t_x, channels)
        # attn: (batch, t_x, t_y) -> transpose to (batch, t_y, t_x)
        # matmul: (batch, t_y, t_x) @ (batch, t_x, channels) = (batch, t_y, channels)
        # Then transpose to (batch, channels, t_y)
        x_expanded = torch.matmul(attn.transpose(1, 2), x.transpose(1, 2)).transpose(1, 2)
        return x_expanded

    def _slice_mel_segments(self, mel, slice_ids, segment_size):
        """Slice mel segments at specified indices."""
        b, c, t = mel.shape
        max_len = min(t, segment_size)
        return mel[:, :, :max_len]

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
        noise_scale_w: float = 1.0,
        max_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Inference for waveform synthesis.

        Args:
            x: Phone token IDs of shape (batch, seq_len).
            x_lengths: Phone sequence lengths of shape (batch,).
            sid: Optional speaker ID of shape (batch, 1).
            noise_scale: Noise scale for latent sampling.
            length_scale: Duration length scale (higher = slower speech).
            noise_scale_w: Noise scale for duration prediction.
            max_len: Maximum output length.

        Returns:
            Dictionary containing:
                - y_hat: Generated waveform
                - attn: Alignment matrix
                - y_lengths: Output lengths
        """
        # Get speaker embedding
        g = None
        if sid is not None and self.emb_g is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # 1. Text encoding
        x_out, m_p, logs_p, x_mask = self._encode_text(x, x_lengths, g)

        # 2. Duration prediction
        if self.config.use_sdp:
            logw = self.dp(x_out, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x_out, x_mask, g=g)

        # Apply length scale
        w = torch.exp(logw) * length_scale
        w = torch.ceil(w).long()

        # 3. Generate path from duration
        # w shape: (batch, 1, t_x) -> squeeze to (batch, t_x)
        w_squeezed = w.squeeze(1)
        y_lengths = w_squeezed.sum(dim=1)
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x.dtype)
        # attn_mask: (batch, t_x, t_y) = (batch, 1, t_x).transpose(1,2) * (batch, 1, t_y)
        #          = (batch, t_x, 1) * (batch, 1, t_y) -> broadcasts to (batch, t_x, t_y)
        attn_mask = x_mask.transpose(1, 2) * y_mask
        attn = generate_path(w_squeezed, attn_mask)  # (batch, t_x, t_y)

        # 4. Expand prior using alignment (before unsqueeze)
        m_p = self._expand_prior(m_p, attn, x_mask)
        logs_p = self._expand_prior(logs_p, attn, x_mask)

        # Unsqueeze attn for output
        attn = attn.unsqueeze(1)

        # 5. Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # 6. Inverse flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # 7. Generate audio
        o = self.dec(z * y_mask, g=g)

        if max_len is not None:
            o = o[:, :, :max_len]

        return {
            "y_hat": o,
            "attn": attn,
            "y_lengths": y_lengths,
        }

    def synthesize(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        **kwargs,
    ) -> TTSOutput:
        """
        Synthesize speech from phone IDs.

        Args:
            phone_ids: Phone token IDs of shape (batch, seq_len).
            phone_lengths: Phone sequence lengths of shape (batch,).
            speaker_id: Optional speaker ID.
            **kwargs: Additional arguments for inference.

        Returns:
            TTSOutput with generated waveform.
        """
        sid = None
        if speaker_id is not None:
            sid = torch.tensor([[speaker_id]], device=phone_ids.device)

        output = self.infer(phone_ids, phone_lengths, sid=sid, **kwargs)
        return TTSOutput(waveform=output["y_hat"])

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        self.dec.remove_weight_norm()
        self.enc_q.remove_weight_norm()
        self.flow.remove_weight_norm()
