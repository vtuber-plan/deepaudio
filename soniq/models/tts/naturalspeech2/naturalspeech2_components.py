# coding=utf-8
"""
NaturalSpeech2 components.

This module contains the core components for NaturalSpeech2:
- Prior Encoder
- WaveNet for Diffusion
- Diffusion/Flow modules
- Duration and Pitch Predictors
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple


# ============================================================================
# Utility Functions
# ============================================================================

def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Create sequence mask."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# ============================================================================
# Duration and Pitch Predictors
# ============================================================================

class DurationPredictor(nn.Module):
    """
    Duration predictor for NaturalSpeech2.

    Predicts log duration for each input frame.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = []
        in_ch = in_channels
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = hidden_channels

        layers.append(nn.Conv1d(hidden_channels, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, channels).
            mask: Optional mask.

        Returns:
            Duration predictions (batch, seq_len).
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        log_dur = self.layers(x).transpose(1, 2)  # (batch, seq_len, 1)
        if mask is not None:
            log_dur = log_dur * mask.unsqueeze(-1)
        return log_dur.squeeze(-1)


class PitchPredictor(nn.Module):
    """
    Pitch predictor for NaturalSpeech2.

    Predicts log pitch for each input frame.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = []
        in_ch = in_channels
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = hidden_channels

        layers.append(nn.Conv1d(hidden_channels, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, channels).
            mask: Optional mask.

        Returns:
            Pitch predictions (batch, seq_len).
        """
        x = x.transpose(1, 2)
        log_pitch = self.layers(x).transpose(1, 2)
        if mask is not None:
            log_pitch = log_pitch * mask.unsqueeze(-1)
        return log_pitch.squeeze(-1)


class LengthRegulator(nn.Module):
    """
    Length regulator expands input by duration.
    """

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, channels).
            duration: Duration tensor (batch, seq_len).
            max_len: Maximum output length.

        Returns:
            Expanded tensor and output lengths.
        """
        batch_size, seq_len, channels = x.shape

        # Expand by duration
        expanded = []
        lengths = []
        for b in range(batch_size):
            frames = []
            for t in range(seq_len):
                d = duration[b, t].long()
                # Ensure at least 1 frame per phone (skip if duration is 0)
                if d > 0:
                    frames.extend([x[b, t]] * d)
            # Handle case where all durations are 0
            if len(frames) == 0:
                frames = [x[b, 0]]  # Use first frame as fallback
            expanded.append(torch.stack(frames))
            lengths.append(len(frames))

        # Pad to max length
        max_out_len = max(lengths) if max_len is None else min(max(lengths), max_len)
        output = x.new_zeros(batch_size, max_out_len, channels)
        out_mask = torch.zeros(batch_size, max_out_len, dtype=torch.bool, device=x.device)

        for b in range(batch_size):
            out_len = min(lengths[b], max_out_len)
            output[b, :out_len] = expanded[b][:out_len]
            out_mask[b, :out_len] = True

        return output, out_mask


# ============================================================================
# Prior Encoder
# ============================================================================

class PriorEncoder(nn.Module):
    """
    Prior encoder for NaturalSpeech2.

    Encodes text and generates conditions for diffusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text embedding
        self.enc_emb_tokens = nn.Embedding(config.vocab_size, config.encoder_hidden)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_hidden * 4,
            dropout=config.encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        # Predictors
        self.duration_predictor = DurationPredictor(
            config.encoder_hidden,
            config.duration_predictor_hidden,
            config.duration_predictor_layers,
        )
        self.pitch_predictor = PitchPredictor(
            config.encoder_hidden,
            config.pitch_predictor_hidden,
            config.pitch_predictor_layers,
        )

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Pitch embedding
        self.pitch_embedding = nn.Embedding(config.pitch_bins_num, config.encoder_hidden)

        # Register pitch bins
        pitch_bins = torch.linspace(config.pitch_min, config.pitch_max, config.pitch_bins_num - 1)
        self.register_buffer("pitch_bins", pitch_bins)

    def forward(
        self,
        phone_id: torch.Tensor,
        duration: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        phone_mask: Optional[torch.Tensor] = None,
        is_inference: bool = False,
        ref_emb: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Args:
            phone_id: Phone IDs (batch, seq_len).
            duration: Ground truth duration (batch, seq_len).
            pitch: Ground truth pitch (batch, output_len) - already expanded.
            phone_mask: Phone mask.
            is_inference: If True, use predicted duration.
            ref_emb: Optional reference embedding for speaker conditioning.

        Returns:
            Duration predictions, pitch predictions, mel lengths, prior output.
        """
        # 1. Text encoding
        x = self.enc_emb_tokens(phone_id)

        # Add speaker conditioning if available
        if ref_emb is not None:
            x = x + ref_emb.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=phone_mask)

        # 2. Duration prediction
        dur_pred_log = self.duration_predictor(x, ~phone_mask if phone_mask is not None else None)
        dur_pred = torch.exp(dur_pred_log) - 1

        if is_inference:
            dur_pred_round = torch.clamp(dur_pred.round(), min=1).long()
        else:
            dur_pred_round = duration

        # 3. Length regulation
        prior_out, mel_mask = self.length_regulator(x, dur_pred_round)
        output_len = prior_out.shape[1]

        # 4. Pitch prediction
        pitch_pred_log = self.pitch_predictor(prior_out, mel_mask)

        # 5. Add pitch embedding if pitch is available
        if pitch is not None:
            # Squeeze last dimension if present (pitch can be (batch, seq) or (batch, seq, 1))
            if pitch.dim() == 3:
                pitch = pitch.squeeze(-1)

            # Trim or pad pitch to match output length
            if pitch.shape[1] != output_len:
                if pitch.shape[1] > output_len:
                    pitch = pitch[:, :output_len]
                else:
                    # Pad with mean pitch
                    mean_pitch = pitch.mean(dim=1, keepdim=True).expand(-1, output_len - pitch.shape[1])
                    pitch = torch.cat([pitch, mean_pitch], dim=1)

            # Bucketize pitch
            pitch_bucket = torch.bucketize(pitch, self.pitch_bins)
            pitch_embed = self.pitch_embedding(pitch_bucket)
            prior_out = prior_out + pitch_embed

        mel_len = mel_mask.sum(dim=1)

        return dur_pred_round, dur_pred_log, dur_pred, pitch_pred_log, pitch_bucket if pitch is not None else None, mel_len, prior_out


# ============================================================================
# WaveNet for Diffusion
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block for WaveNet with dilated convolution.
    """

    def __init__(
        self,
        hidden_dim: int,
        dilation: int,
        condition_dim: int = 0,
        diffusion_step_dim: int = 256,
        has_cattn: bool = False,
        cattn_dim: int = 512,
        nhead: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.has_cattn = has_cattn

        # Dilated convolution
        self.dilated_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        # Diffusion step projection
        self.diffusion_proj = nn.Sequential(
            nn.Linear(diffusion_step_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # Condition projection (to hidden_dim * 2 to match dilated_conv output)
        self.cond_proj = nn.Conv1d(condition_dim, hidden_dim * 2, 1) if condition_dim > 0 else None

        # Output projection
        self.out_proj = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)

        # Cross-attention (optional)
        if has_cattn:
            self.attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
            self.film = nn.Linear(hidden_dim, hidden_dim * 2)
            self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        condition: torch.Tensor,
        diffusion_step_emb: torch.Tensor,
        spk_query_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch, hidden_dim, seq_len).
            x_mask: Mask (batch, 1, seq_len).
            condition: Condition (batch, condition_dim, seq_len).
            diffusion_step_emb: Diffusion step embedding (batch, diffusion_step_dim).
            spk_query_emb: Speaker query embedding (batch, query_len, query_dim).

        Returns:
            Output and skip connection.
        """
        # Diffusion embedding
        diff_emb = self.diffusion_proj(diffusion_step_emb).unsqueeze(-1)

        # Dilated conv
        h = self.dilated_conv(x)

        # Add diffusion and condition
        h = h + diff_emb
        if self.cond_proj is not None:
            h = h + self.cond_proj(condition)

        # Gated activation
        h = F.glu(h, dim=1)

        # Cross-attention (optional)
        if self.has_cattn and spk_query_emb is not None:
            h_trans = h.transpose(1, 2)  # (batch, seq_len, hidden)
            attn_out, _ = self.attn(h_trans, spk_query_emb, spk_query_emb, is_causal=False)
            film_out = self.film(attn_out)
            gate, val = film_out.chunk(2, dim=-1)
            h_trans = self.ln(h_trans)
            h_trans = h_trans * torch.sigmoid(gate) * torch.sigmoid(val)
            h = h_trans.transpose(1, 2)

        # Output projection
        h = self.out_proj(h)

        # Gated activation
        h = F.glu(h, dim=1)

        # Residual connection
        x = (x + h) * x_mask
        skip = x

        return x, skip


class WaveNet(nn.Module):
    """
    WaveNet for diffusion denoising.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_dim = config.diffusion_hidden
        condition_dim = config.encoder_hidden
        speaker_dim = config.query_hidden

        # Input projection
        self.in_proj = nn.Conv1d(config.latent_dim, hidden_dim, 1)

        # Diffusion step embedding
        self.diffusion_embedding = SinusoidalPosEmb(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Condition layer norm
        self.cond_ln = nn.LayerNorm(condition_dim)

        # Residual blocks
        self.layers = nn.ModuleList()
        for i in range(config.diffusion_layers):
            dilation = 2 ** (i % config.dilation_cycle)
            has_cattn = (i % config.cross_attn_per_layer == 0) and (i > 0)
            self.layers.append(
                ResidualBlock(
                    hidden_dim,
                    dilation,
                    condition_dim,
                    hidden_dim,
                    has_cattn=has_cattn,
                    cattn_dim=speaker_dim,
                    nhead=8,
                )
            )

        # Skip projection
        self.skip_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.out_proj = nn.Conv1d(hidden_dim, config.latent_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        condition: torch.Tensor,
        diffusion_step: torch.Tensor,
        spk_query_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, latent_dim, seq_len).
            x_mask: Mask (batch, 1, seq_len).
            condition: Condition (batch, seq_len, condition_dim).
            diffusion_step: Diffusion step (batch,).
            spk_query_emb: Speaker query embedding (batch, query_len, query_dim).

        Returns:
            Denoised output (batch, latent_dim, seq_len).
        """
        # LayerNorm on condition
        condition = self.cond_ln(condition).transpose(1, 2)

        # Project input
        x = self.in_proj(x)

        # Diffusion embedding
        diff_emb = self.diffusion_embedding(diffusion_step)
        diff_emb = self.mlp(diff_emb)

        # Residual blocks
        skip = 0
        for layer in self.layers:
            x, layer_skip = layer(x, x_mask, condition, diff_emb, spk_query_emb)
            skip = skip + layer_skip

        # Sum skip connections
        skip = self.skip_proj(skip)
        out = self.out_proj(F.relu(skip))

        return out * x_mask
