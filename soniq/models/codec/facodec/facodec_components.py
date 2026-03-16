# coding=utf-8
"""
FACodec components.

This module contains the core components for FACodec:
- FAQuantizer (factorized quantizer)
- Style Encoder
- WaveNet encoder
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple
import math


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
# Activation Functions
# ============================================================================

class SnakeBeta(nn.Module):
    """
    Modified Snake function with separate parameters for magnitude and frequency.

    SnakeBeta := x + 1/b * sin^2(x * a)
    """

    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class Activation1d(nn.Module):
    """1D activation wrapper."""

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


# ============================================================================
# Residual Units
# ============================================================================

class ResidualUnit(nn.Module):
    """Residual unit with SnakeBeta activation."""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================================
# Style Encoder
# ============================================================================

class StyleEncoder(nn.Module):
    """Style encoder for timbre representation."""

    def __init__(self, in_dim: int = 80, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.network = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram (batch, n_mels, seq_len).
            mask: Sequence mask (batch, 1, seq_len).

        Returns:
            Style embedding (batch, out_dim).
        """
        x = self.network(x)
        x = x * mask
        return x.sum(dim=2) / mask.sum(dim=2)


# ============================================================================
# Residual Vector Quantization
# ============================================================================

class ResidualVectorQuantize(nn.Module):
    """
    Residual Vector Quantization for FACodec.

    This implements residual vector quantization where:
    - The first codebook quantizes the input
    - Each subsequent codebook quantizes the residual from the previous
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 1,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout

        # Projection to codebook dim
        self.project_in = nn.Linear(input_dim, codebook_dim)

        # Codebooks - one codebook per RVQ level
        self.codebooks = nn.Parameter(torch.zeros(n_codebooks, codebook_size, codebook_dim))
        nn.init.xavier_uniform_(self.codebooks)

        # Projection back to input dim - use codebook_dim (not multiplied by n_codebooks)
        # because we sum all quantized outputs, each with codebook_dim
        self.project_out = nn.Linear(codebook_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        n_q: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, input_dim, seq_len).
            n_q: Number of codebooks to use.

        Returns:
            z: Quantized output (batch, input_dim, seq_len).
            codes: Quantization codes (batch, n_q, seq_len).
            latents: Latent representations.
            commit_loss: Commitment loss.
            codebook_loss: Codebook loss.
        """
        if n_q is None:
            n_q = self.n_codebooks

        batch_size, input_dim, seq_len = x.shape

        # Transpose to (batch, seq_len, input_dim)
        x = x.transpose(1, 2)

        # Project to codebook dim
        x = self.project_in(x)  # (batch, seq_len, codebook_dim)

        # Residual vector quantization
        codes = []
        residual = x
        quantized_sum = torch.zeros_like(x)
        commit_loss = 0
        codebook_loss = 0

        for i in range(n_q):
            # Get codebook
            codebook = self.codebooks[i]  # (codebook_size, codebook_dim)

            # Flatten for quantization
            residual_flat = residual.view(-1, self.codebook_dim)

            # Find nearest neighbors
            distances = (
                residual_flat.pow(2).sum(1, keepdim=True)
                - 2 * residual_flat @ codebook.t()
                + codebook.pow(2).sum(1)
            )
            code_i = torch.argmin(distances, dim=1)

            # Quantize
            quantized = F.embedding(code_i, codebook)
            quantized = quantized.view(batch_size, seq_len, self.codebook_dim)

            # Update residual
            residual = residual - quantized.detach()
            quantized_sum = quantized_sum + quantized

            # Commitment loss
            commit_loss = commit_loss + F.mse_loss(quantized, x.detach())

            # Store codes
            codes.append(code_i.view(batch_size, seq_len))

        # Stack codes
        codes = torch.stack(codes, dim=1)  # (batch, n_q, seq_len)

        # Project back to input dim
        z = self.project_out(quantized_sum)
        z = z.transpose(1, 2)  # (batch, input_dim, seq_len)

        # Compute codebook loss (simplified)
        codebook_loss = commit_loss

        return z, codes, None, commit_loss, codebook_loss

    def from_codes(
        self,
        codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode from codes.

        Args:
            codes: Quantization codes (batch, n_q, seq_len).

        Returns:
            z: Decoded output.
            codes: Same as input.
            latents: Latent representations.
            commitment: Commitment values.
        """
        batch_size, n_q, seq_len = codes.shape

        # Initialize output
        z = torch.zeros(batch_size, seq_len, self.codebook_dim * self.n_codebooks, device=codes.device)

        for i in range(n_q):
            code_i = codes[:, i, :]  # (batch, seq_len)
            codebook = self.codebooks[i]

            # Lookup embeddings
            embed = F.embedding(code_i, codebook)  # (batch, seq_len, codebook_dim)
            z[:, :, i * self.codebook_dim:(i + 1) * self.codebook_dim] = embed

        # Project back
        z = self.project_out(z)
        z = z.transpose(1, 2)  # (batch, input_dim, seq_len)

        return z, codes, None, None


# ============================================================================
# FACodec Quantizer
# ============================================================================

class FAquantizer(nn.Module):
    """
    Factorized quantizer for FACodec.

    Factorizes audio into prosody, content, timbre, and residual components.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        n_p_codebooks: int = 1,
        n_c_codebooks: int = 2,
        n_t_codebooks: int = 2,
        n_r_codebooks: int = 3,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        causal: bool = False,
        separate_prosody_encoder: bool = False,
        timbre_norm: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_p_codebooks = n_p_codebooks
        self.n_c_codebooks = n_c_codebooks
        self.n_t_codebooks = n_t_codebooks
        self.n_r_codebooks = n_r_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.causal = causal
        self.separate_prosody_encoder = separate_prosody_encoder
        self.timbre_norm = timbre_norm

        # Prosody quantizer
        self.prosody_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_p_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        # Content quantizer
        self.content_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_c_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        # Timbre handling
        if timbre_norm:
            self.timbre_encoder = StyleEncoder(
                in_dim=80, hidden_dim=512, out_dim=in_dim
            )
            self.timbre_linear = nn.Linear(in_dim, in_dim * 2)
            self.timbre_linear.bias.data[:in_dim] = 1
            self.timbre_linear.bias.data[in_dim:] = 0
            self.timbre_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        else:
            self.timbre_quantizer = ResidualVectorQuantize(
                input_dim=in_dim,
                n_codebooks=n_t_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )

        # Residual quantizer
        self.residual_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_r_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.prob_random_mask_residual = 0.75

    def forward(
        self,
        x: torch.Tensor,
        wave_segments: Optional[torch.Tensor] = None,
        n_c: Optional[int] = None,
        n_t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, in_dim, seq_len).
            wave_segments: Optional waveform for separate prosody encoder.
            n_c: Number of content codebooks to use.
            n_t: Number of timbre codebooks to use.

        Returns:
            outs: Combined quantized output.
            quantized: List of quantized components [prosody, content, timbre, residual].
            commitment_losses: Total commitment loss.
            codebook_losses: Total codebook loss.
        """
        if n_c is None:
            n_c = self.n_c_codebooks
        if n_t is None:
            n_t = self.n_t_codebooks

        outs = 0
        quantized = []
        commitment_losses = 0
        codebook_losses = 0

        # Prosody quantization
        z_p, codes_p, _, commit_loss_p, codebook_loss_p = self.prosody_quantizer(x, self.n_p_codebooks)
        outs = outs + z_p.detach()
        quantized.append(z_p)
        commitment_losses = commitment_losses + commit_loss_p
        codebook_losses = codebook_losses + codebook_loss_p

        # Content quantization
        z_c, codes_c, _, commit_loss_c, codebook_loss_c = self.content_quantizer(x, n_c)
        outs = outs + z_c.detach()
        quantized.append(z_c)
        commitment_losses = commitment_losses + commit_loss_c
        codebook_losses = codebook_losses + codebook_loss_c

        # Timbre handling
        if self.timbre_norm:
            # Use timbre encoder (not implemented in detail here)
            timbre = torch.zeros(x.shape[0], self.in_dim, device=x.device)
        else:
            timbre_residual_feature = x - z_p.detach() - z_c.detach()
            z_t, codes_t, _, commit_loss_t, codebook_loss_t = self.timbre_quantizer(
                timbre_residual_feature, n_t
            )
            outs = outs + z_t
            quantized.append(z_t)
            commitment_losses = commitment_losses + commit_loss_t
            codebook_losses = codebook_losses + codebook_loss_t

        # Residual quantization
        if self.timbre_norm:
            residual_feature = x - outs
        else:
            residual_feature = x - z_p.detach() - z_c.detach() - z_t

        z_r, codes_r, _, commit_loss_r, codebook_loss_r = self.residual_quantizer(
            residual_feature, self.n_r_codebooks
        )

        # Random mask for residual
        bsz = z_r.shape[0]
        res_mask = torch.rand(bsz, 1, 1, device=z_r.device) < (1 - self.prob_random_mask_residual)
        res_mask = res_mask.float()
        outs = outs + z_r * res_mask

        quantized.append(z_r)
        commitment_losses = commitment_losses + commit_loss_r
        codebook_losses = codebook_losses + codebook_loss_r

        return outs, quantized, commitment_losses, codebook_losses
