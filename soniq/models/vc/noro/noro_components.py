# coding=utf-8
"""Noro VC model components."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization for Noro.

    Sequentially quantizes the input using multiple codebooks,
    with each codebook encoding the residual from previous ones.
    """

    def __init__(
        self,
        dim: int,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim or dim

        # Project input to codebook dimension if needed
        self.project_in = nn.Linear(dim, codebook_dim) if dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()

        # Codebooks (using embeddings for lookup)
        self.embeds = nn.ModuleList([
            nn.Embedding(codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ])

        # Initialize embeddings
        for embed in self.embeds:
            nn.init.xavier_uniform_(embed.weight)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get quantized vectors from codebook indices."""
        batch, seq_len, n_codebooks = indices.shape
        codes = torch.zeros(batch, seq_len, self.codebook_dim, device=indices.device)

        for k in range(n_codebooks):
            code_indices = indices[:, :, k]
            codes = codes + F.embedding(code_indices, self.embeds[k].weight)

        return self.project_out(codes)

    def forward(
        self,
        x: torch.Tensor,
        n_quantize: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            n_quantize: Number of codebooks to use.

        Returns:
            quantized: Quantized output (batch, seq_len, dim).
            codes: Code indices (batch, seq_len, n_codebooks).
            commitment_loss: Commitment loss scalar.
        """
        x = self.project_in(x)
        quantized = torch.zeros_like(x)
        residual = x.clone()
        codes = []

        n_quantize = n_quantize or self.n_codebooks
        commitment_loss = 0.0

        for k in range(n_quantize):
            codebook = self.embeds[k].weight  # (codebook_size, codebook_dim)

            # Compute distances: ||x||^2 - 2*x*W^T + ||W||^2
            # Using einsum for proper batched distance computation
            x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            c_norm = codebook.pow(2).sum(dim=-1)  # (codebook_size,)
            distances = x_norm - 2 * torch.matmul(x, codebook.t()) + c_norm  # (batch, seq_len, codebook_size)

            # Get nearest codebook entries
            code_indices = torch.argmin(distances, dim=-1)  # (batch, seq_len)
            codes.append(code_indices)

            # Quantize
            quantized_k = F.embedding(code_indices, codebook)  # (batch, seq_len, codebook_dim)
            quantized = quantized + quantized_k

            # Update residual for next codebook
            residual = residual - quantized_k
            x = residual

        codes = torch.stack(codes, dim=-1)  # (batch, seq_len, n_codebooks)

        # Straight-through estimator
        quantized = self.project_in(x) + (quantized - self.project_in(x)).detach()
        quantized = self.project_out(quantized)

        return quantized, codes, commitment_loss


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Layer norms
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            mask: Attention mask (batch, seq_len), True means keep.

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        # Self-attention
        x_norm = self.attn_norm(x)
        if mask is not None:
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=~mask,
            )
        else:
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feedforward
        x = x + self.ffn(self.ffn_norm(x))

        return x


class CodecEncoder(nn.Module):
    """
    Codec encoder for Noro.

    Encodes audio into discrete codes using residual vector quantization.
    """

    def __init__(
        self,
        dim: int = 512,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Input projection
        self.input_proj = nn.Linear(80, dim)  # From mel spectrogram

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Residual vector quantizer
        self.quantizer = ResidualVectorQuantizer(
            dim=dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_quantize: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, 80) mel spectrogram.
            mask: Attention mask (batch, seq_len).
            n_quantize: Number of codebooks to use.

        Returns:
            codes: Quantized codes (batch, seq_len, n_codebooks).
            quantized: Quantized features (batch, seq_len, dim).
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        quantized, codes, _ = self.quantizer(x, n_quantize)

        return codes, quantized


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for Noro.

    Extracts speaker embeddings from mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int = 80,
        hidden_dim: int = 512,
        speaker_dim: int = 512,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, speaker_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, n_mel).

        Returns:
            Speaker embedding (batch, speaker_dim).
        """
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder for Noro.

    Decodes quantized features back to mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 80,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim=hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, out_dim).
        """
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.out_proj(x)
