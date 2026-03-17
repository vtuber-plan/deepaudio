# coding=utf-8
"""
Vector Quantization modules for DualCodec.

Contains VectorQuantize and ResidualVectorQuantize implementations.
"""

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dac_layers import WNConv1d


class VectorQuantize(nn.Module):
    """
    Vector Quantization module.

    Implementation similar to Karpathy's deep-vector-quantization with
    improvements from Improved VQGAN:
    - Factorized codes for improved codebook usage
    - L2-normalized codes for training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor):
        """
        Quantize input tensor.

        Args:
            z: Input tensor (B, D, T).

        Returns:
            z_q: Quantized tensor (B, D, T).
            commitment_loss: Commitment loss.
            codebook_loss: Codebook loss.
            indices: Codebook indices (B, T).
            z_e: Projected latents (B, codebook_dim, T).
        """
        # Project to codebook dimension
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        # Compute losses
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Embed codebook indices."""
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to latent vectors."""
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        """Decode latents to quantized representation."""
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute distances
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Residual Vector Quantization.

    Introduced in SoundStream: An end-to-end neural audio codec.
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList([
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ])

    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int = None,
        possibly_no_quantizer: bool = False,
    ):
        """
        Quantize input tensor using residual VQ.

        Args:
            z: Input tensor (B, D, T).
            n_quantizers: Number of quantizers to use.
            possibly_no_quantizer: Allow zero quantizers during training.

        Returns:
            z_q: Quantized tensor (B, D, T).
            codes: Codebook indices (B, N, T).
            latents: Projected latents (B, N*D, T).
            commitment_loss: Commitment loss.
            codebook_loss: Codebook loss.
            z_q_1: First layer quantized output.
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if self.training:
            n_quantizers_tensor = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            if possibly_no_quantizer:
                dropout = torch.randint(0, self.n_codebooks + 1, (z.shape[0],))
            else:
                dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers_tensor[:n_dropout] = dropout[:n_dropout]
            n_quantizers_tensor = n_quantizers_tensor.to(z.device)
        else:
            n_quantizers_tensor = torch.tensor([n_quantizers] * z.shape[0], device=z.device)

        z_q_1 = None

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            if i == 0:
                z_q_1 = z_q_i.clone()

            # Apply quantizer dropout mask
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers_tensor)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, z_q_1

    def from_codes(self, codes: torch.Tensor):
        """
        Reconstruct continuous representation from codes.

        Args:
            codes: Codebook indices (B, N, T).

        Returns:
            z_q: Quantized tensor (B, D, T).
            z_p: Projected latents (B, N*D, T).
            codes: Input codes.
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]

        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """
        Reconstruct from unquantized latents.

        Args:
            latents: Continuous latents (B, N*D, T).

        Returns:
            z_q: Quantized representation.
            z_p: Latent space representation.
            codes: Codebook indices.
        """
        import numpy as np
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]

        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)