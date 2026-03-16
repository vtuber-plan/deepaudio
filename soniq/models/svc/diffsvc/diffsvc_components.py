# coding=utf-8
"""DiffSVC model components."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


def get_beta_schedule(
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """Get the noise schedule for diffusion."""
    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_type == "cosine":
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    return betas


class DiffusionEmbedding(nn.Module):
    """
    Diffusion step embedding.

    Embeds the diffusion timestep into a high-dimensional space.
    """

    def __init__(self, num_steps: int, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            "frequencies",
            torch.exp(torch.arange(dim // 2) * -(math.log(num_steps) / (dim // 2))),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: Diffusion timestep (batch,).

        Returns:
            Embedded timestep (batch, dim).
        """
        # timestep: (batch,) -> (batch, 1)
        timestep = timestep.float().unsqueeze(-1)
        # freqs: (batch, dim//2)
        freqs = timestep * self.frequencies.unsqueeze(0)
        # x: (batch, dim)
        x = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return x


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Conditions the network on diffusion timestep and speaker embedding.
    """

    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.modulator = nn.Linear(cond_dim, in_dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            cond: Condition tensor (batch, cond_dim) or (batch, seq_len, dim).

        Returns:
            Modulated output (batch, seq_len, dim).
        """
        # Handle 3D condition (batch, seq_len, cond_dim)
        if cond.dim() == 3:
            gamma, beta = self.modulator(cond).chunk(2, dim=-1)
            return x * (1 + gamma) + beta
        # Handle 2D condition (batch, cond_dim)
        else:
            gamma, beta = self.modulator(cond).chunk(2, dim=-1)
            return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ConformerBlock(nn.Module):
    """
    Conformer block for DiffSVC.

    Combines convolution and self-attention for sequence modeling.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.dim = dim

        # FFN 1
        self.ffn1_norm = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Convolution
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Dropout(dropout),
        )

        # FFN 2
        self.ffn2_norm = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

        # FiLM for conditioning
        self.film = FiLM(dim, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            cond: Condition tensor (batch, seq_len, dim) or (batch, dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        # FFN 1
        x = x + self.ffn1(self.ffn1_norm(x))

        # Self-attention
        x_attn = self.attention_norm(x)
        if mask is not None:
            attn_out, _ = self.attention(
                x_attn, x_attn, x_attn,
                key_padding_mask=~mask,
            )
        else:
            attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + attn_out

        # Convolution
        x_conv = self.conv_norm(x).transpose(1, 2)
        x_conv = self.conv(x_conv).transpose(1, 2)
        x = x + x_conv

        # FFN 2
        x = x + self.ffn2(self.ffn2_norm(x))

        # Apply FiLM conditioning (handle both 2D and 3D cond)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (batch, 1, dim) -> broadcast to (batch, seq_len, dim)
        x = self.film(x, cond)

        return x


class UNet1D(nn.Module):
    """
    1D U-Net for diffusion.

    Processes sequence data with downsampling and upsampling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # Encoder
        self.encoder = nn.ModuleList([
            ConformerBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                cond_dim=cond_dim,
            )
            for _ in range(n_layers)
        ])

        # Bottleneck
        self.bottleneck = ConformerBlock(
            dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            cond_dim=cond_dim,
        )

        # Decoder
        self.decoder = nn.ModuleList([
            ConformerBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                cond_dim=cond_dim,
            )
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_dim).
            cond: Condition tensor (batch, cond_dim).
            mask: Attention mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, out_dim).
        """
        x = self.in_proj(x)

        # Encoder
        skips = []
        for layer in self.encoder:
            x = layer(x, cond, mask)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x, cond, mask)

        # Decoder with skip connections
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x + skip, cond, mask)

        return self.out_proj(x)


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for DiffSVC.

    Extracts speaker embeddings from mel spectrograms.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 512,
        out_dim: int = 512,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_dim).

        Returns:
            Speaker embedding (batch, out_dim).
        """
        return self.layers(x)


class F0Encoder(nn.Module):
    """
    F0 encoder for DiffSVC.

    Encodes F0 contour into embeddings.
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: F0 contour (batch, seq_len, 1).

        Returns:
            F0 embedding (batch, seq_len, out_dim).
        """
        x = self.in_proj(f0)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)
