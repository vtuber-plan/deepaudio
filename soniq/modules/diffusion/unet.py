# coding=utf-8
"""
Diffusion model modules for Soniq.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.

    Implements DDPM-style noise scheduling.

    Args:
        num_steps: Number of diffusion steps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        schedule: Schedule type ("linear", "cosine").
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_steps = num_steps

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            t = torch.linspace(0, num_steps, num_steps + 1)
            alphas_cumprod = torch.cos(t / num_steps * torch.pi / 2) ** 2
            self.betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to data.

        Args:
            x0: Clean data.
            noise: Noise tensor.
            t: Timestep tensor.

        Returns:
            Noisy data.
        """
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x0 + torch.sqrt(1 - alpha_cumprod_t) * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps.

        Args:
            batch_size: Batch size.
            device: Device.

        Returns:
            Random timesteps.
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)


class DDIMScheduler:
    """
    DDIM scheduler for faster sampling.

    Args:
        num_steps: Number of diffusion steps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform one DDIM step.

        Args:
            x_t: Current noisy data.
            pred_noise: Predicted noise.
            t: Current timestep.
            t_prev: Previous timestep.
            eta: DDIM eta parameter.

        Returns:
            Less noisy data.
        """
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1)

        # Predict x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = pred_x0.clamp(-1, 1)

        # Direction pointing to x_t
        direction = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise

        # Variance
        variance = 0
        if eta > 0:
            variance = torch.sqrt(
                eta * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            ) * torch.randn_like(x_t)

        return torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction + variance


class UNetBlock(nn.Module):
    """
    1D UNet Block for diffusion models.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        emb_channels: Embedding channels for timestep.
        num_res_blocks: Number of residual blocks.
        dropout: Dropout probability.
        resample: Whether to upsample/downsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        resample: bool = False,
    ):
        super().__init__()
        self.resample = resample

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(in_channels if i == 0 else out_channels, out_channels, emb_channels, dropout)
            )

        if resample:
            self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, emb)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block with timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = None

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = x + self.emb_proj(F.silu(emb)).unsqueeze(-1)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + residual


class UNet1d(nn.Module):
    """
    1D UNet for diffusion models.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        base_channels: Base channel count.
        channel_mult: Channel multiplier at each level.
        num_res_blocks: Number of residual blocks per level.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)

        # Timestep embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        # Initial convolution
        self.input_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            self.down_blocks.append(
                UNetBlock(ch, out_ch, base_channels * 4, num_res_blocks, dropout, resample=(level > 0))
            )
            ch = out_ch

        # Middle blocks
        self.middle_block = UNetBlock(ch, ch, base_channels * 4, num_res_blocks, dropout)

        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            self.up_blocks.append(
                UNetBlock(ch * 2, out_ch, base_channels * 4, num_res_blocks, dropout, resample=False)
            )
            ch = out_ch

        # Output
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv1d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Timestep embedding
        t_emb = self.time_emb(t.float().view(-1, 1))

        # Initial convolution
        x = self.input_conv(x)

        # Downsample
        skips = [x]
        for block in self.down_blocks:
            x = block(x, t_emb)
            skips.append(x)

        # Middle
        x = self.middle_block(x, t_emb)

        # Upsample
        for block in self.up_blocks:
            x = block(torch.cat([x, skips.pop()], dim=1), t_emb)

        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)

        return x
