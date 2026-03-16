# coding=utf-8
"""
Noise scheduler for diffusion models.
"""

import torch


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
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

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
        return (
            torch.sqrt(alpha_cumprod_t) * x0
            + torch.sqrt(1 - alpha_cumprod_t) * noise
        )

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Sample random timesteps.

        Args:
            batch_size: Batch size.
            device: Device.

        Returns:
            Random timesteps.
        """
        return torch.randint(
            0, self.num_steps, (batch_size,), device=device, dtype=torch.long
        )

    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get variance at timestep t.

        Args:
            t: Timestep tensor.

        Returns:
            Variance.
        """
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]

        beta_t = self.betas[t]
        variance = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * beta_t

        return variance.clamp(1e-20)
