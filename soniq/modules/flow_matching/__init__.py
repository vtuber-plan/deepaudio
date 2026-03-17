# coding=utf-8
"""Flow Matching: Continuous-time normalizing flows via ODE.

Flow Matching learns a continuous-time flow from source to target distribution.
Unlike discrete normalizing flows, it parameterizes the flow as the solution
to an ODE, enabling flexible architectures and efficient training.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""

import math
from typing import Callable, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding.

        Args:
            t: Timesteps (B,)

        Returns:
            Embeddings (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding with MLP projection.

    Args:
        time_dim: Time embedding dimension
        hidden_dim: Hidden dimension for MLP
    """

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embedding.

        Args:
            t: Timesteps (B,)

        Returns:
            Time embeddings (B, hidden_dim)
        """
        t_emb = self.sinusoidal(t)
        return self.mlp(t_emb)


class FlowMatcher(nn.Module):
    """Flow Matching base class.

    Implements conditional flow matching for learning continuous-time flows.

    Args:
        sigma: Noise parameter for interpolation (default: 1e-5)
        time_scheduler: Time scheduler type ("linear" or "cosine")
    """

    def __init__(
        self,
        sigma: float = 1e-5,
        time_scheduler: str = "linear",
    ):
        super().__init__()
        self.sigma = sigma
        self.time_scheduler = time_scheduler

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Timesteps (B,)
        """
        t = torch.rand(batch_size, device=device)
        t = torch.clamp(t, 1e-5, 1.0)

        # Apply time scheduler
        if self.time_scheduler == "cosine":
            t = 1 - torch.cos(t * math.pi * 0.5)

        return t

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and data.

        Flow Matching interpolation:
        xt = (1 - (1 - sigma) * t) * x0 + t * x1

        Args:
            x0: Source (noise) (B, T, D)
            x1: Target (data) (B, T, D)
            t: Timesteps (B,)

        Returns:
            Interpolated samples (B, T, D)
        """
        t = t.view(-1, 1, 1)  # (B, 1, 1)
        xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
        return xt

    def compute_flow_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow target (ground truth velocity).

        Flow target: v = x1 - (1 - sigma) * x0

        Args:
            x0: Source (noise)
            x1: Target (data)

        Returns:
            Flow target (velocity)
        """
        return x1 - (1 - self.sigma) * x0

    def compute_loss(
        self,
        flow_pred: torch.Tensor,
        flow_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            flow_pred: Predicted flow (B, T, D)
            flow_target: Target flow (B, T, D)
            mask: Optional mask (B, T, 1)

        Returns:
            Loss value
        """
        loss = F.mse_loss(flow_pred, flow_target, reduction='none')

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class ConditionalFlowMatcher(FlowMatcher):
    """Conditional Flow Matching.

    Extends FlowMatcher with conditioning support.

    Args:
        sigma: Noise parameter
        time_scheduler: Time scheduler type
    """

    def __init__(
        self,
        sigma: float = 1e-5,
        time_scheduler: str = "linear",
    ):
        super().__init__(sigma, time_scheduler)

    def forward_train(
        self,
        x1: torch.Tensor,
        velocity_net: nn.Module,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            x1: Target data (B, T, D)
            velocity_net: Velocity prediction network
            condition: Optional conditioning (B, T, D_cond)
            mask: Optional mask (B, T, 1)

        Returns:
            Tuple of (loss, flow_target)
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample timesteps
        t = self.sample_t(batch_size, device)

        # Sample noise
        x0 = torch.randn_like(x1)

        # Interpolate
        xt = self.interpolate(x0, x1, t)

        # Compute flow target
        flow_target = self.compute_flow_target(x0, x1)

        # Predict flow
        flow_pred = velocity_net(xt, t, condition, mask)

        # Compute loss
        loss = self.compute_loss(flow_pred, flow_target, mask)

        return loss, flow_target


class ODESolver:
    """ODE solver for flow matching inference.

    Supports Euler and Heun (2nd order) methods.

    Args:
        method: Solver method ("euler" or "heun")
    """

    def __init__(self, method: str = "euler"):
        self.method = method

    @torch.no_grad()
    def euler_step(
        self,
        velocity_net: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler method step.

        Args:
            velocity_net: Velocity network
            xt: Current state (B, T, D)
            t: Current time (B,)
            dt: Step size
            condition: Optional conditioning
            mask: Optional mask

        Returns:
            Next state
        """
        v = velocity_net(xt, t, condition, mask)
        return xt + v * dt

    @torch.no_grad()
    def heun_step(
        self,
        velocity_net: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Heun's method (2nd order Runge-Kutta) step.

        Args:
            velocity_net: Velocity network
            xt: Current state (B, T, D)
            t: Current time (B,)
            dt: Step size
            condition: Optional conditioning
            mask: Optional mask

        Returns:
            Next state
        """
        # First evaluation
        v1 = velocity_net(xt, t, condition, mask)
        xt_next = xt + v1 * dt

        # Second evaluation at predicted point
        t_next = t + dt
        v2 = velocity_net(xt_next, t_next, condition, mask)

        # Average
        return xt + 0.5 * (v1 + v2) * dt

    @torch.no_grad()
    def solve(
        self,
        velocity_net: nn.Module,
        x0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        n_steps: int = 10,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Solve ODE from noise to data.

        Args:
            velocity_net: Velocity network
            x0: Initial noise (B, T, D)
            condition: Optional conditioning
            mask: Optional mask
            n_steps: Number of integration steps
            return_trajectory: Whether to return all intermediate states

        Returns:
            Final sample (B, T, D) or trajectory (n_steps, B, T, D)
        """
        dt = 1.0 / n_steps
        xt = x0.clone()
        trajectory = [xt] if return_trajectory else None

        for i in range(n_steps):
            t = torch.full((x0.shape[0],), i * dt, device=x0.device)

            if self.method == "euler":
                xt = self.euler_step(velocity_net, xt, t, dt, condition, mask)
            elif self.method == "heun":
                xt = self.heun_step(velocity_net, xt, t, dt, condition, mask)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if return_trajectory:
                trajectory.append(xt)

        if return_trajectory:
            return xt, torch.stack(trajectory)
        return xt


class VelocityNet(nn.Module):
    """Base velocity network for flow matching.

    Subclasses should implement the forward method.

    Args:
        data_dim: Data dimension
        hidden_dim: Hidden dimension
        time_dim: Time embedding dimension
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Time embedding
        self.time_emb = TimeEmbedding(time_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity.

        Args:
            x: Noisy input (B, T, D)
            t: Timesteps (B,)
            condition: Optional conditioning (B, T, D_cond)
            mask: Optional mask (B, T, 1)

        Returns:
            Predicted velocity (B, T, D)
        """
        raise NotImplementedError("Subclasses must implement forward")


class TransformerVelocityNet(VelocityNet):
    """Transformer-based velocity network.

    Args:
        data_dim: Data dimension
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_dim: int = 128,
    ):
        super().__init__(data_dim, hidden_dim, time_dim)

        # Condition projection
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)  # Assume condition has hidden_dim

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity using transformer.

        Args:
            x: Noisy input (B, T, D)
            t: Timesteps (B,)
            condition: Optional conditioning (B, T, D_cond)
            mask: Optional mask (B, T, 1) - True for valid, False for padding

        Returns:
            Predicted velocity (B, T, D)
        """
        # Input projection
        h = self.input_proj(x)

        # Time embedding
        t_emb = self.time_emb(t)  # (B, hidden_dim)
        h = h + t_emb.unsqueeze(1)  # Add to all positions

        # Add condition
        if condition is not None:
            cond = self.cond_proj(condition)
            h = h + cond

        # Create attention mask for transformer
        src_key_padding_mask = None
        if mask is not None:
            # mask is (B, T, 1), convert to boolean and invert for transformer
            mask_bool = mask.squeeze(-1).bool() if mask.dtype != torch.bool else mask.squeeze(-1)
            src_key_padding_mask = ~mask_bool  # (B, T), True for padding

        # Transformer layers
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        h = self.norm(h)

        # Output projection
        v = self.output_proj(h)

        return v


class FlowMatchingModel(nn.Module):
    """Complete Flow Matching model.

    Combines flow matching training with velocity network.

    Args:
        data_dim: Data dimension
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        sigma: Noise parameter
        time_scheduler: Time scheduler type
        solver_method: ODE solver method
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        sigma: float = 1e-5,
        time_scheduler: str = "linear",
        solver_method: str = "euler",
    ):
        super().__init__()
        self.data_dim = data_dim

        # Velocity network
        self.velocity_net = TransformerVelocityNet(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Flow matcher
        self.flow_matcher = ConditionalFlowMatcher(
            sigma=sigma,
            time_scheduler=time_scheduler,
        )

        # ODE solver
        self.solver = ODESolver(method=solver_method)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward pass.

        Args:
            x: Target data (B, T, D)
            condition: Optional conditioning
            mask: Optional mask

        Returns:
            Loss value
        """
        loss, _ = self.flow_matcher.forward_train(
            x1=x,
            velocity_net=self.velocity_net,
            condition=condition,
            mask=mask,
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Generate samples via ODE.

        Args:
            shape: Output shape (B, T, D)
            condition: Optional conditioning
            mask: Optional mask
            n_steps: Number of integration steps

        Returns:
            Generated samples (B, T, D)
        """
        device = next(self.parameters()).device
        x0 = torch.randn(shape, device=device)

        return self.solver.solve(
            velocity_net=self.velocity_net,
            x0=x0,
            condition=condition,
            mask=mask,
            n_steps=n_steps,
        )


__all__ = [
    "SinusoidalPosEmb",
    "TimeEmbedding",
    "FlowMatcher",
    "ConditionalFlowMatcher",
    "ODESolver",
    "VelocityNet",
    "TransformerVelocityNet",
    "FlowMatchingModel",
]