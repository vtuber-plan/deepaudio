# coding=utf-8
"""
VITS modules.

This module contains the core components for VITS:
- Text Encoder
- Posterior Encoder
- Residual Coupling Block (Flow)
- Duration Predictors
- Generator (based on HiFiGAN)
"""

import math
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, Embedding
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import functional as F
from typing import Tuple, Optional, List
from soniq.modules.transformer.encoder import TransformerEncoder
from soniq.utils.model_utils import get_padding, init_weights
from soniq.modules.activation_functions import Snake, SnakeBeta
from soniq.modules.anti_aliasing import Activation1d


# ============================================================================
# Utility Functions
# ============================================================================

def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """
    Create a sequence mask.

    Args:
        length: Length tensor of shape (batch,).
        max_length: Maximum length.

    Returns:
        Boolean mask of shape (batch, max_length).
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Generate path from duration.

    Args:
        duration: Duration tensor of shape (batch, t_x).
        mask: Mask tensor of shape (batch, t_x, t_y).

    Returns:
        Path matrix of shape (batch, t_x, t_y).
    """
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path * mask

    return path


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slice segments from input.

    Args:
        x: Input tensor of shape (batch, channels, time).
        x_lengths: Length tensor of shape (batch,).
        segment_size: Size of segment to slice.

    Returns:
        Tuple of (sliced tensor, start indices).
    """
    b, c, t = x.shape

    # Handle case where segment_size is larger than input
    segment_size = min(segment_size, t)

    max_start = x_lengths - segment_size
    start = (torch.rand(b, device=x.device) * max_start.clamp(min=0)).long()
    ids_slice = start.unsqueeze(-1) + torch.arange(segment_size, device=x.device)
    ids_slice = ids_slice.unsqueeze(1).expand(b, c, segment_size)

    x_slice = x.gather(2, ids_slice)
    return x_slice, start


# ============================================================================
# Text Encoder
# ============================================================================

class TextEncoder(nn.Module):
    """
    Text Encoder for VITS.

    Encodes text/phoneme sequences into hidden representations.

    Args:
        n_vocab: Vocabulary size.
        out_channels: Output dimension.
        hidden_channels: Hidden dimension.
        filter_channels: FFN hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        kernel_size: Kernel size for convolutions.
        p_dropout: Dropout rate.
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # Use TransformerEncoder from soniq.modules.transformer
        self.encoder = TransformerEncoder(
            d_model=hidden_channels,
            n_heads=n_heads,
            d_ff=filter_channels,
            n_layers=n_layers,
            dropout=p_dropout,
            norm_first=True,
        )

        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len).
            x_lengths: Length tensor of shape (batch,).

        Returns:
            Tuple of (x, m, logs, x_mask) where:
            - x: Encoded features (hidden before proj)
            - m: Mean of latent distribution
            - logs: Log standard deviation
            - x_mask: Mask tensor
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        x = self.encoder(x.transpose(1, -1), x_mask.squeeze(1))
        x = x.transpose(1, -1) * x_mask

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Return hidden features (before proj) as the first element
        return x, m, logs, x_mask


# ============================================================================
# Posterior Encoder
# ============================================================================

class WN(torch.nn.Module):
    """
    WaveNet-style encoder for Posterior Encoder.

    Args:
        in_channels: Input channels.
        hidden_channels: Hidden dimension.
        kernel_size: Kernel size for dilated convolutions.
        n_layers: Number of WaveNet layers.
        gin_channels: Speaker embedding channels.
        p_dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size * dilation - dilation) // 2

            in_layer = Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer)
            self.in_layers.append(in_layer)

            res_skip_layer = Conv1d(hidden_channels, hidden_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

        # Input projection
        self.input_proj = Conv1d(in_channels, hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, time).
            x_mask: Mask tensor of shape (batch, 1, time).
            g: Optional speaker embedding of shape (batch, gin_channels, 1).

        Returns:
            Output tensor of shape (batch, out_channels * 2, time).
        """
        x = self.input_proj(x) * x_mask
        x = self.drop(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            h = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                h = h + g_l

            # Split into gate and output
            a, b = h.split(self.hidden_channels, dim=1)
            a = torch.tanh(a)
            b = torch.sigmoid(b)
            h = a * b
            h = self.drop(h)

            h = self.res_skip_layers[i](h)
            x = (x + h) * x_mask

        return x * x_mask

    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder for VITS.

    Encodes mel spectrograms into latent space.

    Args:
        in_channels: Input channels (n_mel).
        out_channels: Output channels.
        hidden_channels: Hidden dimension.
        kernel_size: Kernel size.
        n_layers: Number of layers.
        gin_channels: Speaker embedding channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            gin_channels=gin_channels,
            p_dropout=0.0,
        )
        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input mel spectrogram of shape (batch, n_mel, time).
            x_lengths: Length tensor of shape (batch,).
            g: Optional speaker embedding.

        Returns:
            Tuple of (z, m, logs, x_mask).
        """
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


# ============================================================================
# Residual Coupling Block (Flow)
# ============================================================================

class ResidualCouplingLayer(nn.Module):
    """
    Residual coupling layer for normalizing flow.

    Args:
        in_channels: Input channels.
        hidden_channels: Hidden dimension.
        kernel_size: Kernel size.
        dilation_rate: Dilation rate.
        n_layers: Number of layers.
        gin_channels: Speaker embedding channels.
        mean_only: If True, only predict mean (not log_std).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.mean_only = mean_only
        self.p_dropout = p_dropout

        self.pre = Conv1d(in_channels // 2, hidden_channels, 1)
        self.enc = WN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            gin_channels=gin_channels,
            p_dropout=p_dropout,
        )
        self.post = Conv1d(hidden_channels, in_channels // 2 * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Optional speaker embedding.
            reverse: If True, apply inverse transformation.

        Returns:
            Transformed tensor.
        """
        x0, x1 = torch.split(x, [self.in_channels // 2] * 2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.in_channels // 2] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class Flip(nn.Module):
    """Flip layer for normalizing flow."""

    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingBlock(nn.Module):
    """
    Residual coupling block for VITS.

    Args:
        in_channels: Input channels.
        hidden_channels: Hidden dimension.
        kernel_size: Kernel size.
        dilation_rate: Dilation rate.
        n_layers: Number of layers.
        gin_channels: Speaker embedding channels.
        n_flows: Number of flow steps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        n_flows: int = 4,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.n_flows = n_flows
        self.p_dropout = p_dropout

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    in_channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels,
                    mean_only=True,
                    p_dropout=p_dropout,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Optional speaker embedding.
            reverse: If True, apply inverse transformation.

        Returns:
            Transformed tensor.
        """
        output = x
        if not reverse:
            for flow in self.flows:
                if isinstance(flow, Flip):
                    output, _ = flow(output)
                else:
                    output, _ = flow(output, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, Flip):
                    output = flow(output, reverse=reverse)
                else:
                    output = flow(output, x_mask, g=g, reverse=reverse)
        return output

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()


# ============================================================================
# Duration Predictors
# ============================================================================

class DurationPredictor(nn.Module):
    """
    Duration Predictor for VITS.

    Predicts the duration of each phoneme.

    Args:
        in_channels: Input channels.
        filter_channels: Filter channels.
        kernel_size: Kernel size.
        p_dropout: Dropout rate.
        gin_channels: Speaker embedding channels.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels)
        self.conv_2 = Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels)
        self.proj = Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, time).
            x_mask: Mask tensor.
            g: Optional speaker embedding.

        Returns:
            Duration predictions of shape (batch, 1, time).
        """
        if g is not None:
            x = x + self.cond(g)
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.norm_1(x.transpose(1, -1)).transpose(1, -1)
        x = self.drop(x)
        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.norm_2(x.transpose(1, -1)).transpose(1, -1)
        x = self.drop(x)
        x = self.proj(x)
        return x * x_mask


class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor for VITS.

    Uses normalizing flows for probabilistic duration prediction.

    Args:
        in_channels: Input channels.
        filter_channels: Filter channels.
        kernel_size: Kernel size.
        p_dropout: Dropout rate.
        n_flows: Number of flow steps.
        gin_channels: Speaker embedding channels.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        # affine_flow operates on filter_channels dimension
        self.affine_flow = AffineFlow(filter_channels, n_flows)
        self.conv_pre = Conv1d(in_channels, filter_channels, 1)
        self.conv_post = Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            w: Target duration (for training).
            g: Optional speaker embedding.
            reverse: If True, sample from prior.
            noise_scale: Noise scale for sampling.

        Returns:
            Duration predictions or sampled durations.
        """
        if g is not None:
            g = self.cond(g)

        x = self.conv_pre(x)
        x = F.gelu(x)

        if reverse:
            # Sampling - noise should match filter_channels dimension
            z = torch.randn(x.shape[0], x.shape[1], x.shape[2], device=x.device) * noise_scale
            z = self.affine_flow(z, x_mask, reverse=True)
            logw = self.conv_post(x + z) * x_mask
            return logw
        else:
            # Training: compute NLL
            logw = self.conv_post(x) * x_mask
            if w is not None:
                logw_ = torch.log(w + 1e-6) * x_mask
                loss = torch.sum((logw - logw_) ** 2) / torch.sum(x_mask)
                return loss
            return logw

    def remove_weight_norm(self):
        pass


class Log(nn.Module):
    """Log transform layer."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5))
            logdet = torch.sum(-y, [1, 2])
            return y * x_mask, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class AffineFlow(nn.Module):
    """
    Affine flow for stochastic duration predictor.

    Args:
        in_channels: Input channels.
        n_flows: Number of flow steps.
    """

    def __init__(self, in_channels: int, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                nn.ModuleList([
                    Conv1d(in_channels, in_channels * 2, 1),
                    Flip(),
                ])
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                m, s = flow[0](x).chunk(2, dim=1)
                s = torch.sigmoid(s)
                x = (x + m) * s
                x, _ = flow[1](x)
            return x
        else:
            for flow in reversed(self.flows):
                x = flow[1](x, reverse=True)
                m, s = flow[0](x).chunk(2, dim=1)
                s = torch.sigmoid(s)
                x = x / s - m
            return x


# ============================================================================
# Generator (HiFiGAN-based)
# ============================================================================

class HifiGANGenerator(nn.Module):
    """
    Generator for VITS (based on HiFiGAN).

    Args:
        initial_channel: Initial channels.
        resblock: Residual block type.
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilation_sizes: Dilation sizes for residual blocks.
        upsample_rates: Upsampling rates.
        upsample_initial_channel: Initial channels for upsampling.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        gin_channels: Speaker embedding channels.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)
        else:
            self.cond = None

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class ResBlock(nn.Module):
    """
    Residual block for generator.

    Args:
        channels: Number of channels.
        kernel_size: Kernel size.
        dilation: Dilation sizes.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + residual
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)
