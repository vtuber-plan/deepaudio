# coding=utf-8
"""Vocos discriminator modules."""

from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
    use_complex: bool = False,
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor (B, T)
        fft_size: FFT size
        hop_size: Hop size
        win_length: Window length
        window: Window function tensor
        use_complex: If True, return complex spectrogram

    Returns:
        Magnitude spectrogram (B, #frames, fft_size // 2 + 1) or complex spectrogram
    """
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    if not use_complex:
        # Return magnitude spectrogram
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(1, 2)
    else:
        # Return complex spectrogram in real/imag format
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res


class HiFiGANPeriodDiscriminator(nn.Module):
    """HiFiGAN period discriminator module.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        period: Period for reshaping
        kernel_sizes: Kernel sizes for conv layers
        channels: Number of initial channels
        downsample_scales: List of downsampling scales
        max_downsample_channels: Maximum downsampling channels
        channel_increasing_factor: Channel increasing factor
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        period: int = 3,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 32,
        downsample_scales: List[int] = [3, 3, 3, 3, 1],
        max_downsample_channels: int = 1024,
        channel_increasing_factor: int = 4,
    ):
        super().__init__()
        self.period = period

        # Build conv layers
        self.convs = nn.ModuleList()
        in_chs = in_channels
        out_chs = channels

        for downsample_scale in downsample_scales:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            in_chs = out_chs
            out_chs = min(out_chs * channel_increasing_factor, max_downsample_channels)

        # Output conv
        self.output_conv = nn.Conv2d(
            in_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        # Apply weight norm
        self.apply_weight_norm()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            List of layer outputs
        """
        # Transform 1d to 2d: (B, C, T) -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # Forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs.append(x)
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs.append(x)

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization to all conv layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                weight_norm(m)

        self.apply(_apply_weight_norm)


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """HiFiGAN multi-period discriminator.

    Args:
        periods: List of periods
    """

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            HiFiGANPeriodDiscriminator(period=p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input noise signal (B, 1, T)

        Returns:
            List of discriminator outputs
        """
        outs = []
        for disc in self.discriminators:
            outs.append(disc(x))
        return outs


class NLayerSpecDiscriminator(nn.Module):
    """2D CNN discriminator for spectrogram.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_sizes: Kernel sizes
        channels: Initial channels
        max_downsample_channels: Maximum downsampling channels
        downsample_scales: Downsampling scales
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Tuple[int, int] = (5, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Tuple[int, ...] = (2, 2, 2),
    ):
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        # Initial layer
        model["layer_0"] = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling layers
        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            model[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
            in_chs = out_chs

        # Output layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[1],
                padding=kernel_sizes[1] // 2,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        model[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )

        self.model = model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input spectrogram (B, C, T, F)

        Returns:
            List of layer outputs
        """
        results = []
        for layer in self.model.values():
            x = layer(x)
            results.append(x)
        return results


class SpecDiscriminator(nn.Module):
    """Spectrogram discriminator with multi-resolution STFT.

    Args:
        stft_params: STFT parameters dict
        in_channels: Input channels
        out_channels: Output channels
        kernel_sizes: Kernel sizes for discriminator
        channels: Initial channels
        max_downsample_channels: Maximum downsampling channels
        downsample_scales: Downsampling scales
        use_complex: If True, use complex spectrogram
    """

    def __init__(
        self,
        stft_params: Optional[dict] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Tuple[int, int] = (7, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Tuple[int, ...] = (2, 2, 2),
        use_complex: bool = False,
    ):
        super().__init__()
        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240],
                "window": "hann_window",
            }

        self.stft_params = stft_params
        self.use_complex = use_complex

        # Build discriminators for each FFT size
        self.discriminators = nn.ModuleList()
        for _ in stft_params["fft_sizes"]:
            self.discriminators.append(
                NLayerSpecDiscriminator(
                    in_channels=2 if use_complex else 1,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    downsample_scales=downsample_scales,
                )
            )

        # Register window buffers
        for win_length in stft_params["win_lengths"]:
            window = getattr(torch, stft_params["window"])(win_length)
            self.register_buffer(f"window_{win_length}", window)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input audio (B, 1, T)

        Returns:
            List of discriminator outputs for each resolution
        """
        x = x.squeeze(1)  # (B, T)
        results = []

        for i, disc in enumerate(self.discriminators):
            # Compute STFT
            spec = stft(
                x,
                self.stft_params["fft_sizes"][i],
                self.stft_params["hop_sizes"][i],
                self.stft_params["win_lengths"][i],
                getattr(self, f"window_{self.stft_params['win_lengths'][i]}"),
                use_complex=self.use_complex,
            )

            # Format for discriminator
            if not self.use_complex:
                spec = spec.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
            else:
                spec = spec.transpose(2, 3)  # (B, 2, T, F) -> (B, 2, F, T)

            results.append(disc(spec))

        return results


__all__ = [
    "HiFiGANMultiPeriodDiscriminator",
    "HiFiGANPeriodDiscriminator",
    "SpecDiscriminator",
    "NLayerSpecDiscriminator",
]
