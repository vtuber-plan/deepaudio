# coding=utf-8
"""
BigVGAN vocoder implementation.

BigVGAN uses periodic activation functions (Snake/SnakeBeta) with anti-aliasing
for high-quality waveform generation.
"""

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List, Tuple, Optional

from transformers.utils import logging
from soniq.models.base.outputs import VocoderOutput
from soniq.models.vocoders.base import BaseVocoderModel
from soniq.models.vocoders.bigvgan.configuration_bigvgan import BigVGANConfig
from soniq.modules.activation_functions import Snake, SnakeBeta
from soniq.modules.anti_aliasing import Activation1d
from soniq.utils.model_utils import init_weights, get_padding


logger = logging.get_logger(__name__)


class AMPBlock1(nn.Module):
    """
    Anti-aliased Multi-Period Block 1 for BigVGAN.

    This block uses dilated convolutions with Snake/SnakeBeta activation
    and anti-aliasing for high-quality waveform generation.

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation sizes for residual connections.
        activation: Activation function name ("snake" or "snakebeta").
        snake_logscale: If True, snake parameters are learned in log scale.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )

        # Create activations with anti-aliasing
        for _ in range(len(self.convs1) + len(self.convs2)):
            if activation == "snake":
                act = Snake(channels, alpha_logscale=snake_logscale)
            elif activation == "snakebeta":
                act = SnakeBeta(channels, alpha_logscale=snake_logscale)
            else:
                raise ValueError(f"activation must be 'snake' or 'snakebeta', got {activation}")
            self.activations.append(Activation1d(act))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = a2(xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(nn.Module):
    """
    Anti-aliased Multi-Period Block 2 for BigVGAN (simpler variant).

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation sizes for residual connections.
        activation: Activation function name ("snake" or "snakebeta").
        snake_logscale: If True, snake parameters are learned in log scale.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        for d in dilation:
            self.convs.append(
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
            )

        # Create activations with anti-aliasing
        for _ in range(len(self.convs)):
            if activation == "snake":
                act = Snake(channels, alpha_logscale=snake_logscale)
            elif activation == "snakebeta":
                act = SnakeBeta(channels, alpha_logscale=snake_logscale)
            else:
                raise ValueError(f"activation must be 'snake' or 'snakebeta', got {activation}")
            self.activations.append(Activation1d(act))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(BaseVocoderModel):
    """
    BigVGAN neural vocoder.

    BigVGAN uses periodic activation functions (Snake/SnakeBeta) with
    anti-aliasing for high-quality waveform generation. It extends HiFiGAN
    by replacing LeakyReLU with Snake activations and adding anti-aliasing.

    Example:
        ```python
        config = BigVGANConfig()
        model = BigVGAN(config)
        mel = torch.randn(1, 128, 100)  # (batch, n_mel, time)
        output = model.synthesize(mel)  # (batch, 1, time * 512)
        ```
    """

    config_class = BigVGANConfig
    base_model_prefix = "bigvgan"
    supports_gradient_checkpointing = False

    def __init__(self, config: BigVGANConfig):
        super().__init__(config)
        self.config = config

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Pre-convolution
        self.conv_pre = weight_norm(
            Conv1d(
                in_channels=config.inter_channels,
                out_channels=config.upsample_initial_channel,
                kernel_size=config.pre_kernel_size,
                stride=1,
                padding=(config.pre_kernel_size - 1) // 2,
            )
        )

        # Select residual block type
        resblock_class = AMPBlock1 if config.resblock == "1" else AMPBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.ModuleList([
                    weight_norm(
                        ConvTranspose1d(
                            in_channels=config.upsample_initial_channel // (2 ** i),
                            out_channels=config.upsample_initial_channel // (2 ** (i + 1)),
                            kernel_size=k,
                            stride=u,
                            padding=(k - u) // 2,
                        )
                    )
                ])
            )

        # Residual blocks with AMP and anti-aliasing
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(
                        channels=ch,
                        kernel_size=k,
                        dilation=d,
                        activation=config.activation,
                        snake_logscale=config.snake_logscale,
                    )
                )

        # Post activation and convolution
        if config.activation == "snake":
            self.activation_post = Activation1d(Snake(ch, alpha_logscale=config.snake_logscale))
        elif config.activation == "snakebeta":
            self.activation_post = Activation1d(SnakeBeta(ch, alpha_logscale=config.snake_logscale))

        self.conv_post = weight_norm(
            Conv1d(
                in_channels=ch,
                out_channels=1,
                kernel_size=config.post_kernel_size,
                stride=1,
                padding=(config.post_kernel_size - 1) // 2,
                bias=False,
            )
        )

        # Initialize weights
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def synthesize(
        self,
        acoustic_features: torch.Tensor,
        **kwargs
    ) -> VocoderOutput:
        """
        Synthesize waveform from mel spectrogram.

        Args:
            acoustic_features: Mel spectrogram of shape (batch, n_mel, time).
            **kwargs: Additional arguments.

        Returns:
            VocoderOutput with waveform.
        """
        return self.forward(acoustic_features, **kwargs)

    def forward(
        self,
        acoustic_features: torch.Tensor,
        **kwargs
    ) -> VocoderOutput:
        """
        Forward pass.

        Args:
            acoustic_features: Mel spectrogram of shape (batch, n_mel, time).

        Returns:
            VocoderOutput with waveform of shape (batch, 1, time * hop_ratio).
        """
        x = acoustic_features
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsample
            for up_layer in self.ups[i]:
                x = up_layer(x)

            # Apply residual blocks and average
            xs = None
            for j in range(self.num_kernels):
                resblock = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = resblock(x)
                else:
                    xs += resblock(x)
            x = xs / self.num_kernels

        # Final activation and output
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return VocoderOutput(waveform=x)

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.ups:
            for up_layer in l:
                remove_weight_norm(up_layer)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def infer(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
        """
        Inference method for synthesizing waveform.

        Args:
            acoustic_features: Mel spectrogram of shape (batch, n_mel, time).
            **kwargs: Additional arguments.

        Returns:
            VocoderOutput with waveform.
        """
        self.eval()
        with torch.no_grad():
            return self.synthesize(acoustic_features, **kwargs)
