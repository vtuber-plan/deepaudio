# coding=utf-8
"""HiFiGAN vocoder implementation."""

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List, Tuple, Optional

from transformers.utils import logging
from soniq.models.base.modeling_base import SoniqModel
from soniq.models.base.outputs import VocoderOutput
from soniq.models.vocoders.base import BaseVocoderModel
from soniq.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig
from soniq.utils.model_utils import init_weights, get_padding


logger = logging.get_logger(__name__)


class HifiGANResBlock(nn.Module):
    """
    Residual block for HiFiGAN.

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation sizes for residual connections.
        lrelu_slope: Slope for LeakyReLU activation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

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

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, self.lrelu_slope)
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


class HifiGAN(BaseVocoderModel):
    """
    HiFiGAN neural vocoder.

    This model converts mel spectrograms to waveforms using a fully
    convolutional architecture with multi-resolution residual blocks.

    Example:
        ```python
        config = HifiGANConfig()
        model = HifiGAN(config)
        mel = torch.randn(1, 80, 100)  # (batch, n_mel, time)
        output = model.synthesize(mel)  # (batch, 1, time * hop_ratio)
        ```
    """

    config_class = HifiGANConfig
    base_model_prefix = "hifigan"
    supports_gradient_checkpointing = False

    def __init__(self, config: HifiGANConfig):
        super().__init__(config)
        self.config = config

        initial_channel = config.inter_channels
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.lrelu_slope = config.lrelu_slope

        # Pre-convolution
        self.conv_pre = Conv1d(
            in_channels=initial_channel,
            out_channels=config.upsample_initial_channel,
            kernel_size=config.pre_kernel_size,
            stride=1,
            padding=(config.pre_kernel_size - 1) // 2,
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k, d) in enumerate(
            zip(
                config.upsample_rates,
                config.upsample_kernel_sizes,
                config.upsample_dilation_sizes,
            )
        ):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=config.upsample_initial_channel // (2**i),
                    out_channels=config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(((k - 1) * d + 1) - u) // 2,
                    dilation=d,
                )
            )

        # Residual blocks - create separate ModuleList for each upsample stage
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            stage_resblocks = nn.ModuleList()
            for k in config.resblock_kernel_sizes:
                for d in config.resblock_dilation_sizes:
                    stage_resblocks.append(HifiGANResBlock(ch, k, (d,), self.lrelu_slope))
            self.resblocks.append(stage_resblocks)

        # Post-convolution
        self.conv_post = Conv1d(
            in_channels=ch,
            out_channels=1,
            kernel_size=config.post_kernel_size,
            stride=1,
            padding=(config.post_kernel_size - 1) // 2,
            bias=False,
        )

        # Initialize weights
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
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            up_x = self.ups[i](x)

            # Sum residual blocks for this stage
            xs = None
            for resblock in self.resblocks[i]:
                if xs is None:
                    xs = resblock(up_x)
                else:
                    xs += resblock(up_x)
            x = xs / len(self.resblocks[i])

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return VocoderOutput(waveform=x)

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

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


class HiFiGANGenerator(nn.Module):
    """
    HiFiGAN Generator module for use as decoder in TTS/SVC models.

    This is a standalone generator without the full model wrapper,
    designed to be used as a decoder component in models like VITS or VitsSVC.

    Args:
        in_channels: Number of input channels (latent dimension).
        out_channels: Number of output channels (usually 1 for mono audio).
        upsample_rates: List of upsampling rates.
        upsample_initial_channel: Number of channels after initial convolution.
        upsample_kernel_sizes: List of kernel sizes for upsampling layers.
        resblock_kernel_sizes: List of kernel sizes for residual blocks.
        resblock_dilation_sizes: List of dilation sizes for residual blocks.
        gin_channels: Number of global conditioning channels (for speaker embedding).
        lrelu_slope: Slope for LeakyReLU activation.

    Example:
        ```python
        generator = HiFiGANGenerator(
            in_channels=192,
            out_channels=1,
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        z = torch.randn(1, 192, 100)  # Latent features
        audio = generator(z)  # (1, 1, 100 * 256)
        ```
    """

    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 1,
        upsample_rates: List[int] = None,
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = None,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        gin_channels: int = 0,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_upsamples = len(upsample_rates) if upsample_rates else 4
        self.lrelu_slope = lrelu_slope
        self.gin_channels = gin_channels

        # Default values
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        # Pre-convolution
        self.conv_pre = Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Global conditioning projection
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        in_channels=upsample_initial_channel // (2**i),
                        out_channels=upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )

            # Residual blocks for this upsampling stage
            ch = upsample_initial_channel // (2 ** (i + 1))
            for res_k, res_d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(HifiGANResBlock(ch, res_k, tuple(res_d), lrelu_slope))

        # Post-convolution
        self.conv_post = weight_norm(
            Conv1d(
                in_channels=ch,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, time).
            g: Global conditioning tensor of shape (batch, gin_channels, 1).

        Returns:
            Generated audio of shape (batch, out_channels, time * upsample_factor).
        """
        x = self.conv_pre(x)

        # Apply global conditioning
        if g is not None and self.gin_channels > 0:
            x = x + self.cond(g)

        # Upsampling and residual blocks
        resblock_idx = 0
        for i, up in enumerate(self.ups):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = up(x)

            # Sum residual blocks
            xs = None
            for _ in range(len(self.resblocks) // self.num_upsamples):
                if resblock_idx < len(self.resblocks):
                    if xs is None:
                        xs = self.resblocks[resblock_idx](x)
                    else:
                        xs = xs + self.resblocks[resblock_idx](x)
                    resblock_idx += 1

            if xs is not None:
                x = xs / (len(self.resblocks) // self.num_upsamples)

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for up in self.ups:
            remove_weight_norm(up)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_post)
