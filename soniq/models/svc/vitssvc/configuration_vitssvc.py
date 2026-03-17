# coding=utf-8
"""
VitsSVC configuration.
"""

from typing import List, Optional

from transformers import PretrainedConfig


class VitsSVCConfig(PretrainedConfig):
    """
    Configuration for VitsSVC model.

    VitsSVC is a singing voice conversion model based on VITS architecture,
    using content features (ContentVec, Whisper, etc.) and F0 as input.
    """

    model_type = "vitssvc"

    def __init__(
        self,
        # VITS core parameters
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_flow_layer: int = 4,
        gin_channels: int = 256,
        # Speaker parameters
        n_speakers: int = 1,
        # F0 parameters
        n_bins_f0: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        use_uv: bool = True,
        # Content encoder parameters
        content_dim: int = 256,
        use_whisper: bool = False,
        whisper_dim: int = 1024,
        use_contentvec: bool = True,
        contentvec_dim: int = 256,
        use_wenet: bool = False,
        wenet_dim: int = 512,
        # Decoder type
        decoder_type: str = "hifigan",
        # Decoder parameters (for HiFiGAN)
        resblock_kernel_sizes: Optional[List[int]] = None,
        resblock_dilation_sizes: Optional[List[List[int]]] = None,
        upsample_rates: Optional[List[int]] = None,
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: Optional[List[int]] = None,
        # Training parameters
        segment_size: int = 8192,
        spec_channels: int = 80,
        # Sample rate
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # VITS core parameters
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flow_layer = n_flow_layer
        self.gin_channels = gin_channels

        # Speaker parameters
        self.n_speakers = n_speakers

        # F0 parameters
        self.n_bins_f0 = n_bins_f0
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_uv = use_uv

        # Content encoder parameters
        self.content_dim = content_dim
        self.use_whisper = use_whisper
        self.whisper_dim = whisper_dim
        self.use_contentvec = use_contentvec
        self.contentvec_dim = contentvec_dim
        self.use_wenet = use_wenet
        self.wenet_dim = wenet_dim

        # Decoder parameters
        self.decoder_type = decoder_type
        self.resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        self.resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_rates = upsample_rates or [8, 8, 2, 2]
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 4, 4]

        # Training parameters
        self.segment_size = segment_size
        self.spec_channels = spec_channels
        self.sample_rate = sample_rate


class ConditionEncoderConfig(PretrainedConfig):
    """Configuration for condition encoder."""

    model_type = "condition_encoder"

    def __init__(
        self,
        # Content features
        use_contentvec: bool = True,
        contentvec_dim: int = 256,
        use_whisper: bool = False,
        whisper_dim: int = 1024,
        use_wenet: bool = False,
        wenet_dim: int = 512,
        # Prosody features
        use_f0: bool = True,
        n_bins_f0: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        use_uv: bool = True,
        # Speaker features
        use_spkid: bool = True,
        n_speakers: int = 1,
        # Output dimensions
        content_encoder_dim: int = 256,
        output_melody_dim: int = 256,
        output_singer_dim: int = 256,
        # Merge mode
        merge_mode: str = "add",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_contentvec = use_contentvec
        self.contentvec_dim = contentvec_dim
        self.use_whisper = use_whisper
        self.whisper_dim = whisper_dim
        self.use_wenet = use_wenet
        self.wenet_dim = wenet_dim

        self.use_f0 = use_f0
        self.n_bins_f0 = n_bins_f0
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_uv = use_uv

        self.use_spkid = use_spkid
        self.n_speakers = n_speakers

        self.content_encoder_dim = content_encoder_dim
        self.output_melody_dim = output_melody_dim
        self.output_singer_dim = output_singer_dim

        self.merge_mode = merge_mode