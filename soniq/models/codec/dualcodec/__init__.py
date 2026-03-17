# coding=utf-8
"""
DualCodec: Low frame-rate semantic-enhanced neural audio codec.

DualCodec combines semantic features from W2V-BERT with acoustic features
using a dual-path quantization approach.
"""

from .configuration_dualcodec import DualCodecConfig
from .modeling_dualcodec import DualCodec
from .dac import DAC, Encoder, Decoder
from .quantize import VectorQuantize, ResidualVectorQuantize
from .cnn import ConvNeXtBlock, AdaLayerNorm
from .dac_layers import Snake1d, WNConv1d, WNConvTranspose1d

__all__ = [
    "DualCodecConfig",
    "DualCodec",
    "DAC",
    "Encoder",
    "Decoder",
    "VectorQuantize",
    "ResidualVectorQuantize",
    "ConvNeXtBlock",
    "AdaLayerNorm",
    "Snake1d",
    "WNConv1d",
    "WNConvTranspose1d",
]