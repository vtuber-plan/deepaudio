# coding=utf-8
"""Base module for Soniq models."""

from soniq.models.base.configuration_base import SoniqModelConfig
from soniq.models.base.modeling_base import SoniqModel
from soniq.models.base.outputs import (
    ModelOutput,
    VocoderOutput,
    TTSOutput,
    CodecOutput,
    SVCOutput,
)

__all__ = [
    "SoniqModelConfig",
    "SoniqModel",
    "ModelOutput",
    "VocoderOutput",
    "TTSOutput",
    "CodecOutput",
    "SVCOutput",
]
