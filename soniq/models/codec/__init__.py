# coding=utf-8
"""Neural audio codec models."""

from soniq.models.codec.base import BaseCodecModel
from soniq.models.codec.dualcodec import DualCodec, DualCodecConfig

__all__ = [
    "BaseCodecModel",
    "DualCodec",
    "DualCodecConfig",
]
