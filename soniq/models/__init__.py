# coding=utf-8
"""Models for Soniq."""

from soniq.models.base.modeling_base import SoniqModel
from soniq.models.base.configuration_base import SoniqModelConfig

# Aliases for backwards compatibility
SoniqPreTrainedModel = SoniqModel
SoniqConfig = SoniqModelConfig

__all__ = [
    "SoniqModel",
    "SoniqModelConfig",
    "SoniqPreTrainedModel",
    "SoniqConfig",
]