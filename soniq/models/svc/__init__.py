# coding=utf-8
"""Singing voice conversion models."""

from soniq.models.svc.base import BaseSVCModel
from soniq.models.svc.vitssvc import VitsSVC, VitsSVCConfig

__all__ = [
    "BaseSVCModel",
    "VitsSVC",
    "VitsSVCConfig",
]