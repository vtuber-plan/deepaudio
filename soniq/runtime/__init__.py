# coding=utf-8
"""Soniq runtime module."""

from soniq.runtime.engine import FabricEngine

__all__ = ["FabricEngine", "FabricTrainer"]


# Alias FabricEngine as FabricTrainer for compatibility
FabricTrainer = FabricEngine

