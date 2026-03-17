# coding=utf-8
"""
Datasets for Soniq.

.. deprecated::
    Use `soniq.data` instead. This module is kept for backwards compatibility.
"""

# Backwards compatibility - import from new location
from soniq.data import BaseDataset, BaseCollator, build_dataloader

__all__ = ["BaseDataset", "BaseCollator", "build_dataloader"]