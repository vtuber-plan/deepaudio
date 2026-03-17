# coding=utf-8
"""
Processors for audio preprocessing in Soniq.

.. deprecated::
    Use `soniq.processing.extractors` instead. This module is kept for backwards compatibility.
"""

from soniq.processing.extractors import AcousticExtractor, PhoneExtractor

__all__ = ["AcousticExtractor", "PhoneExtractor"]