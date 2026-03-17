# coding=utf-8
"""
Task systems for Soniq.

.. deprecated::
    Use `soniq.training` instead. This module is kept for backwards compatibility.

    For backwards compatibility, this module re-exports classes from soniq.training.
    New code should import directly from soniq.training.vocoder and soniq.training.base.
"""

# Don't import anything here to avoid circular imports
# Users should import from soniq.training instead

__all__ = [
    "BaseTaskSystem",
    "StepOutput",
    "VocoderTaskSystem",
    "VocoderConfig",
    "VocoderDataset",
    "VocoderCollator",
    "VocosTaskSystem",
    "VocosConfig",
]

def __getattr__(name):
    """Lazy import for backwards compatibility."""
    if name in __all__:
        if name in ["BaseTaskSystem", "StepOutput"]:
            from soniq.training.base.system import BaseTaskSystem, StepOutput
            return locals()[name]
        elif name in ["VocoderTaskSystem", "VocoderConfig"]:
            from soniq.training.vocoder.system import VocoderTaskSystem, VocoderConfig
            return locals()[name]
        elif name == "VocoderDataset":
            from soniq.training.vocoder.datasets import VocoderDataset
            return VocoderDataset
        elif name == "VocoderCollator":
            from soniq.training.vocoder.collators import VocoderCollator
            return VocoderCollator
        elif name in ["VocosTaskSystem", "VocosConfig"]:
            from soniq.training.vocoder.vocos_system import VocosTaskSystem, VocosConfig
            return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")