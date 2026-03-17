# coding=utf-8
"""
VitsSVC: Singing Voice Conversion based on VITS.

This module provides VitsSVC model for singing voice conversion using
content features (ContentVec, Whisper, etc.) and F0 as input.
"""

from .configuration_vitssvc import VitsSVCConfig
from .modeling_vitssvc import VitsSVC
from .vitssvc_components import (
    ConditionEncoder,
    MelodyEncoder,
    SpeakerEncoder,
    ContentFeatureEncoder,
    VitsSVCContentEncoder,
    f0_to_coarse,
)

__all__ = [
    "VitsSVCConfig",
    "VitsSVC",
    "ConditionEncoder",
    "MelodyEncoder",
    "SpeakerEncoder",
    "ContentFeatureEncoder",
    "VitsSVCContentEncoder",
    "f0_to_coarse",
]