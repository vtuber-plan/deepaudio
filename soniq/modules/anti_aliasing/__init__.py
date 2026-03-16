# coding=utf-8
"""Anti-aliasing modules for Soniq."""

from soniq.modules.anti_aliasing.filter import (
    UpSample1d,
    DownSample1d,
    LowPassFilter1d,
    Activation1d,
    kaiser_sinc_filter1d,
)

__all__ = [
    "UpSample1d",
    "DownSample1d",
    "LowPassFilter1d",
    "Activation1d",
    "kaiser_sinc_filter1d",
]
