# coding=utf-8
"""
Soniq Training Base Module.

提供训练的基础类和接口。
"""

from .system import BaseTaskSystem, StepOutput
from .outputs import StepOutput, EvalOutput, InferOutput
from .context import EngineContext
from .callback import Callback, CallbackList

__all__ = [
    "BaseTaskSystem",
    "StepOutput",
    "EvalOutput",
    "InferOutput",
    "EngineContext",
    "Callback",
    "CallbackList",
]