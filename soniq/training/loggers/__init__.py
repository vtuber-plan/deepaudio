# coding=utf-8
"""
Logger abstraction layer for Soniq Training.

提供统一的日志记录接口，支持多种后端（TensorBoard, WandB, MLflow 等）。
"""

from .base import BaseLogger, LoggerRegistry
from .tensorboard import TensorBoardLoggerAdapter
from .composite import CompositeLogger

__all__ = [
    "BaseLogger",
    "LoggerRegistry",
    "TensorBoardLoggerAdapter",
    "CompositeLogger",
]