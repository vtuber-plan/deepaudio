# coding=utf-8
"""
Soniq Training Module.

提供统一的训练接口，支持 accelerate 和 fabric 引擎切换。

Features:
- Trainer: 统一训练器，支持多引擎切换
- Engines: 训练引擎 (Accelerate, Fabric)
- Loggers: 日志记录器 (TensorBoard, WandB, MLflow, etc.)
- Task Systems: 任务系统 (VocoderTaskSystem, VocosTaskSystem)
- Callbacks: 训练回调
- Loss Functions: 损失函数

Example:
    ```python
    from soniq.training import Trainer, BaseTaskSystem

    # 使用 Accelerate 引擎
    trainer = Trainer(
        engine="accelerate",
        run_path="./runs/exp1",
        max_steps=100000,
    )

    # 使用 Fabric 引擎
    trainer = Trainer(
        engine="fabric",
        run_path="./runs/exp1",
        max_epochs=100,
    )

    # 训练
    system = MyTaskSystem(config)
    trainer.fit(system, train_dataloader, val_dataloader)
    ```
"""

# 核心训练器
from .trainer import Trainer

# 引擎
from .engines import (
    BaseEngine,
    AccelerateEngineAdapter,
    FabricEngineAdapter,
    create_engine,
    list_engines,
    get_available_engines,
)

# 日志记录器
from .loggers import (
    BaseLogger,
    LoggerRegistry,
    TensorBoardLoggerAdapter,
    CompositeLogger,
)

# 基类
from .base.system import BaseTaskSystem
from .base.outputs import StepOutput, EvalOutput, InferOutput
from .base.context import EngineContext
from .base.callback import Callback, CallbackList

# Vocoder 任务系统
from .vocoder import (
    VocoderTaskSystem,
    VocoderConfig,
    VocoderDataset,
    VocoderCollator,
    VocosTaskSystem,
    VocosConfig,
)

__all__ = [
    # 统一 Trainer
    "Trainer",
    # 引擎
    "BaseEngine",
    "AccelerateEngineAdapter",
    "FabricEngineAdapter",
    "create_engine",
    "list_engines",
    "get_available_engines",
    # 日志记录器
    "BaseLogger",
    "LoggerRegistry",
    "TensorBoardLoggerAdapter",
    "CompositeLogger",
    # 基类
    "BaseTaskSystem",
    "StepOutput",
    "EvalOutput",
    "InferOutput",
    "EngineContext",
    "Callback",
    "CallbackList",
    # Vocoder 任务系统
    "VocoderTaskSystem",
    "VocoderConfig",
    "VocoderDataset",
    "VocoderCollator",
    "VocosTaskSystem",
    "VocosConfig",
]