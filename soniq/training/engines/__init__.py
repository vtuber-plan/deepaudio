# coding=utf-8
"""
Soniq Training Engines.

提供统一的训练引擎接口，支持 Accelerate 和 Fabric 后端。

Example:
    ```python
    from soniq.training.engines import create_engine, AccelerateEngineAdapter

    # 使用工厂函数
    engine = create_engine("accelerate", ctx)

    # 或直接使用引擎类
    engine = AccelerateEngineAdapter(ctx, mixed_precision="bf16")
    ```
"""

from .base import BaseEngine
from .factory import (
    create_engine,
    register_engine,
    list_engines,
    get_engine_class,
    get_available_engines,
    ENGINE_REGISTRY,
)


def __getattr__(name: str):
    """延迟导入引擎类。"""
    if name == "AccelerateEngineAdapter":
        from .accelerate_engine import AccelerateEngineAdapter
        return AccelerateEngineAdapter
    elif name == "FabricEngineAdapter":
        from .fabric_engine import FabricEngineAdapter
        return FabricEngineAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 基类
    "BaseEngine",
    # 工厂函数
    "create_engine",
    "register_engine",
    "list_engines",
    "get_engine_class",
    "get_available_engines",
    "ENGINE_REGISTRY",
    # 引擎适配器（延迟加载）
    "AccelerateEngineAdapter",
    "FabricEngineAdapter",
]