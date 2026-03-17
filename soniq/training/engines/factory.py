# coding=utf-8
"""
Engine Factory for Soniq Training.

引擎工厂，支持注册和创建不同引擎。
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Type

if TYPE_CHECKING:
    from .base import BaseEngine
    from ..base.context import EngineContext
    from ..base.callback import CallbackList


# 引擎注册表
ENGINE_REGISTRY: Dict[str, Type["BaseEngine"]] = {}


def register_engine(name: str) -> callable:
    """
    装饰器：注册引擎。

    Args:
        name: 引擎名称

    Returns:
        装饰器函数

    Example:
        ```python
        @register_engine("my_engine")
        class MyEngine(BaseEngine):
            ...
        ```
    """

    def decorator(cls: Type["BaseEngine"]) -> Type["BaseEngine"]:
        ENGINE_REGISTRY[name] = cls
        return cls

    return decorator


def create_engine(
    engine_type: str,
    ctx: "EngineContext",
    callbacks: Optional["CallbackList"] = None,
    **kwargs,
) -> "BaseEngine":
    """
    创建引擎实例。

    Args:
        engine_type: 引擎类型 ("accelerate" 或 "fabric")
        ctx: 训练上下文
        callbacks: 回调列表
        **kwargs: 传递给引擎的额外参数

    Returns:
        引擎实例

    Raises:
        ValueError: 未知的引擎类型

    Example:
        ```python
        ctx = EngineContext(seed=42)
        engine = create_engine("accelerate", ctx, mixed_precision="bf16")

        # 使用多个 logger
        engine = create_engine(
            "fabric",
            ctx,
            logger_types=["tensorboard", "wandb"],
            logger_configs={"wandb": {"project": "my-project"}}
        )
        ```
    """
    # 确保引擎已注册
    _ensure_engines_registered()

    if engine_type not in ENGINE_REGISTRY:
        available = list(ENGINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown engine type: {engine_type}. "
            f"Available engines: {available}"
        )

    engine_cls = ENGINE_REGISTRY[engine_type]
    return engine_cls(ctx=ctx, callbacks=callbacks, **kwargs)


def list_engines() -> list:
    """
    列出所有可用的引擎。

    Returns:
        引擎名称列表
    """
    _ensure_engines_registered()
    return list(ENGINE_REGISTRY.keys())


def get_engine_class(name: str) -> Type["BaseEngine"]:
    """
    获取引擎类。

    Args:
        name: 引擎名称

    Returns:
        引擎类

    Raises:
        ValueError: 未知的引擎类型
    """
    _ensure_engines_registered()

    if name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {name}")
    return ENGINE_REGISTRY[name]


def _ensure_engines_registered() -> None:
    """确保引擎已注册。"""
    if ENGINE_REGISTRY:
        return

    # 延迟导入以避免循环依赖
    try:
        from .accelerate_engine import AccelerateEngineAdapter
        register_engine("accelerate")(AccelerateEngineAdapter)
    except ImportError:
        pass

    try:
        from .fabric_engine import FabricEngineAdapter
        register_engine("fabric")(FabricEngineAdapter)
    except ImportError:
        pass


# 便捷函数
def get_available_engines() -> Dict[str, bool]:
    """
    获取可用引擎及其状态。

    Returns:
        引擎名称到可用状态的映射
    """
    engines = {}

    try:
        from accelerate import Accelerator
        engines["accelerate"] = True
    except ImportError:
        engines["accelerate"] = False

    try:
        from lightning import Fabric
        engines["fabric"] = True
    except ImportError:
        engines["fabric"] = False

    return engines