# coding=utf-8
"""
Logger components for Fabric Engine.

提供 Fabric 引擎的 Logger 创建和管理功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...base.context import EngineContext


def create_fabric_loggers(
    ctx: EngineContext,
    logger_types: Optional[List[str]] = None,
    logger_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Any]:
    """
    为 Fabric 引擎创建 Logger 列表。

    Args:
        ctx: 训练上下文
        logger_types: Logger 类型列表，如 ["tensorboard", "wandb"]
        logger_configs: 各 Logger 的配置

    Returns:
        Fabric Logger 对象列表

    Example:
        ```python
        loggers = create_fabric_loggers(
            ctx,
            logger_types=["tensorboard", "wandb"],
            logger_configs={
                "wandb": {"project": "my-project", "name": "exp1"}
            }
        )
        fabric = Fabric(loggers=loggers)
        ```
    """
    logger_types = logger_types or ["tensorboard"]
    logger_configs = logger_configs or {}
    loggers = []

    for logger_type in logger_types:
        config = logger_configs.get(logger_type, {})

        if logger_type in ["tensorboard", "tb"]:
            logger = _create_tensorboard_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        elif logger_type == "wandb":
            logger = _create_wandb_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        elif logger_type == "mlflow":
            logger = _create_mlflow_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        elif logger_type == "csv":
            logger = _create_csv_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        elif logger_type == "neptune":
            logger = _create_neptune_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        elif logger_type == "aim":
            logger = _create_aim_logger(ctx, config)
            if logger is not None:
                loggers.append(logger)

        else:
            print(f"[Warning] Unknown logger type: {logger_type}")

    return loggers


def _create_tensorboard_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 TensorBoard Logger。"""
    try:
        from lightning.fabric.loggers import TensorBoardLogger

        return TensorBoardLogger(
            root_dir=config.get("root_dir", ctx.metrics_path),
            name=config.get("name", ""),
            version=config.get("version", None),
            default_hp_metric=config.get("default_hp_metric", True),
            prefix=config.get("prefix", ""),
        )
    except ImportError as e:
        print(f"[Warning] TensorBoardLogger not available: {e}")
        return None


def _create_wandb_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 WandB Logger。"""
    try:
        from lightning.fabric.loggers import WandbLogger

        return WandbLogger(
            name=config.get("name", ctx.experiment_name or "experiment"),
            project=config.get("project", "soniq"),
            entity=config.get("entity", None),
            offline=config.get("offline", False),
            version=config.get("version", None),
            save_dir=config.get("save_dir", str(ctx.log_path)),
            prefix=config.get("prefix", ""),
            experiment=config.get("experiment", None),
            log_model=config.get("log_model", False),
            **config.get("kwargs", {}),
        )
    except ImportError as e:
        print(f"[Warning] WandbLogger not available: {e}")
        return None


def _create_mlflow_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 MLflow Logger。"""
    try:
        from lightning.fabric.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=config.get("experiment_name", ctx.experiment_name or "default"),
            run_name=config.get("run_name", None),
            tracking_uri=config.get("tracking_uri", None),
            tags=config.get("tags", None),
            save_dir=config.get("save_dir", str(ctx.log_path)),
            prefix=config.get("prefix", ""),
            artifact_location=config.get("artifact_location", None),
            run_id=config.get("run_id", None),
        )
    except ImportError as e:
        print(f"[Warning] MLFlowLogger not available: {e}")
        return None


def _create_csv_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 CSV Logger。"""
    try:
        from lightning.fabric.loggers import CSVLogger

        return CSVLogger(
            root_dir=config.get("root_dir", ctx.log_path),
            name=config.get("name", ctx.experiment_name or "csv_logs"),
            version=config.get("version", None),
            prefix=config.get("prefix", ""),
            flush_logs_every_n_steps=config.get("flush_logs_every_n_steps", 100),
        )
    except ImportError as e:
        print(f"[Warning] CSVLogger not available: {e}")
        return None


def _create_neptune_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 Neptune Logger。"""
    try:
        from lightning.fabric.loggers import NeptuneLogger

        return NeptuneLogger(
            api_key=config.get("api_key", None),
            project=config.get("project", None),
            name=config.get("name", ctx.experiment_name or "experiment"),
            description=config.get("description", None),
            tags=config.get("tags", None),
            log_model_diagram=config.get("log_model_diagram", False),
            prefix=config.get("prefix", ""),
            **config.get("kwargs", {}),
        )
    except ImportError as e:
        print(f"[Warning] NeptuneLogger not available: {e}")
        return None


def _create_aim_logger(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 Aim Logger。"""
    try:
        from lightning.fabric.loggers import AimLogger

        return AimLogger(
            experiment=config.get("experiment", ctx.experiment_name or "experiment"),
            repo=config.get("repo", None),
            run_name=config.get("run_name", None),
            train_run_prefix=config.get("train_run_prefix", "train_"),
            test_run_prefix=config.get("test_run_prefix", "test_"),
            val_run_prefix=config.get("val_run_prefix", "val_"),
        )
    except ImportError as e:
        print(f"[Warning] AimLogger not available: {e}")
        return None


# Logger 映射表
FABRIC_LOGGER_MAP = {
    "tensorboard": _create_tensorboard_logger,
    "tb": _create_tensorboard_logger,
    "wandb": _create_wandb_logger,
    "mlflow": _create_mlflow_logger,
    "csv": _create_csv_logger,
    "neptune": _create_neptune_logger,
    "aim": _create_aim_logger,
}