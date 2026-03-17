# coding=utf-8
"""
Tracker components for Accelerate Engine.

提供 Accelerate 引擎的 Tracker 创建和管理功能。
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...base.context import EngineContext


def create_accelerate_trackers(
    ctx: EngineContext,
    tracker_types: Optional[List[str]] = None,
    tracker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Any]:
    """
    为 Accelerate 引擎创建 Tracker 列表。

    Args:
        ctx: 训练上下文
        tracker_types: Tracker 类型列表，如 ["tensorboard", "wandb"]
        tracker_configs: 各 Tracker 的配置

    Returns:
        Accelerate Tracker 对象列表

    Example:
        ```python
        trackers = create_accelerate_trackers(
            ctx,
            tracker_types=["tensorboard", "wandb"],
            tracker_configs={
                "wandb": {"project": "my-project"}
            }
        )
        accelerator = Accelerator(log_with=trackers)
        ```
    """
    tracker_types = tracker_types or ["tensorboard"]
    tracker_configs = tracker_configs or {}
    trackers = []

    for tracker_type in tracker_types:
        config = tracker_configs.get(tracker_type, {})

        if tracker_type in ["tensorboard", "tb"]:
            tracker = _create_tensorboard_tracker(ctx, config)
            if tracker is not None:
                trackers.append(tracker)

        elif tracker_type == "wandb":
            tracker = _create_wandb_tracker(ctx, config)
            if tracker is not None:
                trackers.append(tracker)

        elif tracker_type == "mlflow":
            tracker = _create_mlflow_tracker(ctx, config)
            if tracker is not None:
                trackers.append(tracker)

        elif tracker_type == "aim":
            tracker = _create_aim_tracker(ctx, config)
            if tracker is not None:
                trackers.append(tracker)

        elif tracker_type == "comet":
            tracker = _create_comet_tracker(ctx, config)
            if tracker is not None:
                trackers.append(tracker)

        else:
            print(f"[Warning] Unknown tracker type: {tracker_type}")

    return trackers


def _create_tensorboard_tracker(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 TensorBoard Tracker。"""
    try:
        from accelerate.tracking import TensorBoardTracker

        run_name = config.get(
            "run_name",
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        logging_dir = config.get("logging_dir", ctx.metrics_path)

        tracker = TensorBoardTracker(
            run_name=run_name,
            logging_dir=logging_dir,
        )
        tracker.start()
        return tracker

    except ImportError as e:
        print(f"[Warning] TensorBoardTracker not available: {e}")
        return None


def _create_wandb_tracker(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 WandB Tracker。"""
    try:
        from accelerate.tracking import WandBTracker

        tracker = WandBTracker(
            run_name=config.get("run_name", None),
            _wandb={
                "project": config.get("project", "soniq"),
                "name": config.get("name", ctx.experiment_name or "experiment"),
                "entity": config.get("entity", None),
                "tags": config.get("tags", None),
                "notes": config.get("notes", None),
                "config": config.get("config", {}),
                "dir": config.get("dir", str(ctx.log_path)),
            },
        )
        tracker.start()
        return tracker

    except ImportError as e:
        print(f"[Warning] WandBTracker not available: {e}")
        return None


def _create_mlflow_tracker(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 MLflow Tracker。"""
    try:
        from accelerate.tracking import MLflowTracker

        tracker = MLflowTracker(
            run_name=config.get("run_name", None),
            _mlflow={
                "experiment_name": config.get(
                    "experiment_name",
                    ctx.experiment_name or "default"
                ),
                "run_name": config.get("run_name", None),
                "tracking_uri": config.get(
                    "tracking_uri",
                    str(ctx.log_path / "mlruns")
                ),
                "tags": config.get("tags", None),
            },
        )
        tracker.start()
        return tracker

    except ImportError as e:
        print(f"[Warning] MLflowTracker not available: {e}")
        return None


def _create_aim_tracker(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 Aim Tracker。"""
    try:
        from accelerate.tracking import AimTracker

        tracker = AimTracker(
            run_name=config.get("run_name", None),
            _aim={
                "repo": config.get("repo", None),
                "experiment": config.get(
                    "experiment",
                    ctx.experiment_name or "experiment"
                ),
            },
        )
        tracker.start()
        return tracker

    except ImportError as e:
        print(f"[Warning] AimTracker not available: {e}")
        return None


def _create_comet_tracker(ctx: EngineContext, config: Dict[str, Any]) -> Optional[Any]:
    """创建 Comet Tracker。"""
    try:
        from accelerate.tracking import CometMLTracker

        tracker = CometMLTracker(
            run_name=config.get("run_name", None),
            _comet={
                "project_name": config.get("project", "soniq"),
                "experiment_name": config.get(
                    "experiment_name",
                    ctx.experiment_name or "experiment"
                ),
                "api_key": config.get("api_key", None),
            },
        )
        tracker.start()
        return tracker

    except ImportError as e:
        print(f"[Warning] CometMLTracker not available: {e}")
        return None


# Tracker 映射表
ACCELERATE_TRACKER_MAP = {
    "tensorboard": _create_tensorboard_tracker,
    "tb": _create_tensorboard_tracker,
    "wandb": _create_wandb_tracker,
    "mlflow": _create_mlflow_tracker,
    "aim": _create_aim_tracker,
    "comet": _create_comet_tracker,
}