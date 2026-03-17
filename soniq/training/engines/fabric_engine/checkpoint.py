# coding=utf-8
"""
Checkpoint Management for Fabric Engine.

提供 Fabric 引擎的检查点保存和加载功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from ...base.context import EngineContext


class FabricCheckpointManager:
    """
    Fabric 检查点管理器。

    处理检查点的保存、加载和清理。

    Example:
        ```python
        manager = FabricCheckpointManager(fabric, ctx)

        # 保存检查点
        manager.save(model, optimizer, scheduler, step=1000)

        # 加载检查点
        state = manager.load(path="./checkpoints/checkpoint_1000")

        # 清理旧检查点
        manager.cleanup(keep_last_n=5)
        ```
    """

    def __init__(self, fabric, ctx: EngineContext):
        """
        初始化检查点管理器。

        Args:
            fabric: Lightning Fabric 实例
            ctx: 训练上下文
        """
        self.fabric = fabric
        self.ctx = ctx
        self._checkpoint_dir = ctx.ckpt_save_path

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: Optional[int] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        保存检查点。

        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            step: 步数（默认使用 ctx.iteration）
            extra_state: 额外状态

        Returns:
            检查点路径
        """
        step = step or self.ctx.iteration
        checkpoint_path = self._checkpoint_dir / f"checkpoint_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 构建状态字典
        state = {
            "model": model,
            "epoch": self.ctx.epoch,
            "iteration": step,
            "ctx": self.ctx.to_dict(),
        }

        if optimizer is not None:
            state["optimizer"] = optimizer

        if scheduler is not None:
            state["scheduler"] = scheduler

        if extra_state is not None:
            state.update(extra_state)

        # 等待所有进程同步
        self.fabric.barrier()

        # 保存检查点
        self.fabric.save(str(checkpoint_path / "state.ckpt"), state)

        return checkpoint_path

    def load(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点。

        Args:
            path: 检查点路径
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器

        Returns:
            加载的状态字典
        """
        path = Path(path)
        ckpt_file = path / "state.ckpt" if path.is_dir() else path

        # 构建加载状态
        load_state = {"model": model}
        if optimizer is not None:
            load_state["optimizer"] = optimizer
        if scheduler is not None:
            load_state["scheduler"] = scheduler

        # 加载检查点
        self.fabric.load(str(ckpt_file), load_state)

        # 更新上下文
        if "ctx" in load_state:
            self.ctx.update_from_checkpoint(load_state)
        elif "iteration" in load_state:
            self.ctx.iteration = load_state["iteration"]
            self.ctx.epoch = load_state.get("epoch", 0)

        return load_state

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点路径。"""
        if not self._checkpoint_dir.exists():
            return None

        checkpoints = sorted(self._checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None

        return checkpoints[-1]

    def list_checkpoints(self) -> List[Path]:
        """列出所有检查点。"""
        if not self._checkpoint_dir.exists():
            return []

        return sorted(self._checkpoint_dir.glob("checkpoint_*"))

    def cleanup(self, keep_last_n: int = 5) -> None:
        """
        清理旧检查点，只保留最近 N 个。

        Args:
            keep_last_n: 保留的检查点数量
        """
        import shutil

        if not self.fabric.local_rank == 0:
            return

        checkpoints = self.list_checkpoints()
        while len(checkpoints) > keep_last_n:
            old_ckpt = checkpoints.pop(0)
            shutil.rmtree(old_ckpt)
            print(f"[Checkpoint] Removed: {old_ckpt}")

    def save_model_only(
        self,
        model: torch.nn.Module,
        path: Union[str, Path],
    ) -> None:
        """
        只保存模型权重（不含优化器等）。

        Args:
            model: 模型
            path: 保存路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.fabric.barrier()
        self.fabric.save(str(path / "model.ckpt"), {"model": model})