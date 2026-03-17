# coding=utf-8
"""
Checkpoint Management for Accelerate Engine.

提供 Accelerate 引擎的检查点保存和加载功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from ...base.context import EngineContext


class AccelerateCheckpointManager:
    """
    Accelerate 检查点管理器。

    处理检查点的保存、加载和清理。

    Example:
        ```python
        manager = AccelerateCheckpointManager(accelerator, ctx)

        # 保存检查点
        manager.save(model, optimizer, scheduler, step=1000)

        # 加载检查点
        state = manager.load(path="./checkpoints/checkpoint_1000")

        # 清理旧检查点
        manager.cleanup(keep_last_n=5)
        ```
    """

    def __init__(self, accelerator, ctx: EngineContext):
        """
        初始化检查点管理器。

        Args:
            accelerator: Accelerator 实例
            ctx: 训练上下文
        """
        self.accelerator = accelerator
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

        # 等待所有进程同步
        self.accelerator.wait_for_everyone()

        # 使用 Accelerate 的 save_state
        self.accelerator.save_state(str(checkpoint_path))

        # 保存额外状态
        if self.accelerator.is_main_process:
            extra_state_path = checkpoint_path / "extra_state.pt"
            extra = {
                "ctx": self.ctx.to_dict(),
                "step": step,
            }
            if extra_state is not None:
                extra.update(extra_state)
            torch.save(extra, extra_state_path)

        return checkpoint_path

    def load(
        self,
        path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点。

        Args:
            path: 检查点路径
            model: 模型（未使用，Accelerate 自动处理）
            optimizer: 优化器（未使用）
            scheduler: 学习率调度器（未使用）

        Returns:
            加载的状态字典
        """
        path = Path(path)

        # 使用 Accelerate 的 load_state
        self.accelerator.load_state(str(path))

        # 加载额外状态
        state = {}
        extra_state_path = path / "extra_state.pt"
        if extra_state_path.exists():
            state = torch.load(extra_state_path, map_location="cpu")

            if "ctx" in state:
                self.ctx.update_from_checkpoint(state)
            elif "step" in state:
                self.ctx.iteration = state["step"]

        return state

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

        if not self.accelerator.is_local_main_process:
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

        self.accelerator.wait_for_everyone()

        # 手动解包模型
        unwrapped_model = model
        while hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module

        if self.accelerator.is_main_process:
            state_dict = unwrapped_model.state_dict()
            torch.save(state_dict, str(path / "model.pt"))

            # 保存 config
            if hasattr(unwrapped_model, 'config'):
                if hasattr(unwrapped_model.config, 'save_pretrained'):
                    unwrapped_model.config.save_pretrained(str(path))