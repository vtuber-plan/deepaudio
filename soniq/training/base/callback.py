# coding=utf-8
"""
Callback System for Soniq Training.

统一的回调接口，支持训练全生命周期钩子。
"""

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import torch

if TYPE_CHECKING:
    from .context import EngineContext
    from ..trainer import Trainer


class Callback(ABC):
    """
    统一的回调基类。

    提供训练全生命周期的钩子方法，子类可按需覆盖。

    Example:
        ```python
        class MyCallback(Callback):
            def on_train_batch_end(self, trainer, ctx, batch, batch_idx, outputs):
                if ctx.iteration % 100 == 0:
                    print(f"Iteration {ctx.iteration}: loss = {outputs['loss']}")

        trainer = Trainer(callbacks=[MyCallback()])
        ```
    """

    # ===== 生命周期钩子 =====

    def on_fit_start(self, trainer: "Trainer", ctx: "EngineContext") -> None:
        """
        训练开始时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
        """
        pass

    def on_fit_end(self, trainer: "Trainer", ctx: "EngineContext") -> None:
        """
        训练结束时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
        """
        pass

    # ===== Epoch 钩子 =====

    def on_epoch_start(
        self, trainer: "Trainer", ctx: "EngineContext", epoch: int
    ) -> None:
        """
        每个 epoch 开始时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            epoch: 当前 epoch 索引
        """
        pass

    def on_epoch_end(
        self, trainer: "Trainer", ctx: "EngineContext", epoch: int
    ) -> None:
        """
        每个 epoch 结束时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            epoch: 当前 epoch 索引
        """
        pass

    # ===== 训练 Batch 钩子 =====

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        每个训练 batch 开始时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            batch: 当前批次数据
            batch_idx: 批次索引
        """
        pass

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
        outputs: Union[torch.Tensor, Dict],
    ) -> None:
        """
        每个训练 batch 结束时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            batch: 当前批次数据
            batch_idx: 批次索引
            outputs: 训练步骤输出
        """
        pass

    # ===== 验证 Batch 钩子 =====

    def on_validation_batch_start(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        每个验证 batch 开始时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            batch: 当前批次数据
            batch_idx: 批次索引
        """
        pass

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
        outputs: Union[torch.Tensor, Dict],
    ) -> None:
        """
        每个验证 batch 结束时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            batch: 当前批次数据
            batch_idx: 批次索引
            outputs: 验证步骤输出
        """
        pass

    # ===== 反向传播钩子 =====

    def on_before_backward(
        self, trainer: "Trainer", ctx: "EngineContext", loss: torch.Tensor
    ) -> None:
        """
        反向传播前调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            loss: 损失值
        """
        pass

    def on_after_backward(
        self, trainer: "Trainer", ctx: "EngineContext", loss: torch.Tensor
    ) -> None:
        """
        反向传播后调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            loss: 损失值
        """
        pass

    # ===== 优化器钩子 =====

    def on_before_optimizer_step(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        优化器步骤前调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            optimizer: 优化器实例
        """
        pass

    def on_after_optimizer_step(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        优化器步骤后调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            optimizer: 优化器实例
        """
        pass

    # ===== 验证钩子 =====

    def on_validation_start(
        self, trainer: "Trainer", ctx: "EngineContext"
    ) -> None:
        """
        验证开始时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
        """
        pass

    def on_validation_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        outputs: List[Dict],
        metrics: Dict[str, float],
    ) -> None:
        """
        验证结束时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            outputs: 所有验证步骤的输出列表
            metrics: 汇总的验证指标
        """
        pass

    # ===== 检查点钩子 =====

    def on_save_checkpoint(
        self, trainer: "Trainer", ctx: "EngineContext", checkpoint: Dict
    ) -> Dict:
        """
        保存检查点时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            checkpoint: 检查点字典

        Returns:
            要添加到检查点的额外状态
        """
        return {}

    def on_load_checkpoint(
        self, trainer: "Trainer", ctx: "EngineContext", checkpoint: Dict
    ) -> None:
        """
        加载检查点时调用。

        Args:
            trainer: 训练器实例
            ctx: 训练上下文
            checkpoint: 检查点字典
        """
        pass


class CallbackList:
    """
    管理多个回调的容器类。

    简化多个回调的批量调用。

    Example:
        ```python
        callbacks = CallbackList([Callback1(), Callback2()])
        callbacks.on_fit_start(trainer, ctx)  # 自动调用所有回调
        ```
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        初始化回调列表。

        Args:
            callbacks: 回调实例列表
        """
        self.callbacks: List[Callback] = callbacks or []

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, index: int) -> Callback:
        return self.callbacks[index]

    def append(self, callback: Callback) -> None:
        """添加回调。"""
        self.callbacks.append(callback)

    def extend(self, callbacks: List[Callback]) -> None:
        """扩展回调列表。"""
        self.callbacks.extend(callbacks)

    # ===== 生命周期钩子委托 =====

    def on_fit_start(self, trainer: "Trainer", ctx: "EngineContext") -> None:
        for cb in self.callbacks:
            cb.on_fit_start(trainer, ctx)

    def on_fit_end(self, trainer: "Trainer", ctx: "EngineContext") -> None:
        for cb in self.callbacks:
            cb.on_fit_end(trainer, ctx)

    # ===== Epoch 钩子委托 =====

    def on_epoch_start(
        self, trainer: "Trainer", ctx: "EngineContext", epoch: int
    ) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, ctx, epoch)

    def on_epoch_end(
        self, trainer: "Trainer", ctx: "EngineContext", epoch: int
    ) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, ctx, epoch)

    # ===== 训练 Batch 钩子委托 =====

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
    ) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_start(trainer, ctx, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
        outputs: Union[torch.Tensor, Dict],
    ) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_end(trainer, ctx, batch, batch_idx, outputs)

    # ===== 验证 Batch 钩子委托 =====

    def on_validation_batch_start(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
    ) -> None:
        for cb in self.callbacks:
            cb.on_validation_batch_start(trainer, ctx, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        batch: Any,
        batch_idx: int,
        outputs: Union[torch.Tensor, Dict],
    ) -> None:
        for cb in self.callbacks:
            cb.on_validation_batch_end(trainer, ctx, batch, batch_idx, outputs)

    # ===== 反向传播钩子委托 =====

    def on_before_backward(
        self, trainer: "Trainer", ctx: "EngineContext", loss: torch.Tensor
    ) -> None:
        for cb in self.callbacks:
            cb.on_before_backward(trainer, ctx, loss)

    def on_after_backward(
        self, trainer: "Trainer", ctx: "EngineContext", loss: torch.Tensor
    ) -> None:
        for cb in self.callbacks:
            cb.on_after_backward(trainer, ctx, loss)

    # ===== 优化器钩子委托 =====

    def on_before_optimizer_step(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for cb in self.callbacks:
            cb.on_before_optimizer_step(trainer, ctx, optimizer)

    def on_after_optimizer_step(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for cb in self.callbacks:
            cb.on_after_optimizer_step(trainer, ctx, optimizer)

    # ===== 验证钩子委托 =====

    def on_validation_start(
        self, trainer: "Trainer", ctx: "EngineContext"
    ) -> None:
        for cb in self.callbacks:
            cb.on_validation_start(trainer, ctx)

    def on_validation_end(
        self,
        trainer: "Trainer",
        ctx: "EngineContext",
        outputs: List[Dict],
        metrics: Dict[str, float],
    ) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(trainer, ctx, outputs, metrics)

    # ===== 检查点钩子委托 =====

    def on_save_checkpoint(
        self, trainer: "Trainer", ctx: "EngineContext", checkpoint: Dict
    ) -> Dict:
        extra = {}
        for cb in self.callbacks:
            cb_extra = cb.on_save_checkpoint(trainer, ctx, checkpoint)
            if cb_extra:
                extra.update(cb_extra)
        return extra

    def on_load_checkpoint(
        self, trainer: "Trainer", ctx: "EngineContext", checkpoint: Dict
    ) -> None:
        for cb in self.callbacks:
            cb.on_load_checkpoint(trainer, ctx, checkpoint)