# coding=utf-8
"""
Unified Trainer for Soniq Training.

统一的训练器，支持 accelerate 和 fabric 引擎切换。
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader

from .base.context import EngineContext
from .base.callback import Callback, CallbackList
from .base.system import BaseTaskSystem
from .engines.base import BaseEngine
from .engines.factory import create_engine


class Trainer:
    """
    统一的训练器，支持 accelerate 和 fabric 引擎切换。

    Example:
        ```python
        # 使用 Accelerate
        trainer = Trainer(
            engine="accelerate",
            run_path="./runs/exp1",
            max_steps=100000,
        )

        # 使用 Fabric
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

    def __init__(
        self,
        # 引擎配置
        engine: str = "accelerate",
        run_path: Union[str, Path] = "./runs/default",
        # 训练配置
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = 1.0,
        gradient_clip_algorithm: str = "norm",
        # 日志配置
        log_interval_steps: int = 200,
        val_interval_steps: int = 2000,
        # 检查点配置
        save_interval_steps: Optional[int] = None,
        save_last_n: int = 2,
        export_interval_steps: Optional[int] = None,
        # 其他配置
        seed: int = 3407,
        debug: bool = False,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[logging.Logger] = None,
        # 引擎特定参数
        **engine_kwargs,
    ):
        """
        初始化 Trainer。

        Args:
            engine: 引擎类型 ("accelerate" 或 "fabric")
            run_path: 运行目录
            max_steps: 最大训练步数（优先于 max_epochs）
            max_epochs: 最大 epoch 数
            gradient_accumulation_steps: 梯度累积步数
            gradient_clip_val: 梯度裁剪值
            gradient_clip_algorithm: 梯度裁剪算法 ("norm" 或 "value")
            log_interval_steps: 日志记录间隔
            val_interval_steps: 验证间隔
            save_interval_steps: 检查点保存间隔
            save_last_n: 保留最近 N 个检查点
            export_interval_steps: 权重导出间隔
            seed: 随机种子
            debug: 调试模式（会调整训练参数以快速验证）
            callbacks: 回调列表
            logger: 日志记录器
            **engine_kwargs: 引擎特定参数
        """
        # 调试模式调整
        if debug:
            max_steps = max_steps or 10
            log_interval_steps = 1
            val_interval_steps = 2
            save_interval_steps = 3
            export_interval_steps = 3

        # 创建上下文
        self.ctx = EngineContext(
            seed=seed,
            debug_mode=debug,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_iterations=max_steps or 100_000,
            max_epochs=max_epochs,
            clip_grad_norm=gradient_clip_val or 1.0,
            gradient_clip_algorithm=gradient_clip_algorithm,
            log_interval_steps=log_interval_steps,
            val_interval_steps=val_interval_steps,
            save_interval_steps=save_interval_steps
            or max(1000, (max_steps or 100_000) // 10),
            save_last_n=save_last_n,
            export_interval_steps=export_interval_steps,
        )
        self.ctx.setup_paths(run_path)

        # 设置回调
        self.callbacks = CallbackList(callbacks or [])

        # 创建引擎
        self.engine: BaseEngine = create_engine(
            engine_type=engine,
            ctx=self.ctx,
            callbacks=self.callbacks,
            **engine_kwargs,
        )

        self.logger = logger or logging.getLogger(__name__)

        # 运行时状态
        self._system: Optional[BaseTaskSystem] = None
        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._is_setup = False
        self._engine_type = engine

    def setup(
        self,
        system: BaseTaskSystem,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        """
        设置训练系统。

        Args:
            system: 任务系统，继承自 BaseTaskSystem
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
        """
        self._system = system

        # 获取优化器配置
        optimizers = system.configure_optimizers()
        if isinstance(optimizers, dict):
            optimizer = optimizers.get("optimizer", optimizers.get("generator"))
            scheduler = optimizers.get("scheduler") or optimizers.get("lr_scheduler")
        elif isinstance(optimizers, (list, tuple)):
            optimizer = optimizers[0]
            scheduler = optimizers[1] if len(optimizers) > 1 else None
        else:
            optimizer = optimizers
            scheduler = None

        # 使用引擎 setup
        model, optimizer, train_dl, val_dl = self.engine.setup(
            model=system,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            scheduler=scheduler,
        )

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._train_dataloader = train_dl
        self._val_dataloader = val_dl
        self._is_setup = True

    def fit(
        self,
        system: BaseTaskSystem,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        训练模型。

        Args:
            system: 任务系统
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            ckpt_path: 恢复训练的检查点路径
        """
        if not self._is_setup:
            self.setup(system, train_dataloader, val_dataloader)

        # 恢复训练
        if ckpt_path is not None:
            self._resume_from_checkpoint(ckpt_path)

        # 开始训练
        self.callbacks.on_fit_start(self, self.ctx)
        system.on_train_start()

        try:
            if self.ctx.max_epochs is not None:
                self._train_by_epochs()
            else:
                self._train_by_steps()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint()
        finally:
            self.callbacks.on_fit_end(self, self.ctx)
            system.on_train_end()
            self.engine.wait_for_everyone()

    def _train_by_steps(self) -> None:
        """按步数训练。"""
        sampler = getattr(self._train_dataloader, "sampler", None)

        while self.ctx.iteration < self.ctx.num_iterations:
            self.callbacks.on_epoch_start(self, self.ctx, self.ctx.epoch)

            # 设置 sampler epoch（用于分布式）
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.ctx.epoch)

            for batch_idx, batch in enumerate(self._train_dataloader):
                if self.ctx.iteration >= self.ctx.num_iterations:
                    break

                # 训练步骤
                self._train_step(batch, batch_idx)

                # 验证
                if self.ctx.need_to_validate and self._val_dataloader is not None:
                    self._validate()

                # 保存检查点
                if self.ctx.need_to_save:
                    self._save_checkpoint()

                # 导出权重
                if self.ctx.need_to_export:
                    self._export_weights()

                self.ctx.iteration += 1

            self.callbacks.on_epoch_end(self, self.ctx, self.ctx.epoch)
            self.ctx.epoch += 1

    def _train_by_epochs(self) -> None:
        """按 epoch 训练。"""
        sampler = getattr(self._train_dataloader, "sampler", None)

        while self.ctx.epoch < self.ctx.max_epochs:
            self.callbacks.on_epoch_start(self, self.ctx, self.ctx.epoch)

            # 设置 sampler epoch
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.ctx.epoch)

            for batch_idx, batch in enumerate(self._train_dataloader):
                self._train_step(batch, batch_idx)
                self.ctx.iteration += 1

            # Epoch 结束验证
            if self._val_dataloader is not None:
                self._validate()

            # Epoch 结束保存
            self._save_checkpoint()

            self.callbacks.on_epoch_end(self, self.ctx, self.ctx.epoch)
            self.ctx.epoch += 1

    def _train_step(self, batch: Any, batch_idx: int) -> None:
        """
        执行一个训练步骤。

        Args:
            batch: 输入批次
            batch_idx: 批次索引
        """
        self._model.train()
        self.ctx.train_flag = True

        self.callbacks.on_train_batch_start(self, self.ctx, batch, batch_idx)

        with self.engine.autocast():
            output = self._system.training_step(batch, batch_idx)

        # 提取 loss
        loss = self._extract_loss(output)

        # 反向传播
        self.callbacks.on_before_backward(self, self.ctx, loss)
        self.engine.backward(loss)
        self.callbacks.on_after_backward(self, self.ctx, loss)

        # 梯度裁剪和优化器步骤
        if self.engine.is_gradient_accumulation_boundary():
            if self.ctx.clip_grad_norm is not None:
                grad_norm = self.engine.clip_gradients(
                    self._model,
                    self.ctx.clip_grad_norm,
                    self.ctx.gradient_clip_algorithm,
                )
                if self.ctx.need_to_log:
                    self.engine.log({"grad_norm": grad_norm}, step=self.ctx.iteration)

            self.callbacks.on_before_optimizer_step(self, self.ctx, self._optimizer)
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.callbacks.on_after_optimizer_step(self, self.ctx, self._optimizer)

            # 学习率调度
            if self._scheduler is not None:
                self._scheduler.step()

            # 记录指标
            if self.ctx.need_to_log:
                self._log_metrics(output, prefix="train")

        self.callbacks.on_train_batch_end(self, self.ctx, batch, batch_idx, output)

    def _extract_loss(self, output: Any) -> torch.Tensor:
        """
        从输出中提取 loss。

        Args:
            output: 训练步骤输出

        Returns:
            损失值
        """
        if hasattr(output, "loss"):
            return output.loss
        elif isinstance(output, dict):
            if "loss" not in output:
                raise ValueError(f"Output dict must have 'loss' key: {output.keys()}")
            return output["loss"]
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise TypeError(f"Unsupported output type: {type(output)}")

    @torch.no_grad()
    def _validate(self) -> None:
        """执行验证。"""
        self._model.eval()
        self.ctx.train_flag = False

        self.callbacks.on_validation_start(self, self.ctx)

        outputs = []
        for batch_idx, batch in enumerate(self._val_dataloader):
            self.callbacks.on_validation_batch_start(
                self, self.ctx, batch, batch_idx
            )

            output = self._system.validation_step(batch, batch_idx)
            outputs.append(output)

            self.callbacks.on_validation_batch_end(
                self, self.ctx, batch, batch_idx, output
            )

        # 计算验证指标
        metrics = self._aggregate_validation_metrics(outputs)
        self._log_metrics(metrics, prefix="val")

        self.callbacks.on_validation_end(self, self.ctx, outputs, metrics)

    def _aggregate_validation_metrics(
        self, outputs: List[Any]
    ) -> Dict[str, float]:
        """
        聚合验证指标。

        Args:
            outputs: 所有验证步骤的输出

        Returns:
            汇总的指标
        """
        metrics = {}

        # 收集所有 loss
        losses = []
        for output in outputs:
            loss = self._extract_loss(output)
            if isinstance(loss, torch.Tensor):
                losses.append(loss.item())
            else:
                losses.append(loss)

        if losses:
            metrics["loss"] = sum(losses) / len(losses)

        # 收集其他指标
        for output in outputs:
            if hasattr(output, "metrics") and output.metrics:
                for k, v in output.metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)

        # 平均指标
        for k, v in list(metrics.items()):
            if isinstance(v, list):
                metrics[k] = sum(v) / len(v)

        return metrics

    def _log_metrics(self, output: Any, prefix: str = "train") -> None:
        """
        记录指标。

        Args:
            output: 输出
            prefix: 指标前缀
        """
        metrics = {}

        if hasattr(output, "metrics") and output.metrics:
            metrics = {f"{prefix}/{k}": v for k, v in output.metrics.items()}
        elif isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, (int, float)) or (
                    isinstance(v, torch.Tensor) and v.numel() == 1
                ):
                    metrics[f"{prefix}/{k}"] = v.item() if isinstance(v, torch.Tensor) else v

        loss = self._extract_loss(output)
        if f"{prefix}/loss" not in metrics:
            metrics[f"{prefix}/loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss

        if self._scheduler is not None and prefix == "train":
            metrics[f"{prefix}/lr"] = self._scheduler.get_last_lr()[0]

        self.engine.log(metrics, step=self.ctx.iteration)

        # 打印日志
        metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.engine.print(f"[Step {self.ctx.iteration}] {metric_str}")

    def _save_checkpoint(self) -> None:
        """保存检查点。"""
        if self.ctx.iteration == 0:
            return

        checkpoint = {
            "ctx": self.ctx.to_dict(),
            "epoch": self.ctx.epoch,
            "iteration": self.ctx.iteration,
        }

        # 回调钩子
        extra = self.callbacks.on_save_checkpoint(self, self.ctx, checkpoint)
        checkpoint.update(extra)

        path = self.ctx.ckpt_save_path / f"checkpoint_{self.ctx.iteration}"
        self.engine.save_checkpoint(path, checkpoint)

        self.logger.info(f"Checkpoint saved to {path}")

        # 清理旧检查点
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """清理旧检查点。"""
        if not self.engine.is_local_main_process():
            return

        checkpoints = sorted(self.ctx.ckpt_save_path.glob("checkpoint_*"))
        while len(checkpoints) > self.ctx.save_last_n:
            old_ckpt = checkpoints.pop(0)
            shutil.rmtree(old_ckpt)
            self.logger.info(f"Removed old checkpoint: {old_ckpt}")

    def _resume_from_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        """
        从检查点恢复训练。

        Args:
            ckpt_path: 检查点路径
        """
        state = self.engine.load_checkpoint(ckpt_path)
        self.ctx.update_from_checkpoint(state)
        self.callbacks.on_load_checkpoint(self, self.ctx, state)
        self.ctx.iteration += 1  # 跳过已保存的步骤
        self.logger.info(
            f"Resumed from checkpoint: iteration={self.ctx.iteration}, epoch={self.ctx.epoch}"
        )

    def _export_weights(self) -> None:
        """导出模型权重。"""
        path = self.ctx.weights_save_path / f"step_{self.ctx.iteration}"
        self.engine.save_model(self._model, path)
        self.logger.info(f"Model weights exported to {path}")

    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        保存模型权重。

        Args:
            path: 保存路径，默认为 weights_save_path/step_{iteration}
        """
        if path is None:
            path = self.ctx.weights_save_path / f"step_{self.ctx.iteration}"
        self.engine.save_model(self._model, path)

    def test(
        self,
        system: BaseTaskSystem,
        test_dataloader: DataLoader,
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        测试模型。

        Args:
            system: 任务系统
            test_dataloader: 测试数据加载器
            ckpt_path: 检查点路径

        Returns:
            所有测试输出的列表
        """
        if ckpt_path is not None:
            self.engine.load_checkpoint(ckpt_path, {"model": system})

        system.eval()
        all_outputs = []

        with torch.no_grad():
            for batch in test_dataloader:
                output = system.inference_step(batch)
                all_outputs.append(output)

        return all_outputs

    @property
    def global_step(self) -> int:
        """获取全局步数。"""
        return self.ctx.iteration

    @property
    def current_epoch(self) -> int:
        """获取当前 epoch。"""
        return self.ctx.epoch

    @property
    def num_gpus(self) -> int:
        """获取 GPU 数量。"""
        return self.ctx.world_size