# coding=utf-8
"""
Accelerate Engine Adapter for Soniq Training.

基于 HuggingFace Accelerate 的引擎适配器。
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
    from accelerate.tracking import GeneralTracker
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from .base import BaseEngine
from ..base.context import EngineContext
from ..base.callback import CallbackList


class AccelerateEngineAdapter(BaseEngine):
    """
    Accelerate 引擎适配器。

    封装 HuggingFace Accelerator，实现 BaseEngine 接口。

    Example:
        ```python
        ctx = EngineContext(seed=42, num_iterations=10000)
        engine = AccelerateEngineAdapter(ctx, mixed_precision="bf16")
        model, optimizer, train_dl, val_dl = engine.setup(
            model, optimizer, train_dataloader, val_dataloader
        )
        ```
    """

    engine_name = "accelerate"

    def __init__(
        self,
        ctx: EngineContext,
        callbacks: Optional[CallbackList] = None,
        accelerator: Optional["Accelerator"] = None,
        **accelerate_kwargs,
    ):
        """
        初始化 Accelerate 引擎适配器。

        Args:
            ctx: 训练上下文
            callbacks: 回调列表
            accelerator: 已有的 Accelerator 实例（如果提供，accelerate_kwargs 将被忽略）
            **accelerate_kwargs: Accelerator 初始化参数
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "Accelerate is not available. Please install it with: pip install accelerate"
            )

        super().__init__(ctx, callbacks)

        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # 设置默认配置
            accelerate_kwargs.setdefault(
                "gradient_accumulation_steps", ctx.gradient_accumulation_steps
            )
            accelerate_kwargs.setdefault("log_with", None)
            self.accelerator = Accelerator(**accelerate_kwargs)

        # 更新 context 中的分布式信息
        ctx.local_rank = self.accelerator.local_process_index
        ctx.rank = self.accelerator.process_index
        ctx.world_size = self.accelerator.num_processes
        ctx.gradient_accumulation_steps = self.accelerator.gradient_accumulation_steps

        # 设置 TensorBoard tracker
        self._setup_tracker(ctx)

    def _setup_tracker(self, ctx: EngineContext) -> None:
        """设置 TensorBoard tracker。"""
        try:
            from trainer.core.tracker import TensorBoardTracker
            import datetime

            start_time = datetime.datetime.fromtimestamp(ctx.start_timestamp)
            run_name = start_time.strftime("%Y-%m-%d-%H-%M-%S")
            self.accelerator.trackers = [
                TensorBoardTracker(run_name, ctx.metrics_path)
            ]
            # 启动 tracker
            for tracker in self.accelerator.trackers:
                tracker.start()
        except ImportError:
            # 如果没有 trainer.core，使用简单的日志
            pass

    def setup(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, Optional[DataLoader]]:
        """
        使用 Accelerator 准备模型、优化器和数据加载器。

        Args:
            model: 模型
            optimizer: 优化器
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            scheduler: 学习率调度器

        Returns:
            (prepared_model, prepared_optimizer, prepared_train_dl, prepared_val_dl)
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 准备训练组件
        prepared_model, prepared_optimizer, prepared_train_dl = self.accelerator.prepare(
            model, optimizer, train_dataloader
        )

        prepared_val_dl = None
        if val_dataloader is not None:
            prepared_val_dl = self.accelerator.prepare(val_dataloader)

        # 注册检查点
        self.accelerator.register_for_checkpointing(self)

        # 保存引用
        self._model = prepared_model
        self._optimizer = prepared_optimizer
        self._scheduler = scheduler
        self._train_dataloader = prepared_train_dl
        self._val_dataloader = prepared_val_dl

        if scheduler is not None:
            self._scheduler = self.accelerator.prepare(scheduler)

        return prepared_model, prepared_optimizer, prepared_train_dl, prepared_val_dl

    def backward(self, loss: torch.Tensor) -> None:
        """执行反向传播。"""
        self.accelerator.backward(loss)

    def step(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """执行优化器步骤。"""
        opt = optimizer or self._optimizer
        opt.step()
        opt.zero_grad()

    def clip_gradients(
        self,
        model: torch.nn.Module,
        clip_val: float,
        clip_algorithm: str = "norm",
    ) -> Optional[float]:
        """
        裁剪梯度。

        Args:
            model: 模型
            clip_val: 裁剪值
            clip_algorithm: 裁剪算法 ("norm" 或 "value")

        Returns:
            梯度范数（如果使用 norm 算法）
        """
        if clip_algorithm == "norm":
            return self.accelerator.clip_grad_norm_(
                model.parameters(), clip_val
            ).item()
        elif clip_algorithm == "value":
            self.accelerator.clip_grad_value_(model.parameters(), clip_val)
            return None
        else:
            raise ValueError(f"Unknown clip algorithm: {clip_algorithm}")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> None:
        """
        保存检查点。

        Args:
            path: 检查点保存路径
            state: 额外状态（会自动保存 ctx）
        """
        self.accelerator.wait_for_everyone()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(path), safe_serialization=False)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        加载检查点。

        Args:
            path: 检查点路径
            state: 要加载的状态字典（暂不使用）

        Returns:
            加载的状态（从 ctx 中获取）
        """
        self.accelerator.load_state(str(path))
        return {"iteration": self.ctx.iteration, "epoch": self.ctx.epoch}

    def save_model(self, model: torch.nn.Module, path: Union[str, Path]) -> None:
        """
        保存模型权重。

        Args:
            model: 模型
            path: 保存路径
        """
        self.accelerator.wait_for_everyone()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_model(model, str(path), safe_serialization=False)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录指标。

        Args:
            metrics: 指标字典
            step: 步数
        """
        self.accelerator.log(metrics, step=step)

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """
        记录音频。

        Args:
            name: 音频名称
            audio: 音频张量
            sample_rate: 采样率
            step: 步数
        """
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, "log_audios"):
                tracker.log_audios(
                    {name: audio.cpu().numpy()}, step=step, sample_rate=sample_rate
                )

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """
        记录图像。

        Args:
            name: 图像名称
            image: 图像张量
            step: 步数
        """
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, "log_images"):
                tracker.log_images({name: image.cpu().numpy()}, step=step)

    def is_main_process(self) -> bool:
        """当前是否为主进程。"""
        return self.accelerator.is_main_process

    def is_local_main_process(self) -> bool:
        """当前是否为本地主进程。"""
        return self.accelerator.is_local_main_process

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从所有进程收集张量。

        Args:
            tensor: 输入张量

        Returns:
            收集后的张量
        """
        return self.accelerator.gather(tensor)

    def all_reduce(
        self, tensor: torch.Tensor, op: str = "mean"
    ) -> torch.Tensor:
        """
        跨进程归约张量。

        Args:
            tensor: 输入张量
            op: 归约操作 ("mean" 或 "sum")

        Returns:
            归约后的张量
        """
        reduction = "mean" if op == "mean" else "sum"
        return self.accelerator.reduce(tensor, reduction=reduction)

    def barrier(self) -> None:
        """同步所有进程。"""
        self.accelerator.wait_for_everyone()

    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        self.accelerator.wait_for_everyone()

    @property
    def device(self) -> torch.device:
        """获取当前设备。"""
        return self.accelerator.device

    @property
    def precision(self) -> str:
        """获取当前精度模式。"""
        return str(self.accelerator.mixed_precision)

    def autocast(self, enabled: bool = True):
        """
        获取自动混合精度上下文管理器。

        Args:
            enabled: 是否启用

        Returns:
            上下文管理器
        """
        return self.accelerator.autocast(enabled=enabled)

    @property
    def gradient_accumulation_steps(self) -> int:
        """获取梯度累积步数。"""
        return self.accelerator.gradient_accumulation_steps

    def is_gradient_accumulation_boundary(self) -> bool:
        """
        当前是否为梯度累积边界。

        Returns:
            是否为梯度累积边界
        """
        return self.accelerator.sync_gradients

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        获取未包装的原始模型。

        Args:
            model: 包装后的模型

        Returns:
            原始模型
        """
        return self.accelerator.unwrap_model(model)

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        """
        禁用梯度同步的上下文管理器。

        Args:
            model: 模型

        Yields:
            None
        """
        with self.accelerator.no_sync(model):
            yield

    def end_training(self) -> None:
        """结束训练，清理资源。"""
        self.accelerator.end_training()