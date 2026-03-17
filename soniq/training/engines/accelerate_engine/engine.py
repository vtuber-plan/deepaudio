# coding=utf-8
"""
Accelerate Engine Adapter for Soniq Training.

基于 HuggingFace Accelerate 的训练引擎主类。
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
    from accelerate.tracking import TensorBoardTracker
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from ..base import BaseEngine
from ...base.context import EngineContext
from ...base.callback import CallbackList
from .trackers import create_accelerate_trackers
from .checkpoint import AccelerateCheckpointManager
from .distributed import AccelerateDistributed


class AccelerateEngineAdapter(BaseEngine):
    """
    Accelerate 引擎适配器。

    封装 HuggingFace Accelerator，实现 BaseEngine 接口。

    Example:
        ```python
        ctx = EngineContext(seed=42, num_iterations=10000)
        engine = AccelerateEngineAdapter(
            ctx,
            mixed_precision="bf16",
            tracker_types=["tensorboard", "wandb"],
        )
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
        # Tracker 配置
        tracker_types: Optional[List[str]] = None,
        tracker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Accelerate 配置
        **accelerate_kwargs,
    ):
        """
        初始化 Accelerate 引擎适配器。

        Args:
            ctx: 训练上下文
            callbacks: 回调列表
            accelerator: 已有的 Accelerator 实例
            tracker_types: Tracker 类型列表
            tracker_configs: 各 Tracker 的配置
            **accelerate_kwargs: Accelerator 初始化参数
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "Accelerate is not available. "
                "Please install it with: pip install accelerate"
            )

        super().__init__(ctx, callbacks)

        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # 过滤并转换 Accelerator 支持的参数
            accelerator_params = self._filter_accelerator_params(
                ctx, accelerate_kwargs, tracker_types, tracker_configs
            )
            self.accelerator = Accelerator(**accelerator_params)

        # 更新 context 中的分布式信息
        ctx.local_rank = self.accelerator.local_process_index
        ctx.rank = self.accelerator.process_index
        ctx.world_size = self.accelerator.num_processes
        ctx.gradient_accumulation_steps = self.accelerator.gradient_accumulation_steps

        # 初始化组件
        self._checkpoint_manager: Optional[AccelerateCheckpointManager] = None
        self._distributed: Optional[AccelerateDistributed] = None
        self._tensorboard_tracker = None

        # 设置 TensorBoard tracker（如果需要）
        self._setup_tensorboard(ctx, tracker_types)

    def _filter_accelerator_params(
        self,
        ctx: EngineContext,
        accelerate_kwargs: Dict[str, Any],
        tracker_types: Optional[List[str]],
        tracker_configs: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """过滤并转换 Accelerator 支持的参数。"""
        accelerator_params = {}

        # 处理 mixed_precision
        if "precision" in accelerate_kwargs:
            precision = accelerate_kwargs.pop("precision")
            if precision in ["16-mixed", "16"]:
                accelerator_params["mixed_precision"] = "fp16"
            elif precision in ["bf16", "bfloat16"]:
                accelerator_params["mixed_precision"] = "bf16"
            elif precision in ["32", "32-true"]:
                accelerator_params["mixed_precision"] = "no"
        elif "mixed_precision" in accelerate_kwargs:
            accelerator_params["mixed_precision"] = accelerate_kwargs.pop("mixed_precision")

        # 复制其他支持的参数
        supported_keys = [
            "gradient_accumulation_steps", "cpu", "device_placement",
            "split_batches", "dispatch_batches", "even_batches",
            "use_seedable_sampler", "step_scheduler_with_optimizer",
            "log_with", "project_dir", "project_config", "tracker_filter"
        ]
        for key in supported_keys:
            if key in accelerate_kwargs:
                accelerator_params[key] = accelerate_kwargs[key]

        # 设置默认值
        accelerator_params.setdefault("gradient_accumulation_steps", ctx.gradient_accumulation_steps)

        # 创建 trackers
        if tracker_types and "log_with" not in accelerator_params:
            trackers = create_accelerate_trackers(
                ctx,
                tracker_types=tracker_types,
                tracker_configs=tracker_configs,
            )
            if trackers:
                accelerator_params["log_with"] = trackers

        return accelerator_params

    def _setup_tensorboard(
        self,
        ctx: EngineContext,
        tracker_types: Optional[List[str]],
    ) -> None:
        """设置 TensorBoard tracker。"""
        import datetime

        # 如果没有指定 tracker 或指定了 tensorboard，创建本地 tracker
        if tracker_types is None or "tensorboard" in tracker_types or "tb" in tracker_types:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._tensorboard_tracker = TensorBoardTracker(
                run_name=run_name,
                logging_dir=ctx.metrics_path,
            )
            self._tensorboard_tracker.start()

            # 添加到 accelerator
            if self.accelerator.trackers is None:
                self.accelerator.trackers = []
            self.accelerator.trackers.append(self._tensorboard_tracker)

    # ==================== 核心方法 ====================

    def setup(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, Optional[DataLoader]]:
        """设置模型、优化器和数据加载器。"""
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 准备训练组件
        prepared_model, prepared_optimizer, prepared_train_dl = self.accelerator.prepare(
            model, optimizer, train_dataloader
        )

        prepared_val_dl = None
        if val_dataloader is not None:
            prepared_val_dl = self.accelerator.prepare(val_dataloader)

        # 保存引用
        self._model = prepared_model
        self._optimizer = prepared_optimizer
        self._scheduler = scheduler
        self._train_dataloader = prepared_train_dl
        self._val_dataloader = prepared_val_dl

        if scheduler is not None:
            self._scheduler = self.accelerator.prepare(scheduler)

        # 初始化组件
        self._checkpoint_manager = AccelerateCheckpointManager(self.accelerator, self.ctx)
        self._distributed = AccelerateDistributed(self.accelerator)

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
        """裁剪梯度。"""
        if clip_algorithm == "norm":
            return self.accelerator.clip_grad_norm_(
                model.parameters(), clip_val
            ).item()
        elif clip_algorithm == "value":
            self.accelerator.clip_grad_value_(model.parameters(), clip_val)
            return None
        else:
            raise ValueError(f"Unknown clip algorithm: {clip_algorithm}")

    # ==================== 检查点方法 ====================

    def save_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> None:
        """保存检查点。"""
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.save(
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                step=self.ctx.iteration,
                extra_state=state,
            )
        else:
            self.accelerator.wait_for_everyone()
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self.accelerator.save_state(str(path))

    def load_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> Dict:
        """加载检查点。"""
        self.accelerator.load_state(str(path))
        return {"iteration": self.ctx.iteration, "epoch": self.ctx.epoch}

    def save_model(self, model: torch.nn.Module, path: Union[str, Path]) -> None:
        """保存模型权重。"""
        self.accelerator.wait_for_everyone()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 手动解包模型
        unwrapped_model = model
        while hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module

        if self.accelerator.is_main_process:
            state_dict = unwrapped_model.state_dict()
            torch.save(state_dict, str(path / "model.pt"))

            if hasattr(unwrapped_model, 'config') and hasattr(unwrapped_model.config, 'save_pretrained'):
                unwrapped_model.config.save_pretrained(str(path))

    # ==================== 日志方法 ====================

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """记录指标。"""
        self.accelerator.log(metrics, step=step)

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """记录音频。"""
        if self._tensorboard_tracker is not None:
            import numpy as np

            writer = self._tensorboard_tracker.writer

            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            if audio_np.ndim == 2:
                audio_np = audio_np[np.newaxis, ...]

            writer.add_audio(name, audio_np, global_step=step or 0, sample_rate=sample_rate)

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """记录图像。"""
        if self._tensorboard_tracker is not None:
            import numpy as np

            writer = self._tensorboard_tracker.writer

            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = image

            writer.add_image(name, image_np, global_step=step or 0)

    # ==================== 分布式方法 ====================

    def is_main_process(self) -> bool:
        """当前是否为主进程。"""
        return self.accelerator.is_main_process

    def is_local_main_process(self) -> bool:
        """当前是否为本地主进程。"""
        return self.accelerator.is_local_main_process

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """从所有进程收集张量。"""
        return self.accelerator.gather(tensor)

    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """跨进程归约张量。"""
        reduction = "mean" if op == "mean" else "sum"
        return self.accelerator.reduce(tensor, reduction=reduction)

    def barrier(self) -> None:
        """同步所有进程。"""
        self.accelerator.wait_for_everyone()

    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        self.accelerator.wait_for_everyone()

    # ==================== 属性 ====================

    @property
    def device(self) -> torch.device:
        """获取当前设备。"""
        return self.accelerator.device

    @property
    def precision(self) -> str:
        """获取当前精度模式。"""
        return str(self.accelerator.mixed_precision)

    def autocast(self, enabled: bool = True):
        """获取自动混合精度上下文管理器。"""
        if enabled and self.accelerator.mixed_precision != "no":
            return self.accelerator.autocast()
        else:
            return nullcontext()

    @property
    def gradient_accumulation_steps(self) -> int:
        """获取梯度累积步数。"""
        return self.accelerator.gradient_accumulation_steps

    def is_gradient_accumulation_boundary(self) -> bool:
        """当前是否为梯度累积边界。"""
        return self.accelerator.sync_gradients

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """获取未包装的原始模型。"""
        unwrapped = model
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        return unwrapped

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        """禁用梯度同步的上下文管理器。"""
        with self.accelerator.no_sync(model):
            yield

    def end_training(self) -> None:
        """结束训练，清理资源。"""
        if self._tensorboard_tracker is not None:
            self._tensorboard_tracker.finish()
        self.accelerator.end_training()

    # ==================== 组件访问 ====================

    @property
    def checkpoint_manager(self) -> Optional[AccelerateCheckpointManager]:
        """获取检查点管理器。"""
        return self._checkpoint_manager

    @property
    def distributed(self) -> Optional[AccelerateDistributed]:
        """获取分布式工具。"""
        return self._distributed