# coding=utf-8
"""
Fabric Engine Adapter for Soniq Training.

基于 Lightning Fabric 的训练引擎主类。
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

try:
    from lightning import Fabric
    from lightning.fabric.loggers import TensorBoardLogger
    FABRIC_AVAILABLE = True
except ImportError:
    FABRIC_AVAILABLE = False

from ..base import BaseEngine
from ...base.context import EngineContext
from ...base.callback import CallbackList
from .loggers import create_fabric_loggers
from .checkpoint import FabricCheckpointManager
from .distributed import FabricDistributed


class FabricEngineAdapter(BaseEngine):
    """
    Lightning Fabric 引擎适配器。

    封装 Lightning Fabric，实现 BaseEngine 接口。

    Example:
        ```python
        ctx = EngineContext(seed=42, num_iterations=10000)
        engine = FabricEngineAdapter(
            ctx,
            precision="16-mixed",
            logger_types=["tensorboard", "wandb"],
        )
        model, optimizer, train_dl, val_dl = engine.setup(
            model, optimizer, train_dataloader, val_dataloader
        )
        ```
    """

    engine_name = "fabric"

    def __init__(
        self,
        ctx: EngineContext,
        callbacks: Optional[CallbackList] = None,
        # Logger 配置
        logger_types: Optional[List[str]] = None,
        logger_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Fabric 配置
        **fabric_kwargs,
    ):
        """
        初始化 Fabric 引擎适配器。

        Args:
            ctx: 训练上下文
            callbacks: 回调列表
            logger_types: Logger 类型列表，如 ["tensorboard", "wandb"]
            logger_configs: 各 Logger 的配置
            **fabric_kwargs: Fabric 初始化参数
        """
        if not FABRIC_AVAILABLE:
            raise ImportError(
                "Lightning Fabric is not available. "
                "Please install it with: pip install lightning"
            )

        super().__init__(ctx, callbacks)

        # 设置默认 Fabric 配置
        fabric_kwargs.setdefault("devices", "auto")
        fabric_kwargs.setdefault("accelerator", "auto")
        fabric_kwargs.setdefault("precision", "32-true")

        # 创建 Logger
        if "loggers" not in fabric_kwargs:
            loggers = create_fabric_loggers(
                ctx,
                logger_types=logger_types,
                logger_configs=logger_configs,
            )
            if loggers:
                fabric_kwargs["loggers"] = loggers
            else:
                # 默认使用 TensorBoard
                fabric_kwargs["loggers"] = [
                    TensorBoardLogger(root_dir=ctx.metrics_path, name="")
                ]

        self.fabric = Fabric(**fabric_kwargs)
        self._fabric_kwargs = fabric_kwargs

        # 初始化组件
        self._checkpoint_manager: Optional[FabricCheckpointManager] = None
        self._distributed: Optional[FabricDistributed] = None

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
        # 启动 Fabric
        self.fabric.launch()

        # 更新分布式信息
        self.ctx.local_rank = self.fabric.local_rank
        self.ctx.rank = self.fabric.global_rank
        self.ctx.world_size = self.fabric.world_size

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 使用 Fabric setup
        prepared_model, prepared_optimizer = self.fabric.setup(model, optimizer)

        prepared_train_dl = None
        if train_dataloader is not None:
            prepared_train_dl = self.fabric.setup_dataloaders(train_dataloader)

        prepared_val_dl = None
        if val_dataloader is not None:
            prepared_val_dl = self.fabric.setup_dataloaders(val_dataloader)

        # 保存引用
        self._model = prepared_model
        self._optimizer = prepared_optimizer
        self._scheduler = scheduler
        self._train_dataloader = prepared_train_dl
        self._val_dataloader = prepared_val_dl

        # 初始化组件
        self._checkpoint_manager = FabricCheckpointManager(self.fabric, self.ctx)
        self._distributed = FabricDistributed(self.fabric)

        return prepared_model, prepared_optimizer, prepared_train_dl, prepared_val_dl

    def backward(self, loss: torch.Tensor) -> None:
        """执行反向传播。"""
        self.fabric.backward(loss)

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
            return self.fabric.clip_gradients(
                model,
                self._optimizer,
                max_norm=clip_val,
            )
        elif clip_algorithm == "value":
            self.fabric.clip_gradients(
                model,
                self._optimizer,
                clip_val=clip_val,
            )
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
            # 直接使用 Fabric 保存
            self.fabric.barrier()
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            save_state = {
                "model": self._model,
                "epoch": self.ctx.epoch,
                "iteration": self.ctx.iteration,
                "ctx": self.ctx.to_dict(),
            }
            if self._optimizer is not None:
                save_state["optimizer"] = self._optimizer
            if self._scheduler is not None:
                save_state["scheduler"] = self._scheduler
            if state is not None:
                save_state.update(state)

            self.fabric.save(str(path / "state.ckpt"), save_state)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> Dict:
        """加载检查点。"""
        path = Path(path)
        ckpt_file = path / "state.ckpt" if path.is_dir() else path

        load_state = {"model": self._model}
        if self._optimizer is not None:
            load_state["optimizer"] = self._optimizer
        if self._scheduler is not None:
            load_state["scheduler"] = self._scheduler

        self.fabric.load(str(ckpt_file), load_state)

        if "ctx" in load_state:
            self.ctx.update_from_checkpoint(load_state)
        elif "iteration" in load_state:
            self.ctx.iteration = load_state["iteration"]
            self.ctx.epoch = load_state.get("epoch", 0)

        return load_state

    def save_model(self, model: torch.nn.Module, path: Union[str, Path]) -> None:
        """保存模型权重。"""
        self.fabric.barrier()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.fabric.save(str(path / "model.ckpt"), {"model": model})

    # ==================== 日志方法 ====================

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """记录指标。"""
        for key, value in metrics.items():
            self.fabric.log(key, value, step=step)

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """记录音频。"""
        import numpy as np

        for logger in self.fabric.loggers:
            if hasattr(logger, 'experiment'):
                writer = logger.experiment

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
        for logger in self.fabric.loggers:
            if hasattr(logger, 'experiment'):
                writer = logger.experiment

                if isinstance(image, torch.Tensor):
                    image_np = image.cpu().numpy()
                else:
                    image_np = image

                writer.add_image(name, image_np, global_step=step or 0)

    # ==================== 分布式方法 ====================

    def is_main_process(self) -> bool:
        """当前是否为主进程。"""
        return self.fabric.global_rank == 0

    def is_local_main_process(self) -> bool:
        """当前是否为本地主进程。"""
        return self.fabric.local_rank == 0

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """从所有进程收集张量。"""
        return self.fabric.all_gather(tensor)

    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """跨进程归约张量。"""
        gathered = self.fabric.all_gather(tensor)
        if op == "mean":
            return gathered.mean()
        elif op == "sum":
            return gathered.sum()
        return gathered

    def barrier(self) -> None:
        """同步所有进程。"""
        self.fabric.barrier()

    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        self.fabric.barrier()

    # ==================== 属性 ====================

    @property
    def device(self) -> torch.device:
        """获取当前设备。"""
        return self.fabric.device

    @property
    def precision(self) -> str:
        """获取当前精度模式。"""
        return str(self.fabric.precision)

    def autocast(self, enabled: bool = True):
        """获取自动混合精度上下文管理器。"""
        if enabled:
            return self.fabric.autocast()
        else:
            return nullcontext()

    @property
    def gradient_accumulation_steps(self) -> int:
        """获取梯度累积步数。"""
        return getattr(self.fabric, "gradient_accumulation_steps", 1)

    def is_gradient_accumulation_boundary(self) -> bool:
        """当前是否为梯度累积边界。"""
        return True

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """获取未包装的原始模型。"""
        return self.fabric.unwrap(model)

    # ==================== 组件访问 ====================

    @property
    def checkpoint_manager(self) -> Optional[FabricCheckpointManager]:
        """获取检查点管理器。"""
        return self._checkpoint_manager

    @property
    def distributed(self) -> Optional[FabricDistributed]:
        """获取分布式工具。"""
        return self._distributed