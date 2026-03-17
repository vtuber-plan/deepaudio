# coding=utf-8
"""
Base Engine for Soniq Training.

训练引擎的抽象基类，定义统一接口。
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

from ..base.context import EngineContext
from ..base.callback import CallbackList


class BaseEngine(ABC):
    """
    训练引擎的抽象基类。

    定义统一的引擎接口，支持不同后端（Accelerate、Fabric）的切换。

    子类需要实现所有抽象方法。
    """

    engine_name: str = "base"

    def __init__(
        self,
        ctx: EngineContext,
        callbacks: Optional[CallbackList] = None,
    ):
        """
        初始化引擎。

        Args:
            ctx: 训练上下文
            callbacks: 回调列表
        """
        self.ctx = ctx
        self.callbacks = callbacks or CallbackList()

        # 运行时状态
        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None

    # ===== 核心抽象方法 =====

    @abstractmethod
    def setup(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, Optional[DataLoader]]:
        """
        设置模型、优化器、数据加载器。

        使用引擎特定的方式包装组件，返回包装后的对象。

        Args:
            model: 模型
            optimizer: 优化器
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            scheduler: 学习率调度器

        Returns:
            (prepared_model, prepared_optimizer, prepared_train_dl, prepared_val_dl)
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """
        执行反向传播。

        Args:
            loss: 损失值
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        执行优化器步骤。

        Args:
            optimizer: 优化器，如果为 None 则使用内部保存的优化器
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> None:
        """
        保存检查点。

        Args:
            path: 检查点保存路径
            state: 额外状态
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        path: Union[str, Path],
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        加载检查点。

        Args:
            path: 检查点路径
            state: 要加载的状态字典

        Returns:
            加载的状态
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model: torch.nn.Module, path: Union[str, Path]) -> None:
        """
        保存模型权重。

        Args:
            model: 模型
            path: 保存路径
        """
        raise NotImplementedError

    # ===== 日志和监控 =====

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录指标。

        Args:
            metrics: 指标字典
            step: 步数
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    # ===== 分布式相关 =====

    @abstractmethod
    def is_main_process(self) -> bool:
        """
        当前是否为主进程。

        Returns:
            是否为主进程
        """
        raise NotImplementedError

    @abstractmethod
    def is_local_main_process(self) -> bool:
        """
        当前是否为本地主进程。

        Returns:
            是否为本地主进程
        """
        raise NotImplementedError

    @abstractmethod
    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从所有进程收集张量。

        Args:
            tensor: 输入张量

        Returns:
            收集后的张量
        """
        raise NotImplementedError

    @abstractmethod
    def all_reduce(
        self, tensor: torch.Tensor, op: str = "mean"
    ) -> torch.Tensor:
        """
        跨进程归约张量。

        Args:
            tensor: 输入张量
            op: 归约操作 ("mean", "sum", "max", "min")

        Returns:
            归约后的张量
        """
        raise NotImplementedError

    @abstractmethod
    def barrier(self) -> None:
        """同步所有进程。"""
        raise NotImplementedError

    @abstractmethod
    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        raise NotImplementedError

    # ===== 设备和精度 =====

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """获取当前设备。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def precision(self) -> str:
        """获取当前精度模式。"""
        raise NotImplementedError

    @abstractmethod
    def autocast(self, enabled: bool = True):
        """
        获取自动混合精度上下文管理器。

        Args:
            enabled: 是否启用

        Returns:
            上下文管理器
        """
        raise NotImplementedError

    # ===== 梯度累积 =====

    @property
    @abstractmethod
    def gradient_accumulation_steps(self) -> int:
        """获取梯度累积步数。"""
        raise NotImplementedError

    @abstractmethod
    def is_gradient_accumulation_boundary(self) -> bool:
        """
        当前是否为梯度累积边界。

        Returns:
            是否为梯度累积边界
        """
        raise NotImplementedError

    # ===== 模型访问 =====

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        获取未包装的原始模型。

        Args:
            model: 包装后的模型

        Returns:
            原始模型
        """
        return model

    # ===== 实用方法 =====

    def print(self, *args, **kwargs) -> None:
        """
        仅在主进程打印。

        Args:
            *args: print 参数
            **kwargs: print 关键字参数
        """
        if self.is_main_process():
            print(*args, **kwargs)

    def should_save(self) -> bool:
        """
        判断是否应该保存检查点。

        默认只在主进程保存。

        Returns:
            是否应该保存
        """
        return self.is_local_main_process()

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        """
        禁用梯度同步的上下文管理器（用于梯度累积）。

        Args:
            model: 模型

        Yields:
            None
        """
        yield

    # ===== 检查点管理 =====

    def get_checkpoint_path(
        self, iteration: int, base_path: Path
    ) -> Path:
        """
        获取检查点路径。

        Args:
            iteration: 迭代次数
            base_path: 基础路径

        Returns:
            检查点路径
        """
        return base_path / f"checkpoint_{iteration}"

    def cleanup_old_checkpoints(
        self,
        checkpoint_dir: Path,
        keep_last_n: int,
    ) -> None:
        """
        清理旧检查点。

        Args:
            checkpoint_dir: 检查点目录
            keep_last_n: 保留最近 N 个
        """
        if not self.is_local_main_process():
            return

        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        while len(checkpoints) > keep_last_n:
            import shutil
            shutil.rmtree(checkpoints.pop(0))