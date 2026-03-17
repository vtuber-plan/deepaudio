# coding=utf-8
"""
Base Logger interface for Soniq Training.

定义统一的日志记录接口，所有 Logger 实现都需要继承此类。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch


class BaseLogger(ABC):
    """
    日志记录器抽象基类。

    所有 Logger 实现（TensorBoard, WandB, MLflow 等）都需要继承此类。

    Example:
        ```python
        class MyLogger(BaseLogger):
            def log(self, metrics, step):
                for k, v in metrics.items():
                    print(f"[{step}] {k}: {v}")

            def log_audio(self, name, audio, sample_rate, step):
                # 保存音频
                pass
        ```
    """

    # Logger 类型标识
    logger_type: str = "base"

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 Logger。

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            **kwargs: 额外参数
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or "default"
        self._kwargs = kwargs

    @abstractmethod
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        记录标量指标。

        Args:
            metrics: 指标字典，值为标量或 tensor
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
            audio: 音频张量 (channels, samples) 或 (batch, channels, samples)
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
            image: 图像张量 (C, H, W) 或 (batch, C, H, W)
            step: 步数
        """
        raise NotImplementedError

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """
        记录文本。

        Args:
            name: 文本名称
            text: 文本内容
            step: 步数
        """
        pass  # 可选实现

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        记录超参数。

        Args:
            params: 超参数字典
        """
        pass  # 可选实现

    def log_model_summary(
        self,
        model: torch.nn.Module,
        input_size: Optional[tuple] = None,
    ) -> None:
        """
        记录模型摘要。

        Args:
            model: 模型
            input_size: 输入尺寸
        """
        pass  # 可选实现

    @abstractmethod
    def finish(self) -> None:
        """完成日志记录，关闭资源。"""
        raise NotImplementedError

    def start(self) -> None:
        """开始日志记录。"""
        pass

    @property
    def version(self) -> Optional[str]:
        """获取版本号。"""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(log_dir={self.log_dir})"


# Logger 注册表
class LoggerRegistry:
    """
    Logger 注册表。

    用于注册和创建不同类型的 Logger。

    Example:
        ```python
        # 注册 Logger
        LoggerRegistry.register("wandb", WandBLogger)

        # 创建 Logger
        logger = LoggerRegistry.create("tensorboard", log_dir="./logs")

        # 列出所有 Logger
        print(LoggerRegistry.list_available())
        ```
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, logger_cls: type) -> None:
        """注册 Logger 类型。"""
        cls._registry[name.lower()] = logger_cls

    @classmethod
    def create(
        cls,
        name: str,
        log_dir: Union[str, Path],
        **kwargs,
    ) -> BaseLogger:
        """
        创建 Logger 实例。

        Args:
            name: Logger 类型名称
            log_dir: 日志目录
            **kwargs: 传递给 Logger 的额外参数

        Returns:
            Logger 实例
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown logger type: {name}. "
                f"Available: {available}"
            )
        return cls._registry[name_lower](log_dir=log_dir, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的 Logger 类型。"""
        return list(cls._registry.keys())

    @classmethod
    def is_available(cls, name: str) -> bool:
        """检查 Logger 类型是否可用。"""
        return name.lower() in cls._registry