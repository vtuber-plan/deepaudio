# coding=utf-8
"""
Composite Logger for Soniq Training.

支持同时使用多个 Logger，统一管理日志记录。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from .base import BaseLogger


class CompositeLogger(BaseLogger):
    """
    组合日志记录器。

    同时管理多个 Logger，统一接口调用。

    Example:
        ```python
        from soniq.training.loggers import TensorBoardLoggerAdapter, CompositeLogger

        # 创建多个 logger
        tb_logger = TensorBoardLoggerAdapter(log_dir="./logs/tb")

        # 组合使用
        logger = CompositeLogger([tb_logger])
        logger.log({"train/loss": 0.5}, step=100)  # 同时记录到所有 logger
        logger.finish()  # 关闭所有 logger
        ```

    支持动态添加 logger：
        ```python
        logger = CompositeLogger()
        logger.add_logger(TensorBoardLoggerAdapter(log_dir="./logs"))
        logger.add_logger(WandBLogger(project="my_project"))
        ```
    """

    logger_type = "composite"

    def __init__(
        self,
        loggers: Optional[List[BaseLogger]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        """
        初始化组合 Logger。

        Args:
            loggers: Logger 列表
            log_dir: 日志目录（用于创建新的 logger）
        """
        self._loggers: List[BaseLogger] = loggers or []
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")

    def add_logger(self, logger: BaseLogger) -> None:
        """添加 Logger。"""
        self._loggers.append(logger)

    def remove_logger(self, logger_type: str) -> None:
        """移除指定类型的 Logger。"""
        self._loggers = [
            l for l in self._loggers
            if l.logger_type != logger_type.lower()
        ]

    def get_logger(self, logger_type: str) -> Optional[BaseLogger]:
        """获取指定类型的 Logger。"""
        for logger in self._loggers:
            if logger.logger_type == logger_type.lower():
                return logger
        return None

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """记录标量指标到所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.log(metrics, step)
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to log metrics: {e}")

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """记录音频到所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.log_audio(name, audio, sample_rate, step)
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to log audio: {e}")

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """记录图像到所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.log_image(name, image, step)
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to log image: {e}")

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """记录文本到所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.log_text(name, text, step)
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to log text: {e}")

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """记录超参数到所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.log_hyperparams(params)
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to log hyperparams: {e}")

    def start(self) -> None:
        """启动所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.start()
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to start: {e}")

    def finish(self) -> None:
        """关闭所有 Logger。"""
        for logger in self._loggers:
            try:
                logger.finish()
            except Exception as e:
                print(f"[Warning] Logger {logger.logger_type} failed to finish: {e}")

    @property
    def num_loggers(self) -> int:
        """获取 Logger 数量。"""
        return len(self._loggers)

    def __len__(self) -> int:
        return len(self._loggers)

    def __iter__(self):
        return iter(self._loggers)

    def __repr__(self) -> str:
        logger_types = [l.logger_type for l in self._loggers]
        return f"CompositeLogger(loggers={logger_types})"