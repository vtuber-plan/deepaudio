# coding=utf-8
"""
TensorBoard Logger Adapter for Soniq Training.

统一的 TensorBoard 日志记录器，可用于 Fabric 和 Accelerate 引擎。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from .base import BaseLogger, LoggerRegistry


class TensorBoardLoggerAdapter(BaseLogger):
    """
    TensorBoard 日志记录器适配器。

    提供统一的 TensorBoard 接口，封装 torch.utils.tensorboard.SummaryWriter。

    Example:
        ```python
        logger = TensorBoardLoggerAdapter(log_dir="./logs")
        logger.log({"train/loss": 0.5}, step=100)
        logger.log_audio("generated", audio_tensor, sample_rate=24000, step=100)
        logger.finish()
        ```
    """

    logger_type = "tensorboard"

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
        **kwargs,
    ):
        """
        初始化 TensorBoard Logger。

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称（作为子目录）
            comment: 日志文件注释
            purge_step: 清除步数（用于恢复训练）
            max_queue: 最大队列大小
            flush_secs: 刷新间隔（秒）
            filename_suffix: 文件名后缀
        """
        super().__init__(log_dir, experiment_name, **kwargs)

        from torch.utils.tensorboard import SummaryWriter

        # 构建完整日志路径
        full_log_dir = self.log_dir
        if experiment_name:
            full_log_dir = full_log_dir / experiment_name

        self.writer = SummaryWriter(
            log_dir=str(full_log_dir),
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

        self._step = 0

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        记录标量指标。

        Args:
            metrics: 指标字典
            step: 步数
        """
        if step is None:
            step = self._step
            self._step += 1

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.cpu().numpy()
            self.writer.add_scalar(key, value, global_step=step)

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
        import numpy as np

        if step is None:
            step = self._step

        # 转换为 numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # 确保形状正确: TensorBoard 需要 (batch, channels, samples)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, np.newaxis, ...]
        elif audio_np.ndim == 2:
            audio_np = audio_np[np.newaxis, ...]

        self.writer.add_audio(name, audio_np, global_step=step, sample_rate=sample_rate)

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
        import numpy as np

        if step is None:
            step = self._step

        # 转换为 numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        self.writer.add_image(name, image_np, global_step=step)

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """记录文本。"""
        if step is None:
            step = self._step
        self.writer.add_text(name, text, global_step=step)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """记录超参数。"""
        # 过滤不可序列化的参数
        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                filtered_params[k] = v
            elif isinstance(v, (list, tuple)):
                if all(isinstance(x, (int, float, str, bool)) for x in v):
                    filtered_params[k] = str(v)

        if filtered_params:
            self.writer.add_hparams(filtered_params, {})

    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """记录直方图。"""
        if step is None:
            step = self._step
        self.writer.add_histogram(name, values, global_step=step)

    def log_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
    ) -> None:
        """记录模型图。"""
        self.writer.add_graph(model, input_to_model)

    def finish(self) -> None:
        """关闭 TensorBoard writer。"""
        self.writer.flush()
        self.writer.close()

    def flush(self) -> None:
        """刷新缓冲区。"""
        self.writer.flush()

    @property
    def version(self) -> Optional[str]:
        """获取版本号。"""
        return None


# 注册 Logger
LoggerRegistry.register("tensorboard", TensorBoardLoggerAdapter)
LoggerRegistry.register("tb", TensorBoardLoggerAdapter)