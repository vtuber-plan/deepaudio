# coding=utf-8
"""
Weights & Biases Logger Adapter for Soniq Training.

统一的 WandB 日志记录器。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from .base import BaseLogger, LoggerRegistry


class WandBLoggerAdapter(BaseLogger):
    """
    Weights & Biases 日志记录器适配器。

    Example:
        ```python
        logger = WandBLoggerAdapter(
            log_dir="./logs",
            project="my-project",
            name="experiment-1",
        )
        logger.log({"train/loss": 0.5}, step=100)
        logger.log_audio("generated", audio_tensor, sample_rate=24000, step=100)
        logger.finish()
        ```
    """

    logger_type = "wandb"

    def __init__(
        self,
        log_dir: Union[str, Path],
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        初始化 WandB Logger。

        Args:
            log_dir: 日志目录
            project: WandB 项目名称
            name: 运行名称
            entity: WandB 实体（用户名或团队名）
            tags: 标签列表
            notes: 运行备注
            config: 配置字典
        """
        super().__init__(log_dir, name, **kwargs)

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Please install it with: pip install wandb"
            )

        self._project = project
        self._name = name
        self._entity = entity
        self._tags = tags
        self._notes = notes
        self._config = config or {}
        self._run = None
        self._initialized = False

    def _init_run(self):
        """延迟初始化 wandb run。"""
        if self._initialized:
            return

        self._run = self._wandb.init(
            project=self._project,
            name=self._name,
            entity=self._entity,
            tags=self._tags,
            notes=self._notes,
            config=self._config,
            dir=str(self.log_dir),
            reinit=True,
        )
        self._initialized = True

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """记录指标。"""
        self._init_run()

        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.cpu().numpy()
            log_dict[key] = value

        self._wandb.log(log_dict, step=step)

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """记录音频。"""
        self._init_run()

        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # WandB 需要 (samples,) 或 (channels, samples) 形状
        if audio_np.ndim == 3:
            # (batch, channels, samples) -> 只取第一个
            audio_np = audio_np[0]

        audio_obj = self._wandb.Audio(audio_np, sample_rate=sample_rate)
        self._wandb.log({name: audio_obj}, step=step)

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """记录图像。"""
        self._init_run()

        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        # WandB 需要 (H, W, C) 形状
        if image_np.ndim == 3:
            if image_np.shape[0] < image_np.shape[2]:
                # (C, H, W) -> (H, W, C)
                image_np = image_np.transpose(1, 2, 0)

        image_obj = self._wandb.Image(image_np)
        self._wandb.log({name: image_obj}, step=step)

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """记录文本。"""
        self._init_run()
        self._wandb.log({name: self._wandb.Html(text)}, step=step)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """记录超参数。"""
        self._init_run()
        self._wandb.config.update(params)

    def finish(self) -> None:
        """完成运行。"""
        if self._run is not None:
            self._wandb.finish()
            self._run = None
            self._initialized = False

    @property
    def url(self) -> Optional[str]:
        """获取 WandB 运行 URL。"""
        if self._run is not None:
            return self._run.url
        return None

    @property
    def version(self) -> Optional[str]:
        """获取运行 ID。"""
        if self._run is not None:
            return self._run.id
        return None


# 注册 Logger
try:
    import wandb  # noqa: F401
    LoggerRegistry.register("wandb", WandBLoggerAdapter)
    LoggerRegistry.register("weights_and_biases", WandBLoggerAdapter)
except ImportError:
    pass