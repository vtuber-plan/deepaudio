# coding=utf-8
"""
MLflow Logger Adapter for Soniq Training.

统一的 MLflow 日志记录器。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from .base import BaseLogger, LoggerRegistry


class MLflowLoggerAdapter(BaseLogger):
    """
    MLflow 日志记录器适配器。

    Example:
        ```python
        logger = MLflowLoggerAdapter(
            log_dir="./logs",
            experiment_name="my-experiment",
            run_name="run-1",
        )
        logger.log({"train/loss": 0.5}, step=100)
        logger.finish()
        ```
    """

    logger_type = "mlflow"

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        初始化 MLflow Logger。

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            run_name: 运行名称
            tracking_uri: MLflow tracking URI
            tags: 运行标签
        """
        super().__init__(log_dir, experiment_name, **kwargs)

        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Please install it with: pip install mlflow"
            )

        self._run_name = run_name
        self._tracking_uri = tracking_uri or str(self.log_dir / "mlruns")
        self._tags = tags
        self._run = None
        self._initialized = False

        # 设置 tracking URI
        self._mlflow.set_tracking_uri(self._tracking_uri)

        # 设置实验
        if experiment_name:
            self._mlflow.set_experiment(experiment_name)

    def _init_run(self):
        """延迟初始化 MLflow run。"""
        if self._initialized:
            return

        self._run = self._mlflow.start_run(run_name=self._run_name, tags=self._tags)
        self._initialized = True

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """记录指标。"""
        self._init_run()

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else float(value.cpu().numpy().mean())
            elif not isinstance(value, (int, float)):
                value = float(value)

            self._mlflow.log_metric(key, value, step=step)

    def log_audio(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        step: Optional[int] = None,
    ) -> None:
        """记录音频。"""
        self._init_run()

        import tempfile
        import soundfile as sf

        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # 保存为临时文件然后记录
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np.T if audio_np.ndim > 1 else audio_np, sample_rate)
            artifact_path = f"audio/{name}_{step or 0}.wav"
            self._mlflow.log_artifact(f.name, artifact_path)

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """记录图像。"""
        self._init_run()

        import tempfile
        import matplotlib.pyplot as plt
        import numpy as np

        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        # 转换为 (H, W, C)
        if image_np.ndim == 3 and image_np.shape[0] < image_np.shape[2]:
            image_np = image_np.transpose(1, 2, 0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.imsave(f.name, image_np)
            artifact_path = f"images/{name}_{step or 0}.png"
            self._mlflow.log_artifact(f.name, artifact_path)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """记录超参数。"""
        self._init_run()

        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                filtered_params[k] = v
            elif isinstance(v, (list, tuple)):
                filtered_params[k] = str(v)

        self._mlflow.log_params(filtered_params)

    def finish(self) -> None:
        """完成运行。"""
        if self._run is not None:
            self._mlflow.end_run()
            self._run = None
            self._initialized = False

    @property
    def run_id(self) -> Optional[str]:
        """获取运行 ID。"""
        if self._run is not None:
            return self._run.info.run_id
        return None

    @property
    def version(self) -> Optional[str]:
        """获取运行 ID。"""
        return self.run_id


# 注册 Logger
try:
    import mlflow  # noqa: F401
    LoggerRegistry.register("mlflow", MLflowLoggerAdapter)
except ImportError:
    pass