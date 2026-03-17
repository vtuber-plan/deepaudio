# coding=utf-8
"""
Engine Context for Soniq Training.

统一训练状态管理类，支持序列化/反序列化用于断点续训。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import datetime


@dataclass
class EngineContext:
    """
    统一的训练状态上下文。

    集中管理训练的所有状态信息，支持序列化用于断点续训。

    Attributes:
        start_timestamp: 训练开始时间戳
        run_timestamp: 运行时间戳
        local_rank: 本地进程索引
        rank: 全局进程索引
        world_size: 进程总数
        run_path: 运行目录
        log_path: 日志目录
        ckpt_save_path: 检查点保存目录
        metrics_path: 指标目录
        weights_save_path: 权重保存目录
        seed: 随机种子
        debug_mode: 调试模式
        train_flag: 是否在训练模式
        iteration: 当前迭代次数
        epoch: 当前 epoch
        iteration_in_epoch: 当前 epoch 内的迭代次数
        first_epoch_since_resume: 是否是恢复后的第一个 epoch
        clip_grad_norm: 梯度裁剪范数
        gradient_accumulation_steps: 梯度累积步数
        num_iterations: 总迭代次数
        max_epochs: 最大 epoch 数
        log_interval_steps: 日志记录间隔
        val_interval_steps: 验证间隔
        save_interval_steps: 检查点保存间隔
        save_last_n: 保留最近 N 个检查点
        export_interval_steps: 权重导出间隔
        export_steps: 指定导出的步数列表
        hp: 超参数字典
    """

    # 时间戳
    start_timestamp: float = 0.0
    run_timestamp: float = 0.0

    # 分布式信息
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1

    # 路径
    run_path: Path = field(default_factory=lambda: Path("./runs"))
    log_path: Path = field(default_factory=lambda: Path("./runs/logs"))
    ckpt_save_path: Path = field(default_factory=lambda: Path("./runs/checkpoints"))
    metrics_path: Path = field(default_factory=lambda: Path("./runs/metrics"))
    weights_save_path: Path = field(default_factory=lambda: Path("./runs/weights"))

    # 训练状态
    seed: int = 3407
    debug_mode: bool = False
    train_flag: bool = True
    iteration: int = 0
    epoch: int = 0
    iteration_in_epoch: int = 0
    first_epoch_since_resume: bool = False

    # 训练配置
    clip_grad_norm: float = 1.0
    gradient_clip_algorithm: str = "norm"
    gradient_accumulation_steps: int = 1
    num_iterations: int = 100_000
    max_epochs: Optional[int] = None
    log_interval_steps: int = 200
    val_interval_steps: int = 2000
    save_interval_steps: int = 10_000
    save_last_n: int = 2
    export_interval_steps: Optional[int] = None
    export_steps: List[int] = field(default_factory=list)

    # 超参数记录
    hp: Dict[str, Union[str, int, float, bool, None]] = field(default_factory=dict)

    @property
    def seed_on_rank(self) -> int:
        """为每个进程生成独立的随机种子。"""
        return self.seed + self.rank

    @property
    def need_to_log(self) -> bool:
        """当前迭代是否需要记录日志。"""
        return (self.iteration + 1) % self.log_interval_steps == 0

    @property
    def need_to_validate(self) -> bool:
        """当前迭代是否需要验证。"""
        return (self.iteration + 1) % self.val_interval_steps == 0

    @property
    def need_to_save(self) -> bool:
        """当前迭代是否需要保存检查点。"""
        return (self.iteration + 1) % self.save_interval_steps == 0

    @property
    def need_to_export(self) -> bool:
        """当前迭代是否需要导出权重。"""
        if self.export_interval_steps is not None:
            return (self.iteration + 1) % self.export_interval_steps == 0
        return self.iteration in self.export_steps

    def to_dict(self) -> dict:
        """
        序列化为字典，支持 JSON/YAML 存储。

        Returns:
            包含所有状态的字典，Path 对象转换为字符串
        """
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                result[k] = str(v)
            elif isinstance(v, datetime.datetime):
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "EngineContext":
        """
        从字典反序列化。

        Args:
            data: 包含状态的字典

        Returns:
            EngineContext 实例
        """
        path_fields = [
            "run_path",
            "log_path",
            "ckpt_save_path",
            "metrics_path",
            "weights_save_path",
        ]
        data = data.copy()  # 避免修改原始数据
        for field_name in path_fields:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = Path(data[field_name])
        return cls(**data)

    def setup_paths(self, run_path: Union[str, Path]) -> None:
        """
        创建训练所需的目录结构。

        Args:
            run_path: 运行根目录
        """
        run_path = Path(run_path)
        self.run_path = run_path
        self.log_path = run_path / "logs"
        self.ckpt_save_path = run_path / "checkpoints"
        self.metrics_path = run_path / "metrics"
        self.weights_save_path = run_path / "weights"

        for path in [
            self.run_path,
            self.log_path,
            self.ckpt_save_path,
            self.metrics_path,
            self.weights_save_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def update_from_checkpoint(self, checkpoint: dict) -> None:
        """
        从检查点更新状态。

        Args:
            checkpoint: 包含状态的检查点字典
        """
        if "ctx" in checkpoint:
            loaded_ctx = self.from_dict(checkpoint["ctx"])
            # 只更新训练进度相关字段
            self.iteration = loaded_ctx.iteration
            self.epoch = loaded_ctx.epoch
            self.iteration_in_epoch = loaded_ctx.iteration_in_epoch
            self.first_epoch_since_resume = True
        elif "iteration" in checkpoint:
            self.iteration = checkpoint["iteration"]
            self.epoch = checkpoint.get("epoch", 0)