# coding=utf-8
"""
Distributed utilities for Accelerate Engine.

提供 Accelerate 引擎的分布式训练功能。
"""

from typing import Any, Optional
import torch


class AccelerateDistributed:
    """
    Accelerate 分布式训练工具类。

    封装常用的分布式操作。

    Example:
        ```python
        dist = AccelerateDistributed(accelerator)

        # 检查主进程
        if dist.is_main():
            print("I'm the main process")

        # 同步所有进程
        dist.barrier()

        # 收集张量
        gathered = dist.gather(tensor)
        ```
    """

    def __init__(self, accelerator):
        """
        初始化分布式工具。

        Args:
            accelerator: Accelerator 实例
        """
        self.accelerator = accelerator

    @property
    def rank(self) -> int:
        """获取全局进程索引。"""
        return self.accelerator.process_index

    @property
    def local_rank(self) -> int:
        """获取本地进程索引。"""
        return self.accelerator.local_process_index

    @property
    def world_size(self) -> int:
        """获取进程总数。"""
        return self.accelerator.num_processes

    def is_main(self) -> bool:
        """是否为主进程（rank=0）。"""
        return self.accelerator.is_main_process

    def is_local_main(self) -> bool:
        """是否为本地主进程（local_rank=0）。"""
        return self.accelerator.is_local_main_process

    def barrier(self) -> None:
        """同步所有进程。"""
        self.accelerator.wait_for_everyone()

    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        self.accelerator.wait_for_everyone()

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
        self,
        tensor: torch.Tensor,
        op: str = "mean",
    ) -> torch.Tensor:
        """
        跨进程归约张量。

        Args:
            tensor: 输入张量
            op: 归约操作 ("mean", "sum")

        Returns:
            归约后的张量
        """
        reduction = "mean" if op == "mean" else "sum"
        return self.accelerator.reduce(tensor, reduction=reduction)

    def print_on_main(self, *args, **kwargs) -> None:
        """只在主进程打印。"""
        if self.is_main():
            print(*args, **kwargs)

    def run_on_main(self, func, *args, **kwargs) -> Optional[any]:
        """只在主进程执行函数。"""
        if self.is_main():
            return func(*args, **kwargs)
        return None

    def run_on_all(self, func, *args, **kwargs) -> None:
        """在所有进程上执行函数（先同步）。"""
        self.barrier()
        func(*args, **kwargs)
        self.barrier()

    @property
    def device(self) -> torch.device:
        """获取当前设备。"""
        return self.accelerator.device

    def to_device(self, data):
        """将数据移动到当前设备。"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(v) for v in data)
        return data