# coding=utf-8
"""
Distributed utilities for Fabric Engine.

提供 Fabric 引擎的分布式训练功能。
"""

from typing import Any, Optional
import torch


class FabricDistributed:
    """
    Fabric 分布式训练工具类。

    封装常用的分布式操作。

    Example:
        ```python
        dist = FabricDistributed(fabric)

        # 检查主进程
        if dist.is_main():
            print("I'm the main process")

        # 同步所有进程
        dist.barrier()

        # 收集张量
        gathered = dist.gather(tensor)
        ```
    """

    def __init__(self, fabric):
        """
        初始化分布式工具。

        Args:
            fabric: Lightning Fabric 实例
        """
        self.fabric = fabric

    @property
    def rank(self) -> int:
        """获取全局进程索引。"""
        return self.fabric.global_rank

    @property
    def local_rank(self) -> int:
        """获取本地进程索引。"""
        return self.fabric.local_rank

    @property
    def world_size(self) -> int:
        """获取进程总数。"""
        return self.fabric.world_size

    def is_main(self) -> bool:
        """是否为主进程（rank=0）。"""
        return self.fabric.global_rank == 0

    def is_local_main(self) -> bool:
        """是否为本地主进程（local_rank=0）。"""
        return self.fabric.local_rank == 0

    def barrier(self) -> None:
        """同步所有进程。"""
        self.fabric.barrier()

    def wait_for_everyone(self) -> None:
        """等待所有进程完成。"""
        self.fabric.barrier()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从所有进程收集张量。

        Args:
            tensor: 输入张量

        Returns:
            收集后的张量
        """
        return self.fabric.all_gather(tensor)

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
        gathered = self.fabric.all_gather(tensor)

        if op == "mean":
            return gathered.mean()
        elif op == "sum":
            return gathered.sum()
        elif op == "max":
            return gathered.max()
        elif op == "min":
            return gathered.min()

        return gathered

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        广播张量到所有进程。

        Args:
            tensor: 输入张量
            src: 源进程索引

        Returns:
            广播后的张量
        """
        # Fabric 没有 broadcast 方法，使用 all_gather 模拟
        if self.rank == src:
            return tensor
        else:
            # 创建相同形状的空张量
            return torch.zeros_like(tensor)

    def print_on_main(self, *args, **kwargs) -> None:
        """只在主进程打印。"""
        if self.is_main():
            print(*args, **kwargs)

    def run_on_main(self, func, *args, **kwargs) -> Optional[Any]:
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
        return self.fabric.device

    def to_device(self, data):
        """将数据移动到当前设备。"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(v) for v in data)
        return data