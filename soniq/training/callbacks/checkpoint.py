# coding=utf-8
"""
Checkpoint callback for Soniq.
"""

import os
import torch
from typing import Optional, Dict, Any


class CheckpointCallback:
    """
    Callback for saving checkpoints during training.

    Example:
        ```python
        callback = CheckpointCallback(
            save_dir="./checkpoints",
            save_interval=1000,
            keep_last=3,
        )
        ```
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_interval: int = 1000,
        keep_last: int = 3,
        save_on_epoch_end: bool = True,
        filename_template: str = "epoch-{epoch:04d}-step-{step:07d}",
    ):
        """
        Initialize CheckpointCallback.

        Args:
            save_dir: Directory to save checkpoints.
            save_interval: Save every N steps.
            keep_last: Keep last N checkpoints.
            save_on_epoch_end: Save at end of each epoch.
            filename_template: Template for checkpoint filename.
        """
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.keep_last = keep_last
        self.save_on_epoch_end = save_on_epoch_end
        self.filename_template = filename_template

        self.saved_checkpoints = []

        os.makedirs(save_dir, exist_ok=True)

    def on_save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict] = None,
    ) -> str:
        """
        Save checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Scheduler to save.
            epoch: Current epoch.
            step: Current step.
            metrics: Metrics to save.

        Returns:
            Path to saved checkpoint.
        """
        filename = self.filename_template.format(epoch=epoch, step=step)
        path = os.path.join(self.save_dir, filename)

        state = {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
        }

        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        if metrics is not None:
            state["metrics"] = metrics

        torch.save(state, path)
        self.saved_checkpoints.append(path)

        # Remove old checkpoints
        if self.keep_last > 0 and len(self.saved_checkpoints) > self.keep_last:
            old_path = self.saved_checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return path

    def should_save(self, step: int, epoch: int, is_epoch_end: bool = False) -> bool:
        """
        Check if checkpoint should be saved.

        Args:
            step: Current step.
            epoch: Current epoch.
            is_epoch_end: Whether it's end of epoch.

        Returns:
            True if should save.
        """
        if is_epoch_end and self.save_on_epoch_end:
            return True
        if step % self.save_interval == 0 and step > 0:
            return True
        return False
