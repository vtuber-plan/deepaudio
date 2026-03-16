# coding=utf-8
"""
Early stopping callback for Soniq.
"""

from typing import Optional, Callable
import torch


class EarlyStoppingCallback:
    """
    Callback for early stopping during training.

    Example:
        ```python
        callback = EarlyStoppingCallback(
            patience=10,
            min_delta=0.001,
            mode="min",
        )
        ```
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        monitor: str = "val/loss",
    ):
        """
        Initialize EarlyStoppingCallback.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" or "max" - whether to minimize or maximize metric.
            monitor: Metric name to monitor.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor

        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def on_epoch_end(self, metrics: dict) -> bool:
        """
        Check for early stopping at end of epoch.

        Args:
            metrics: Dictionary of metrics.

        Returns:
            True if should stop training.
        """
        if self.monitor not in metrics:
            return False

        current_value = metrics[self.monitor]

        if self.best_value is None:
            self.best_value = current_value
            return False

        # Check if improvement
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
