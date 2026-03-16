# coding=utf-8
"""
Fabric Trainer for Soniq.

This module provides the FabricTrainer base class for training models using Lightning Fabric.
"""

import os
import torch
from typing import Optional, Dict, Any, List, Callable, Union
from lightning import Fabric
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import json


class FabricTrainer:
    """
    Base trainer class for Soniq using Lightning Fabric.

    This trainer provides:
        - Automatic distributed training setup
        - Mixed precision support
        - Checkpoint saving and loading
        - Training loop management
        - Logging integration

    Example:
        ```python
        trainer = FabricTrainer(
            accelerator="gpu",
            devices=4,
            strategy="ddp",
            precision="16-mixed",
            max_epochs=100,
        )

        model = MyModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dataloader = DataLoader(dataset, batch_size=32)

        trainer.fit(model, optimizer, dataloader)
        ```
    """

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Optional[Union[int, List[int], str]] = None,
        num_nodes: int = 1,
        precision: str = "32-true",
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        loggers: Optional[List[str]] = None,
        callbacks: Optional[List] = None,
        default_root_dir: str = "./logs",
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Initialize FabricTrainer.

        Args:
            accelerator: Hardware accelerator ("cpu", "gpu", "mps", "auto").
            strategy: Training strategy ("auto", "ddp", "fsdp", "deepspeed").
            devices: Number of devices or device IDs.
            num_nodes: Number of nodes for distributed training.
            precision: Precision mode ("32-true", "16-mixed", "bf16-mixed").
            max_epochs: Maximum number of epochs to train.
            max_steps: Maximum number of steps to train (overrides max_epochs).
            gradient_accumulation_steps: Number of steps for gradient accumulation.
            gradient_clip_val: Gradient clipping value.
            gradient_clip_algorithm: Gradient clipping algorithm ("norm" or "value").
            loggers: List of loggers to use (["tensorboard"], ["csv"], or None).
            callbacks: List of custom callbacks.
            default_root_dir: Default directory for checkpoints and logs.
            resume_from_checkpoint: Path to checkpoint to resume from.
        """
        self.accelerator = accelerator
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.default_root_dir = default_root_dir
        self.resume_from_checkpoint = resume_from_checkpoint

        # Setup loggers
        self.loggers = self._setup_loggers(loggers)

        # Setup callbacks
        self.callbacks = self._setup_callbacks(callbacks)

        # Initialize Fabric
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            loggers=self.loggers,
            callbacks=self.callbacks,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _setup_loggers(self, loggers: Optional[List[str]]) -> List:
        """Setup logging."""
        logger_list = []

        if loggers is None:
            loggers = ["tensorboard"]

        for logger in loggers:
            if logger == "tensorboard":
                logger_list.append(
                    TensorBoardLogger(
                        root_dir=os.path.join(self.default_root_dir, "tensorboard"),
                        name="soniq",
                    )
                )
            elif logger == "csv":
                logger_list.append(
                    CSVLogger(
                        root_dir=os.path.join(self.default_root_dir, "csv"),
                        name="soniq",
                    )
                )

        return logger_list

    def _setup_callbacks(self, callbacks: Optional[List]) -> List:
        """Setup callbacks."""
        callback_list = callbacks if callbacks else []

        # Add default checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.default_root_dir, "checkpoints"),
            filename="epoch-{epoch:02d}-step-{step:07d}",
            auto_insert_metric_name=False,
            save_last=True,
            every_n_epochs=1,
        )
        callback_list.append(checkpoint_callback)

        # Add learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callback_list.append(lr_monitor)

        return callback_list

    def setup(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dataloaders: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Setup model, optimizer, and dataloaders with Fabric.

        Args:
            model: PyTorch model to train.
            optimizer: Optimizer for training.
            dataloaders: Data loaders for training.

        Returns:
            Tuple of (model, optimizer, dataloaders) prepared by Fabric.
        """
        if optimizer is not None:
            model, optimizer = self.fabric.setup(model, optimizer)
        else:
            model = self.fabric.setup(model)

        if dataloaders is not None:
            if isinstance(dataloaders, list):
                dataloaders = [self.fabric.setup_dataloaders(dl) for dl in dataloaders]
            else:
                dataloaders = self.fabric.setup_dataloaders(dataloaders)

        return model, optimizer, dataloaders

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_step_fn: Optional[Callable] = None,
        validate_fn: Optional[Callable] = None,
    ):
        """
        Train the model.

        Args:
            model: PyTorch model to train.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
            train_step_fn: Custom training step function.
            validate_fn: Custom validation function.
        """
        self.fabric.launch()

        # Setup model and optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model, optimizer = self.fabric.setup(model, optimizer)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)

        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self.fabric.load(self.resume_from_checkpoint, state={"model": model, "optimizer": optimizer})

        # Training loop
        self.fabric.barrier()

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self._train_epoch(
                model, train_dataloader, optimizer, scheduler, train_step_fn
            )

            # Validation
            val_loss = None
            if val_dataloader is not None and validate_fn is not None:
                val_loss = self._validate(model, val_dataloader, validate_fn)

            # Log metrics
            self.fabric.log("train/loss", train_loss, on_step=False, on_epoch=True)
            if val_loss is not None:
                self.fabric.log("val/loss", val_loss, on_step=False, on_epoch=True)

            # Check if max_steps reached
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

        self.fabric.barrier()

    def _train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_step_fn: Optional[Callable],
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            with self.fabric.autocast():
                if train_step_fn is not None:
                    loss = train_step_fn(model, batch)
                else:
                    # Default training step
                    loss = self._default_train_step(model, batch)

            # Backward pass
            self.fabric.backward(loss)

            # Gradient clipping
            if self.gradient_clip_val is not None:
                self.fabric.clip_gradients(
                    model, optimizer,
                    clip_val=self.gradient_clip_val,
                    clip_algorithm=self.gradient_clip_algorithm
                )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return total_loss / num_batches

    def _default_train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Default training step - to be overridden by subclasses."""
        # This is a placeholder - subclasses should implement their own
        raise NotImplementedError(
            "Subclasses must implement _default_train_step or provide train_step_fn"
        )

    def _validate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        validate_fn: Callable,
    ) -> float:
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                loss = validate_fn(model, batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict] = None,
    ):
        """
        Save a checkpoint.

        Args:
            path: Path to save the checkpoint.
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Scheduler to save.
            extra_state: Extra state to save.
        """
        state = {
            "model": model,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        if optimizer is not None:
            state["optimizer"] = optimizer

        if scheduler is not None:
            state["scheduler"] = scheduler

        if extra_state is not None:
            state.update(extra_state)

        self.fabric.save(path, state)

    def load_checkpoint(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict:
        """
        Load a checkpoint.

        Args:
            path: Path to the checkpoint.
            model: Model to load weights into.
            optimizer: Optimizer to load state.
            scheduler: Scheduler to load state.

        Returns:
            Extra state from checkpoint.
        """
        state = {
            "model": model,
        }

        if optimizer is not None:
            state["optimizer"] = optimizer

        if scheduler is not None:
            state["scheduler"] = scheduler

        self.fabric.load(path, state)
        self.current_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)

        return state
