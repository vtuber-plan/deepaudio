import datetime
import logging
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .logger import get_logger
from .tracker import TensorBoardTracker


@dataclass
class EngineContext:
    start_timestamp: float
    run_timestamp: float
    local_rank: int
    rank: int
    run_path: Path
    log_path: Path
    ckpt_save_path: Path
    metrics_path: Path
    weights_save_path: Path
    seed: int
    debug_mode: bool
    # record whether the current mode needs training or val
    train_flag: bool
    iteration: int
    epoch: int
    iteration_in_epoch: int  # iteration in current epoch, for resuming dataloader
    first_epoch_since_resume: bool
    clip_grad_norm: float
    # hyperparameters will be logged to tensorboard
    hp: Dict[str, Union[str, int, float, bool, None]]
    gradient_accumulation_steps: int
    num_iterations: int
    log_interval_steps: int
    val_interval_steps: int
    save_interval_steps: int
    save_last_n: int
    export_interval_steps: Optional[int]
    export_steps: List[int]

    def to_dict(self) -> dict:
        return {
            k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()
        }

    @staticmethod
    def from_dict(ctx_dict: dict) -> "EngineContext":
        ctx_dict = {
            k: (Path(v) if k.endswith("_path") else v) for k, v in ctx_dict.items()
        }
        return EngineContext(**ctx_dict)

    @property
    def seed_on_rank(self) -> int:
        return self.seed + self.rank
    
    @property
    def need_to_log(self) -> bool:
        return (self.iteration + 1) % self.log_interval_steps == 0


class AccelerateEngine(ABC):
    def __init__(
        self,
        *,
        run_path: Union[str, Path],
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
        clip_grad_norm: float = 1.0,
        num_iterations: int = 100_000,
        log_interval_steps: int = 200,
        val_interval_steps: int = 2000,
        save_interval_steps: Optional[int] = None,
        save_last_n: int = 2,
        export_interval_steps: Optional[int] = None,
        export_steps: List[int] = [],
        seed: int = 3407,
        callbacks: List["EngineCallback"] = [],
        accelerator: Optional[Accelerator] = None,
        hp: Dict[str, Union[str, int, float, bool, None]] = {},
        accelerate_kwargs: Dict = {},
        **kwargs,
    ):
        # 0.setup unix timestamp (sec)
        start_time = datetime.datetime.now()

        # 1. setup context
        if accelerator is not None:
            if not isinstance(accelerator, Accelerator):
                raise TypeError(
                    "accelerator must be an instance of accelerate.Accelerator"
                )
            self.accelerator = accelerator
            if accelerate_kwargs:
                warnings.warn(
                    "accelerator instance is already set, accelerate_kwargs will be ignored",
                    category=UserWarning,
                    stacklevel=2,
                )
        else:
            accelerate_kwargs.setdefault("project_dir", run_path)
            if "log_with" in accelerate_kwargs:
                warnings.warn(
                    "log_with in accelerate_kwargs is unsupported.",
                    category=UserWarning,
                    stacklevel=2,
                )
            accelerate_kwargs["log_with"] = None
            self.accelerator = Accelerator(**accelerate_kwargs)

        self.device = self.accelerator.device

        run_path = Path(run_path)

        if debug:
            num_iterations = 6
            log_interval_steps = 1
            val_interval_steps = 2
            save_interval_steps = 3
            export_interval_steps = 3
        
        self.ctx = EngineContext(
            start_timestamp=start_time.timestamp(),
            run_timestamp=-1,  # will be updated later
            local_rank=self.accelerator.local_process_index,
            rank=self.accelerator.process_index,
            run_path=run_path,
            log_path=run_path / "logs",
            ckpt_save_path=run_path / "checkpoints",
            metrics_path=run_path / "metrics",
            weights_save_path=run_path / "weights",
            seed=seed,
            debug_mode=debug,
            train_flag=True,
            iteration=0,
            epoch=0,
            iteration_in_epoch=0,
            first_epoch_since_resume=False,
            clip_grad_norm=clip_grad_norm,
            hp=hp,
            gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
            num_iterations=num_iterations,
            log_interval_steps=log_interval_steps,
            val_interval_steps=val_interval_steps,
            save_interval_steps=(
                save_interval_steps
                if save_interval_steps is not None
                else round(num_iterations * 0.1)
            ),
            export_interval_steps=export_interval_steps,
            export_steps=export_steps,
            save_last_n=save_last_n,
        )

        # 2. create directories
        self.ctx.run_path.mkdir(parents=True, exist_ok=True)
        self.ctx.log_path.mkdir(parents=True, exist_ok=True)
        self.ctx.ckpt_save_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.trackers = [
            TensorBoardTracker(
                start_time.strftime("%Y-%m-%d-%H-%M-%S"), self.ctx.metrics_path
            )
        ]

        # 3. setup logger
        if logger is None:
            logger = get_logger(
                name="Engine",
                log_file=self.ctx.log_path
                / f"R{self.ctx.rank}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.log",
                # log_level=logging.DEBUG
                log_level=logging.DEBUG if self.ctx.debug_mode else logging.INFO,
            )
        self.log = logger

        # 4. setup random number generator
        set_seed(self.ctx.seed_on_rank)
        self.rng = torch.Generator(device=self.device)

        # 5. setup callbacks
        self.callbacks = callbacks

        self.info(
            f"Total {self.ctx.num_iterations} iterations, {self.ctx.gradient_accumulation_steps} gradient accumulation steps"
        )

        # Set components
        model = self.get_model()
        optimizer, lr_scheduler = self.get_optimizer(model)
        train_dataloader, val_dataloader = self.get_dataloaders()

        # Prepare models and optimizers with accelerate
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            model, optimizer, train_dataloader
        )
        self.sampler = self.train_dataloader.sampler

        if lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        if val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(val_dataloader)
        else:
            self.val_dataloader = None

        # must before the load_checkpoint
        self.accelerator.register_for_checkpointing(self)

        for callback in self.callbacks:
            callback.on_engine_init(self, kwargs)
    
    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """
        Get the model to be trained.

        :return: The model to be trained.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Get the train and val dataloader.
        If val_dataloader is None, the val_step will not be called.

        :return: Tuple of train DataLoader and val DataLoader (or None if no val dataloader)
        :rtype: Tuple[DataLoader, DataLoader | None]
        """
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(
        self,
        model: torch.nn.Module,
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        """
        Get the optimizer and scheduler.

        :param model: The model to be trained.
        :return: Tuple of optimizer and scheduler (or None if no scheduler)
        :rtype: Tuple[Optimizer, Scheduler | None]
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, batch, idx: int, iteration: int
    ) -> Union[torch.Tensor, dict, None]:
        """
        This function should be called in every iteration.
        `self.model` must can be called with `batch` as input.
        `idx` is the index of the batch in the dataloader.
        `iteration` is the current iteration number.
        Return a loss tensor or dict with `loss` key, or None if the batch is skipped.
        """
        raise NotImplementedError

    def clip_grad_norm(self, ctx: EngineContext, iteration: int) -> None:
        grad_norm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), ctx.clip_grad_norm
        ).item()
        if ctx.need_to_log:
            self.info(f"Iteration {iteration}, grad_norm: {grad_norm:.4f}")
            self.track({"grad_norm/model":grad_norm})

    def val_step(
        self, batch, idx: int, iteration: int
    ) -> Union[torch.Tensor, dict, None]:
        """
        This function should be called in every iteration.
        `self.model` must can be called with `batch` as input.
        `idx` is the index of the batch in the dataloader.
        `iteration` is the current iteration number.
        Return a loss tensor or dict with `loss` key, or None if the batch is skipped.
        """
        raise NotImplementedError

    def on_save_hook(self) -> Optional[dict]:
        """
        This function should be called when the `save_checkpoint` method are called. the return value will be saved in the checkpoint file.
        """
        return None

    def on_load_hook(self, customs: dict):
        """
        This function should be called when the `load_checkpoint` method are called. the customs argument is the return value of the `on_save` method.
        """
        pass

    def record_hyperparameters(self):
        """
        Store hyperparameters to trackers.
        """
        hp = self.ctx.hp.copy()
        hp["seed"] = self.ctx.seed
        hp["gradient_accumulation_steps"] = self.ctx.gradient_accumulation_steps
        hp["clip_grad_norm"] = self.ctx.clip_grad_norm
        hp["mixed_precision"] = self.accelerator.mixed_precision
        for tracker in self.accelerator.trackers:
            tracker.start()
            tracker.store_init_configuration(hp)


    def resume(self, resume_dataload_state: Optional[bool] = None):
        """
        Resume training
        resume_dataload_state: resume dataloader state，if None, will be set to False for IterableDataset, True otherwise

        Return Self for chained call
        """
        try:
            self.load_checkpoint()
            self.ctx.iteration += (
                1  # skip last step to avoid saving the same checkpoint
            )
        except FileNotFoundError:
            self.info("No checkpoint found, starting from scratch.")
        except Exception as e:
            self.error(f"Failed to load checkpoint: {e}")
            raise e

        # resume dataloader state if needed
        if resume_dataload_state is None:
            resume_dataload_state = not isinstance(
                self.train_dataloader.dataset, torch.utils.data.IterableDataset
            )
        if self.ctx.iteration_in_epoch > 0 and resume_dataload_state:
            self.info(f"Skipping {self.ctx.iteration_in_epoch} batches.")
            self.epoch_train_dataloader = self.accelerator.skip_first_batches(
                self.train_dataloader, num_batches=self.ctx.iteration_in_epoch
            )
            self.ctx.first_epoch_since_resume = True
        return self


    def run(self):
        """
        This function should be called to start the training.
        """
        if self.ctx.iteration >= self.ctx.num_iterations:
            self.warning(
                f"Training already finished at iteration {self.ctx.iteration}."
            )
            return
        # update the start time
        self.ctx.run_timestamp = self.current_timestamp()
        self.debug(
            f"[Timer] Initialization time: {self.elapsed_seconds():.4f} seconds.",
            only_local_rank_0=False,
        )
        self.record_hyperparameters()

        for callback in self.callbacks:
            callback.on_train_begin(self)

        self.model.train()

        while self.ctx.iteration < self.ctx.num_iterations:
            for callback in self.callbacks:
                callback.on_epoch_begin(self, self.ctx.epoch)

            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.ctx.epoch)

            loader = (
                self.epoch_train_dataloader
                if self.ctx.first_epoch_since_resume
                else self.train_dataloader
            )
            iteration_offset = (
                self.ctx.iteration_in_epoch if self.ctx.first_epoch_since_resume else 0
            )
            self.debug(
                f"[Epoch {self.ctx.epoch}] Data loading offset: {iteration_offset}"
            )

            timer_data = self.current_timestamp()
            for idx, batch in enumerate(loader):
                self.debug(
                    f"[Timer][It {self.ctx.iteration}] Data loading time: {(self.current_timestamp() - timer_data):.4f} seconds.",
                    only_local_rank_0=False,
                )

                if self.ctx.iteration >= self.ctx.num_iterations:
                    break

                self.ctx.iteration_in_epoch = idx + iteration_offset

                try:
                    is_skipped = self._step(
                        batch, self.ctx.iteration_in_epoch, self.ctx.iteration
                    )
                    if is_skipped:
                        timer_data = (
                            self.current_timestamp()
                        )  # reset timer to record the next data loading time
                        continue
                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        self.warning(
                            f"[Engine] - CUDA out of memory at step {self.ctx.iteration}",
                            only_local_rank_0=False,
                        )
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        timer_data = (
                            self.current_timestamp()
                        )  # reset timer to record the next data loading time
                        continue
                    else:
                        self.error(
                            f"Training failed: {e}, iteration {self.ctx.iteration}, batch {self.ctx.iteration_in_epoch}, data {batch}"
                        )
                        raise e

                try:
                    self._val(self.ctx.iteration)
                except Exception as e:
                    self.error(f"Validation failed: {e}")
                    raise e

                if (self.ctx.iteration + 1) % self.ctx.save_interval_steps == 0:
                    self.save_checkpoint()

                if (
                    self.ctx.export_interval_steps
                    and (self.ctx.iteration + 1) % self.ctx.export_interval_steps == 0
                ) or self.ctx.iteration in self.ctx.export_steps:
                    self.export_weights()

                self.ctx.iteration += 1

                timer_data = self.current_timestamp()

            self.ctx.first_epoch_since_resume = False
            self.ctx.epoch += 1
            for callback in self.callbacks:
                callback.on_epoch_end(self, self.ctx.epoch)

        # Wait for all processes to finish
        self.accelerator.wait_for_everyone()

        self.end()
        self.info(
            f"Training finished, total {self.ctx.iteration} iterations.",
            only_local_rank_0=False,
        )
        for callback in self.callbacks:
            callback.on_train_end(self)

    def _step(self, batch, idx: int, iteration: int) -> bool:
        """
        This function should be called in every iteration.
        if return True, the batch will be skipped.
        """
        if not self.ctx.train_flag:
            return True

        self.debug(f"Training step {iteration}", only_local_rank_0=False)

        for callback in self.callbacks:
            callback.on_step_begin(self, iteration, batch)

        with self.accelerator.accumulate(self.model):
            timer_forward = self.current_timestamp()
            outputs = self.train_step(batch, idx, iteration)
            self.debug(
                f"[Timer][It {iteration}] Train step time: {(self.current_timestamp() - timer_forward):.4f} seconds.",
                only_local_rank_0=False,
            )
            if outputs is None:
                self.debug(f"Skipping batch {iteration}", only_local_rank_0=False)
                return True

            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                if loss is None:
                    raise ValueError(
                        f"Expected outputs to have a 'loss' key, but got {outputs.keys()}"
                    )
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                raise TypeError(
                    f"Expected outputs to be a Tensor or a dict with 'loss' key, but got {type(outputs)}"
                )

            # Backward pass using accelerator
            timer_backward = self.current_timestamp()
            self.accelerator.backward(loss)
            self.debug(
                f"[Timer][It {iteration}] Backward time: {(self.current_timestamp() - timer_backward):.4f} seconds.",
                only_local_rank_0=False,
            )
            for callback in self.callbacks:
                callback.on_backward_end(self, iteration, batch)

            # Gradient clipping and optimizer step will only happen on accumulation boundaries
            if self.accelerator.sync_gradients:
                self.clip_grad_norm(self.ctx, iteration)

                self.optimizer.step()
                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

        # Log metrics only on sync boundaries
        if self.accelerator.sync_gradients:
            if self.ctx.need_to_log:
                metrics = {
                    "loss": self.reduce(loss).item(),
                }
                if self.lr_scheduler is not None:
                    metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                self.info(
                    f"Iteration {iteration}, {', '.join([f'{k}: {v:.8f}' for k, v in metrics.items()])}"
                )
                self.track({f"{k}/model": v for k, v in metrics.items()})

        for callback in self.callbacks:
            callback.on_step_end(self, iteration, batch, outputs)

        return False

    # TODO: need review/rewrite
    @torch.no_grad()
    def _val(self, iteration: int):
        if self.val_dataloader is None:
            return
        raise NotImplementedError
        if (iteration + 1) % self.ctx.val_interval_steps != 0 and self.train_flag:
            return

        # set model to eval mode
        self.train_flag = False
        self.info(f"Start validation for {iteration} iterations")
        self.model.eval()

        for callback in self.callbacks:
            callback.on_validation_begin(self, iteration)

        losses: List[float] = []
        predictions = []
        # num_batches = len(self.val_dataloader)

        for idx, batch in enumerate(self.val_dataloader):
            outputs = self.val_step(batch, idx, iteration)
            if outputs is None:
                self.debug(f"Skipping batch {iteration}", only_local_rank_0=False)
                continue

            predictions.append(outputs)

            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                if loss is None:
                    raise ValueError(
                        f"Expected outputs to contain 'loss' key, but got {outputs}"
                    )
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                raise TypeError(
                    f"Expected outputs to be a Tensor or a dict with 'loss' key, but got {type(outputs)}"
                )

            losses.append(loss.item())

        # Average loss across all processes
        if losses:
            loss_mean = self.accelerator.reduce(
                torch.tensor(sum(losses) / len(losses), device=self.device),
                reduction="mean",
            ).item()
        else:
            loss_mean = 0.0

        for callback in self.callbacks:
            callback.on_validation_end(self, iteration, predictions, losses)

        # reset model to train mode
        self.info(f"Validation iteration {iteration}, loss: {loss_mean}")
        self.model.train()
        self.train_flag = True

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None):
        """
        Save checkpoint.
        """
        if self.ctx.iteration == 0:
            return
        if path is None:
            checkpoint_path = (
                self.ctx.ckpt_save_path / f"checkpoint_{self.ctx.iteration}"
            )
        else:
            checkpoint_path = Path(path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_path, safe_serialization=False)  # type: ignore
        self.info(
            f"Trainer state saved at iteration {self.ctx.iteration}, checkpoint saved to {checkpoint_path}."
        )
        # remove old checkpoints
        if self.accelerator.is_local_main_process:
            ckpts = self.get_checkpoints()
            if len(ckpts) >= self.ctx.save_last_n:
                self.info("Removing old checkpoints...")
                _ = [shutil.rmtree(i) for i in ckpts[: -self.ctx.save_last_n]]

    def get_last_checkpoint_path(self) -> Optional[Path]:
        """
        Get the last checkpoint path.
        Return None if not found.
        """
        ckpt_list = self.get_checkpoints()
        if len(ckpt_list) == 0:
            return None
        self.info(f"Last checkpoint found: {ckpt_list[-1]}")
        return ckpt_list[-1]

    def get_checkpoints(self) -> List[Path]:
        """
        Get all checkpoints. sorted by iteration.
        Return empty list if not found.
        """
        if not self.ctx.ckpt_save_path.exists():
            return []

        checkpoints = sorted(
            [
                i
                for i in os.listdir(self.ctx.ckpt_save_path)
                if i.startswith("checkpoint_")
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        if len(checkpoints) == 0:
            return []
        return [self.ctx.ckpt_save_path / i for i in checkpoints]

    def load_checkpoint(self, path: Optional[Union[str, Path]] = None):
        """
        This method loads everything and should be used to resume training.
        """
        if path is None:
            checkpoint_path = self.get_last_checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint found, please specify the path.")
        else:
            checkpoint_path = Path(path)

        # Load the main checkpoint file
        self.accelerator.load_state(checkpoint_path)  # type: ignore

        self.info(
            f"Current iteration is set to {self.ctx.iteration}", only_local_rank_0=False
        )
        self.info(f"Checkpoint loaded from {checkpoint_path}", only_local_rank_0=False)

    def end(self):
        """
        This method should be called when the training is finished.
        """
        self.save_checkpoint()
        self.export_weights()
        self.accelerator.end_training()

    def export_weights(self, path: Optional[Union[str, Path]] = None):
        """
        Export the weights of the model.
        """
        if path is None:
            model_path = self.ctx.weights_save_path / f"step_{self.ctx.iteration}"
        else:
            model_path = Path(path)

        # Use accelerator to save the model
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.model, model_path, safe_serialization=False)
        self.info(f"Model weights are exported to {model_path}", only_local_rank_0=True)

    @property
    def unwrapped_model(self) -> torch.nn.Module:
        """
        Get the model without any wrappers.
        """
        return self.accelerator.unwrap_model(self.model)

    def state_dict(self) -> dict:
        """
        Get the state dict of the engine.
        """
        return {
            "ctx": self.ctx.to_dict(),
            "customs": self.on_save_hook(),
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load the state dict of the engine.
        """
        ctx = EngineContext.from_dict(state_dict["ctx"])
        if ctx.seed != self.ctx.seed:
            # reset seed and random number generator
            warnings.warn(f"Detected seed was changed: {ctx.seed} != {self.ctx.seed}, new seed {self.ctx.seed} will be used", category=UserWarning)
            ctx.seed = self.ctx.seed
        ctx.start_timestamp = self.ctx.start_timestamp
        ctx.local_rank = self.ctx.local_rank
        ctx.rank = self.ctx.rank
        self.ctx = ctx
        self.on_load_hook(state_dict["customs"])

    def reduce(
        self, tensor: Union[torch.Tensor, float, int], average: bool = True
    ) -> torch.Tensor:
        """
        Reduce the tensor across all processes.
        If average is True, the tensor will be averaged.
        If average is False, the tensor will be summed.
        """
        if not isinstance(tensor, torch.Tensor):
            rt = torch.tensor(
                tensor, dtype=torch.float32, device=self.device, requires_grad=False
            )
        else:
            rt = tensor.detach().clone()
        return self.accelerator.reduce(rt, reduction="mean" if average else "sum")  # type: ignore

    def debug(self, msg: str, only_local_rank_0: bool = True):
        """
        Print the message only if the local rank is 0.
        """
        if not only_local_rank_0 or self.accelerator.is_local_main_process:
            self.log.debug(f"[R{self.ctx.rank}] {msg}")

    def info(self, msg: str, only_local_rank_0: bool = True):
        """
        Print the message only if the local rank is 0.
        """
        if not only_local_rank_0 or self.accelerator.is_local_main_process:
            self.log.info(f"[R{self.ctx.rank}] {msg}")

    def warning(self, msg: str, only_local_rank_0: bool = True):
        """
        Print the warning message only if the local rank is 0.
        """
        if not only_local_rank_0 or self.accelerator.is_local_main_process:
            self.log.warning(f"[R{self.ctx.rank}] {msg}")

    def error(self, msg: str, only_local_rank_0: bool = True):
        """
        Print the error message only if the local rank is 0.
        """
        if not only_local_rank_0 or self.accelerator.is_local_main_process:
            self.log.error(f"[R{self.ctx.rank}] {msg}")

    def print(self, *args, **kwargs):
        """
        Print using accelerator's print method (only prints on main process).
        """
        self.accelerator.print(*args, **kwargs)

    def current_timestamp(self) -> float:
        """
        Get the current timestamp in seconds.
        """
        return datetime.datetime.now().timestamp()

    def elapsed_seconds(self) -> float:
        """
        Get the elapsed seconds since the start of the init/run.
        """
        return self.current_timestamp() - self.ctx.start_timestamp

    def track(self, metrics: Dict[str, Any]):
        """Track metrics."""
        self.accelerator.log(metrics, step=self.ctx.iteration)

    def get_tracker(self) -> TensorBoardTracker:
        """Get the tracker instance (default 'tensorboard')."""
        return self.accelerator.get_tracker("tensorboard")  # type: ignore


class EngineCallback(ABC):
    """Base class for callbacks."""

    def on_backward_end(self, engine: AccelerateEngine, iteration: int, batch):
        pass

    def on_engine_init(self, engine: AccelerateEngine, kwargs: dict):
        """Called when the engine is initialized."""
        pass

    def on_train_begin(self, engine: AccelerateEngine):
        """Called at the begin of training."""
        pass

    def on_train_end(self, engine: AccelerateEngine):
        """Called at the end of training."""
        pass

    def on_step_begin(self, engine: AccelerateEngine, iteration: int, batch):
        """Called at the begin of each training step."""
        pass

    def on_step_end(
        self,
        engine: AccelerateEngine,
        iteration: int,
        batch,
        output: Union[torch.Tensor, dict],
    ):
        """Called at the end of each training step."""
        pass

    def on_validation_begin(self, engine: AccelerateEngine, iteration: int):
        """Called at begin the of each validation epoch."""
        pass

    def on_validation_end(
        self,
        engine: AccelerateEngine,
        iteration: int,
        outputs: List[Union[torch.Tensor, dict]],
        losses: List[float],
    ):
        """Called at the end of each validation epoch."""
        pass

    def on_epoch_begin(self, engine: AccelerateEngine, epoch: int):
        """Called at the begin of each epoch."""
        pass

    def on_epoch_end(self, engine: AccelerateEngine, epoch: int):
        """Called at the end of each epoch."""
        pass


if __name__ == "__main__":
    net = torch.nn.Linear(1, 1)

    class Trainer(AccelerateEngine):
        def train_step(self, batch, idx: int, iteration: int):
            self.info(f"Training step {idx} batch shape {batch.shape}")
            x = batch.float().unsqueeze(-1)
            y = 2 * x + 1
            preds = self.model(x)
            loss = torch.nn.functional.mse_loss(preds, y)
            return loss
        def get_dataloaders(self):
            return  torch.utils.data.DataLoader(range(100), batch_size=4), None
        def get_optimizer(self, model):
            return torch.optim.Adam(model.parameters(), lr=0.001), None

    trainer = Trainer(
        model=net,
        run_path="./test_run",
        debug=True,
    )
    trainer.run()
