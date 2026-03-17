import os
from typing import Optional, Union

import yaml
from accelerate.tracking import GeneralTracker, on_main_process
from accelerate.utils import listify

from .logger import get_logger

logger = get_logger()

class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        **kwargs (additional keyword arguments, *optional*):
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir_param = logging_dir
        self.init_kwargs = kwargs

    @on_main_process
    def start(self):
        from torch.utils import tensorboard
        self.logging_dir = os.path.join(self.logging_dir_param, self.run_name)
        self.writer = tensorboard.writer.SummaryWriter(self.logging_dir, **self.init_kwargs)
        logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        from torch.utils.tensorboard.summary import hparams
        exp, ssi, sei = hparams(values, {}, None)
        file_writer = self.writer._get_file_writer()
        file_writer.add_summary(exp)
        file_writer.add_summary(ssi)
        file_writer.add_summary(sei)
        file_writer.flush()
        # self.writer.add_hparams(values, metric_dict={}, run_name='hparams')
        # self.writer.flush()
        with open(os.path.join(self.logging_dir, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(values, outfile)
            except yaml.representer.RepresenterError:
                logger.error("Serialization to store hyperparameters failed")
                raise
        logger.debug("Stored initial configuration hyperparameters to TensorBoard and hparams yaml file")

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        values = listify(values) # type: ignore
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()
        logger.debug("Successfully logged to TensorBoard")

    @on_main_process
    def log_images(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. (N, 3, H, W)
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `SummaryWriter.add_image` method.
        """
        for k, v in values.items():
            self.writer.add_images(k, v, global_step=step, **kwargs)
        logger.debug("Successfully logged images to TensorBoard")

    @on_main_process
    def log_audios(self, values: dict, step: Optional[int], sample_rate: int, **kwargs):
        """
        Logs `audios` to the current run.

        Args:
            values (Dictionary `str` to `np.ndarray`):
                Values to be logged as key-value pairs. The values need to have type `np.ndarray`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            sample_rate (`int`):
                The sample rate of the audio data.
            kwargs:
                Additional key word arguments passed along to the `SummaryWriter.add_audio` method.
        """
        for k, v in values.items():
            self.writer.add_audio(k, v, global_step=step, sample_rate=sample_rate, **kwargs)
        logger.debug("Successfully logged audios to TensorBoard")

    @on_main_process
    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.debug("TensorBoard writer closed")
