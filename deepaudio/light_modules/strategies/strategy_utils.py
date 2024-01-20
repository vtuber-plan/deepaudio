

import platform


def setup_strategy_fsdp(hparams, world_size, rank, devices):
    from lightning.pytorch.strategies import FSDPStrategy
    strategy_params = hparams.get("strategy_params", {})
    if platform.system().lower() == 'windows' and \
            "process_group_backend" in strategy_params and \
            strategy_params["process_group_backend"] == "nccl":
        raise ValueError("Windows does not support nccl")
    fsdp = FSDPStrategy(**strategy_params)
    return fsdp

def setup_strategy_deepspeed(hparams, world_size, rank, devices):
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from katheryne.utils.ds_utils import get_train_ds_config

    strategy_params = hparams.get("strategy_params", {})
    if world_size is None:
        ds_world_size = len(devices)
    else:
        ds_world_size = int(world_size)
    
    if "fp16" in hparams and hparams.fp16:
        ds_precision = "fp16"
    elif "bf16" in hparams and hparams.bf16:
        ds_precision = "bf16"
    else:
        ds_precision = "fp32"
    ds_config = get_train_ds_config(
        offload=strategy_params.get("offload", False),
        stage=strategy_params.get("zero_stage", 2),
        precision=ds_precision
    )

    ds_config['train_micro_batch_size_per_gpu'] = hparams.per_device_train_batch_size
    ds_config['train_batch_size'] = hparams.per_device_train_batch_size * ds_world_size * hparams.accumulate_grad_batches
    ds = DeepSpeedStrategy(
        zero_optimization=True,
        stage=strategy_params.get("zero_stage", 2),
        remote_device = hparams.get("remote_device", "cpu"),
        offload_optimizer = strategy_params.get("offload", False),
        offload_optimizer_device = 'cpu',
        offload_parameters = strategy_params.get("offload", False),
        cpu_checkpointing = strategy_params.get("offload", False),
        offload_params_device = "cpu",
        nvme_path=hparams.get("nvme_path", "./nvme_offload"),
        contiguous_memory_optimization=True,
        config=ds_config,
    )
    return ds

def setup_strategy_ddp(hparams, world_size, rank, devices):
    from lightning.pytorch.strategies import DDPStrategy
    strategy_params = hparams.get("strategy_params", {})
    if platform.system().lower() == 'windows' and \
            "process_group_backend" in strategy_params and \
            strategy_params["process_group_backend"] == "nccl":
        raise ValueError("Windows does not support nccl")
    ddp = DDPStrategy(**strategy_params)
    return ddp
