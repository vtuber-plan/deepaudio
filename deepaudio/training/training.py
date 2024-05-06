import sys
import os
import math
from typing import List, Optional, Tuple, Union
import json
import glob
import argparse
import platform

from deepaudio.utils.hparams import HParams
from deepaudio.utils.utils import parse_dtype_str

# from deepaudio.trainers.strategies.strategy_utils import setup_strategy_ddp, setup_strategy_deepspeed, setup_strategy_fsdp
# from deepaudio.trainers.utils.checkpoints import get_lastest_checkpoint
# from deepaudio.utils.hparams import HParams
# from deepaudio.data.collators import DataCollatorWithPadding
# 
# from deepaudio.utils.model.model_utils import create_hf_model, save_hf_format
# from deepaudio.utils.model.tokenizer_utils import load_hf_tokenizer
# from deepaudio.utils.utils import get_optimizer_grouped_parameters, parse_dtype_str
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import lightning_fabric

def parse_args():
    parser = argparse.ArgumentParser(description="Train an audio model.")
    parser.add_argument('--hparams', type=str, default="hparams/vocoders/hifigan_16k.json", help='The hparam file of training')
    parser.add_argument('--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('--device', type=str, default="", help='training device ids')
    parser.add_argument('--seed', type=int, default=43, help='model seed')
    parser.add_argument('--path', type=str, default="llm_trainer", help='experiment save path')

    args = parser.parse_args()
    return args

def train(create_dataset, lightning_module_class):
    # torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    master_port = os.environ.get("MASTER_PORT", None)
    master_addr = os.environ.get("MASTER_ADDR", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    rank = os.environ.get("RANK", None)

    # If passed along, set the training seed now.
    lightning_fabric.seed_everything(args.seed)

    # Load model
    torch_dtype_str = hparams.get("model_torch_dtype", "auto")
    if torch_dtype_str != "auto":
        torch_dtype = parse_dtype_str(torch_dtype_str)
    else:
        torch_dtype = torch_dtype_str
    
    if torch_dtype == torch.bfloat16 and args.accelerator in ["cpu"]:
        raise RuntimeError("Models in bfloat16 cannot run with the accelerator CPU.")
    if torch_dtype == torch.float16 and args.accelerator in ["cpu"]:
        raise RuntimeError("Models in float16 cannot run with the accelerator CPU.")
    
    if hparams.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Prepare the data
    print("***** Prepare Dataset *****")

    # DataLoaders creation:
    print("***** DataLoaders creation *****")
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=hparams.max_seq_len)
    train_dataloader = DataLoader(train_dataset, collate_fn=collator, sampler=train_sampler, num_workers=4, batch_size=hparams.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collator, sampler=valid_sampler, num_workers=4, batch_size=hparams.per_device_eval_batch_size)

    model = lightning_module_class(
        model,
        hparams,
    )

    if len(args.device) == 0:
        devices = [i for i in range(torch.cuda.device_count())]
    else:
        devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [],
    }

    # Logger Settings
    loggers = hparams.get("logger", [{"logger_type": "tb", "save_dir": "lightning_logs"}])
    if isinstance(loggers, list) or isinstance(loggers, tuple):
        loggers = loggers
    elif isinstance(loggers, dict):
        loggers = [loggers]
    else:
        raise Exception("Unsupported type in logger field")
    
    tb_logger = []
    for logger in loggers:
        logger_type = logger.get("logger_type", "tb")
        save_dir = logger.get("save_dir", "logs")
        logger_save_dir = os.path.join(args.path, save_dir)

        if logger_type.lower() in ["tb", "tensorboard"]:
            tb_logger.append(pl_loggers.TensorBoardLogger(save_dir=logger_save_dir, name=logger.get("name", None), version=logger.get("version", None)))
        elif logger_type.lower() in ["comet"]:
            tb_logger.append(pl_loggers.CometLogger(save_dir=logger_save_dir))
        elif logger_type.lower() in ["csv"]:
            tb_logger.append(pl_loggers.CSVLogger(save_dir=logger_save_dir))
        elif logger_type.lower() in ["mlflow"]:
            tb_logger.append(pl_loggers.MLFlowLogger(save_dir=logger_save_dir))
        elif logger_type.lower() in ["neptune"]:
            tb_logger.append(pl_loggers.NeptuneLogger(save_dir=logger_save_dir))
        elif logger_type.lower() in ["wandb"]:
            tb_logger.append(pl_loggers.WandbLogger(save_dir=logger_save_dir))
        
    trainer_params["logger"] = tb_logger
    trainer_params["log_every_n_steps"] = hparams.get("log_every_n_steps", 50)

    # Leaning rate monitor
    if len(tb_logger)> 0:
        learning_rate_callback = LearningRateMonitor(logging_interval="step")
        trainer_params["callbacks"].append(learning_rate_callback)

    # Checkpoint Settings
    if hparams.get("enable_checkpoints", True):
        checkpoint_every_n_train_steps = 100
        if "checkpoint_every_n_train_steps" in hparams:
            checkpoint_every_n_train_steps = hparams.checkpoint_every_n_train_steps
        
        dirpath = None
        if len(tb_logger) == 0:
            dirpath = os.path.join(args.path, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath, save_last=True, every_n_train_steps=checkpoint_every_n_train_steps,
            save_weights_only=False, save_on_train_epoch_end=True, save_top_k=-1
        )
        trainer_params["callbacks"].append(checkpoint_callback)
    
    # Earlystop Settings
    if hparams.get("enable_earlystop", False):
        # monitor="val_loss", mode="min", save_top_k=5
        earlystop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=13)
        trainer_params["callbacks"].append(earlystop_callback)
    
    # GradientAccumulationScheduler
    # accumulator_callback = GradientAccumulationScheduler(scheduling={4: 2})

    # Validation Settings
    trainer_params["val_check_interval"] = hparams.get("val_check_interval", 1.0)

    # Step limit settings
    trainer_params["limit_train_batches"] = hparams.get("limit_train_batches", None)
    trainer_params["limit_val_batches"] = hparams.get("limit_val_batches", None)
    trainer_params["limit_test_batches"] = hparams.get("limit_test_batches", None)
    trainer_params["limit_predict_batches"] = hparams.get("limit_predict_batches", None)

    # Devices
    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if "fp16" in hparams and hparams.fp16:
        print("using fp16")
        precision = "16-mixed"
        assert (args.accelerator not in ["cpu"]), "models in float16 cannot run with the accelerator CPU."
    elif "bf16" in hparams and hparams.bf16:
        print("using bf16")
        precision = "bf16-mixed"
        assert (args.accelerator not in ["cpu"]), "models in bfloat16 cannot run with the accelerator CPU."
    else:
        print("using fp32")
        precision = 32
    trainer_params["precision"] = precision

    if "strategy" in hparams:
        if hparams.strategy == None:
            strategy = "auto"
        elif hparams.strategy == "fsdp":
            strategy = setup_strategy_fsdp(hparams, world_size, rank, devices)
        elif hparams.strategy == "deepspeed":
            strategy = setup_strategy_deepspeed(hparams, world_size, rank, devices)
        elif hparams.strategy == "ddp":
            strategy = setup_strategy_ddp(hparams, world_size, rank, devices)
        else:
            raise ValueError("Unknown training strategy")
    elif len(devices) > 1:
        strategy = setup_strategy_ddp(hparams, world_size, rank, devices)
    else:
        strategy = "auto"
    trainer_params["strategy"] = strategy

    # gradient clip
    trainer_params["gradient_clip_algorithm"] = hparams.get("gradient_clip_algorithm", "norm")
    trainer_params["gradient_clip_val"] = hparams.get("gradient_clip_val", None)

    trainer_params["max_epochs"] = hparams.get("max_epochs", None)
    trainer_params["max_steps"] = hparams.get("max_steps", -1)
    trainer_params["accumulate_grad_batches"] = hparams.get("accumulate_grad_batches", 1)

    # Profiler
    if "advanced_profiler" in hparams:
        profiler = AdvancedProfiler(**hparams.advanced_profiler)
        trainer_params["profiler"] = profiler
    elif "simple_profiler" in hparams:
        profiler = SimpleProfiler(**hparams.simple_profiler)
        trainer_params["profiler"] = profiler

    # detect_anomaly
    trainer_params["detect_anomaly"] = hparams.get("detect_anomaly", False)
    trainer_params["deterministic"] = hparams.get("deterministic", None)
    trainer_params["benchmark"] = hparams.get("benchmark", None)

    # Other params
    if "trainer" in hparams and isinstance(hparams.trainer, dict):
        for key, value in hparams.trainer:
            trainer_params[key] = value
    
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # Resume training
    ckpt_path = get_lastest_checkpoint(os.path.join(args.path, "lightning_logs"), "checkpoints")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=ckpt_path)
