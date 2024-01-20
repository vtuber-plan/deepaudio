import sys
import os
import math
from typing import List, Optional, Tuple, Union
import json
import glob
import argparse
import platform
from deepaudio.light_modules.strategies.strategy_utils import setup_strategy_ddp, setup_strategy_deepspeed, setup_strategy_fsdp

from deepaudio.light_modules.utils.checkpoints import get_lastest_checkpoint
from deepaudio.utils.hparams import HParams


from deepaudio.utils.utils import get_optimizer_grouped_parameters, parse_dtype_str
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import lightning_fabric

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a deep learning model on a audio task")
    parser.add_argument('--hparams', type=str, default="hparams/hparams_hifigan_48k.json", help='The hparam file of training')
    parser.add_argument('--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('--device', type=str, default="", help='training device ids')
    parser.add_argument('--seed', type=int, default=43, help='model seed')

    args = parser.parse_args()
    return args


def setup_lora(
        base_model,
        r: int=128,
        target_modules: Optional[List[str]]=None,
        lora_alpha: int=8,
        lora_dropout: float=0.0,
        fan_in_fan_out: bool=False,
        bias: str="none"
    ) -> "PeftModel":
    from peft import LoftQConfig, LoraConfig, get_peft_model, PeftModel
    # loftq_config = LoftQConfig(loftq_bits=4, ...)           # set 4bit quantization
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=fan_in_fan_out,
        bias=bias,
    )
    model = get_peft_model(base_model, lora_config)
    return model

def train():
    torch.autograd.set_detect_anomaly(True)
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
    
    model = HifiGan.from_pretrained()

    # Setup LORA
    if "lora" in hparams:
        model = setup_lora(
            model,
            r=hparams.lora.get("r", 128),
            target_modules=hparams.lora.get("target_modules", []),
            lora_alpha=hparams.lora.get("lora_alpha", 8),
            lora_dropout=hparams.lora.get("lora_dropout", 0.0),
            fan_in_fan_out=hparams.lora.get("fan_in_fan_out", False),
            bias=hparams.lora.get("bias", 'none')
        )
        if hparams.get("gradient_checkpointing", False):
            model.enable_input_require_grads()
        # model.print_trainable_parameters()
    
    if hparams.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Save Model
    # save_hf_format(model, tokenizer, "./lightning_logs/huggingface_format", sub_folder=f"checkpoint-step-0")
    
    # Prepare the data
    print("***** Prepare Dataset *****")
    train_dataset, valid_dataset = create_dataset(
        hparams,
        hparams.data_path,
        hparams.data_output_path,
        args.seed,
        tokenizer,
        hparams.max_seq_len
    )

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

    # Checkpoint Settings
    checkpoint_every_n_train_steps = 100
    if "checkpoint_every_n_train_steps" in hparams:
        checkpoint_every_n_train_steps = hparams.checkpoint_every_n_train_steps
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=None, save_last=True, every_n_train_steps=checkpoint_every_n_train_steps,
        save_weights_only=False, save_on_train_epoch_end=True, save_top_k=-1
    )

    # Earlystop Settings
    # monitor="val_loss", mode="min", save_top_k=5
    # earlystop_callback = EarlyStopping(monitor="valid/loss_mel_epoch", mode="min", patience=13)

    # Leaning rate monitor
    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    # GradientAccumulationScheduler
    # accumulator_callback = GradientAccumulationScheduler(scheduling={4: 2})
    if len(args.device) == 0:
        devices = [i for i in range(torch.cuda.device_count())]
    else:
        devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback, learning_rate_callback],
    }

    # Logger Settings
    trainer_params["log_every_n_steps"] = hparams.get("log_every_n_steps", 50)
    trainer_params["val_check_interval"] = hparams.get("val_check_interval", 1.0)

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

    trainer_params["max_epochs"] = hparams.get("max_epochs", 1000)
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
    ckpt_path = get_lastest_checkpoint("./lightning_logs", "checkpoints")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=ckpt_path)

