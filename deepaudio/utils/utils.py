# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
from typing import Iterable
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError:
    pass

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank: int, save_dir: str, zero_stage: int=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

def chunked(l: Iterable, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_dtype_str(dtype: str) -> torch.dtype:
    if dtype in ["bfloat16", "bf16"]:
        torch_dtype = torch.bfloat16
    elif dtype in ["half", "float16", "fp16"]:
        torch_dtype = torch.half
    elif dtype in ["float", "float32", "fp32"]:
        torch_dtype = torch.float32
    elif dtype in ["int", "int32"]:
        torch_dtype = torch.int
    elif dtype in ["short", "int16"]:
        torch_dtype = torch.short
    elif dtype in ["long", "int64"]:
        torch_dtype = torch.long
    else:
        raise Exception("Unknown torch dtype.")
    return torch_dtype

def searchsorted(arr, value):
    left = 0
    right = len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid

    return left
