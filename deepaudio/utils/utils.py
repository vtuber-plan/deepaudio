import torch

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