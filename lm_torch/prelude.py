__all__ = [
    "smp", "gpus", "gpu",
    "bf16", "f16", "f32", "f64",
    "i8", "i16", "i32", "i64",
    "u8",
]

import torch

def _init_torch_gpus():
    gpus = []
    ngpu = torch.cuda.device_count()
    for idx in range(ngpu):
        gpus.append(torch.device("cuda", idx))
    return gpus

smp = torch.device("cpu")
gpus = _init_torch_gpus()
gpu = None
if len(gpus) > 0:
    gpu = gpus[0]

bf16 = torch.bfloat16
f16 = torch.float16
f32 = torch.float32
f64 = torch.float64
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
u8 = torch.uint8
