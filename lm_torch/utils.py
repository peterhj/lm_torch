__all__ = [
    "reseed_torch",
    "load_params",
    "load_params_to",
    "load_params_to_f32",
]

from .prelude import smp, f32

import safetensors.torch
import torch

from glob import glob
import os
import random

def reseed_torch():
    torch.manual_seed(random.randint(0, 0xffff_ffff_ffff_ffff))

def load_params(model_path, model_format = "safetensors"):
    params = dict()
    if model_format == "safetensors":
        for f in sorted(glob("model-*-of-*.safetensors", root_dir = model_path)):
            params.update(safetensors.torch.load_file(os.path.join(model_path, f)))
    elif model_format == "pickle":
        state = dict()
        for f in sorted(glob("pytorch_model-*-of-*.bin", root_dir = model_path)):
            state.update(torch.load(os.path.join(model_path, f)))
        for k, v in state.items():
            if k.find(".weight") >= 0:
                params[k] = v
        del state
    else:
        raise ValueError("unsupported model_format")
    save_params = params
    params = dict()
    for save_k in save_params:
        k = save_k
        if k.find("model.") == 0:
            k = k[6:]
        params[k] = save_params[save_k]
    del save_params
    return params

def load_params_to(model_path, model_format = "safetensors", dtype = f32):
    save_params = load_params(model_path, model_format)
    params = dict()
    with torch.device(smp):
        for k in list(save_params.keys()):
            params[k] = save_params[k].to(dtype=dtype)
            del save_params[k]
    del save_params
    return params

def load_params_to_f32(model_path, model_format = "safetensors"):
    return load_params_to(model_path, model_format, dtype = f32)
