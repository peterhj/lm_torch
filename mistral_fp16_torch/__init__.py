__all__ = [
    "MistralConfig",
    "Mistral",
    "LlamaConfig",
    "Llama",
]

from torch_prelude import f16, f32, i64
import torch
import torch.functional
import torch.nn as nn

from dataclasses import dataclass
import enum
from enum import Enum
from typing import Optional

@dataclass
class MistralConstants:
    cos: torch.Tensor
    sin: torch.Tensor
    mask_w: torch.Tensor
    mask_b: torch.Tensor

@dataclass
class MistralConfig:
    tok_dim: int
    num_layer: int
    num_head: int
    head_dim: int
    mlp_inner_dim: int
    rms_norm_eps: float
    linear_scale: Optional[float] = None
    q_group: int = 1

    @staticmethod
    def llama_7b():
        return MistralConfig(
                tok_dim = 32000,
                num_layer = 32,
                num_head = 32,
                head_dim = 128,
                mlp_inner_dim = 11008,
                rms_norm_eps = 1.0e-5,
        )

    @staticmethod
    def mistral_7b():
        return MistralConfig(
                tok_dim = 32000,
                num_layer = 32,
                num_head = 32,
                head_dim = 128,
                mlp_inner_dim = 14336,
                rms_norm_eps = 1.0e-5,
                q_group = 4,
        )

    def constants(self, batch_size, max_seq_len, device = None) -> MistralConstants:
        cfg = self

        # cos, sin for rotary embedding.
        base = torch.asarray(10000.0, dtype=f32, device=device)
        exp_factor = torch.asarray(-2.0 / cfg.head_dim, dtype=f32, device=device)
        exp = torch.arange(0, cfg.head_dim // 2, dtype=f32, device=device) * exp_factor
        inv_freq = torch.pow(base, exp)
        pos = torch.arange(0, max_seq_len, dtype=f32, device=device)
        freq = torch.outer(pos, inv_freq)
        freq2 = torch.tile(freq, (1, 2))
        cos = torch.cos(freq2).to(dtype=f16)
        sin = torch.sin(freq2).to(dtype=f16)
        cos = cos.reshape((1, max_seq_len, 1, cfg.head_dim))
        sin = sin.reshape((1, max_seq_len, 1, cfg.head_dim))

        # Causal attention mask, multiply-add style.
        mask_w = torch.tril(torch.full((max_seq_len, max_seq_len), 1.0, dtype=f32, device=device), 0)
        mask_b = torch.triu(torch.full((max_seq_len, max_seq_len), -torch.inf, dtype=f32, device=device), 1)
        mask_w = mask_w.reshape((1, 1, max_seq_len, max_seq_len))
        mask_b = mask_b.reshape((1, 1, max_seq_len, max_seq_len))

        return MistralConstants(
                cos = cos,
                sin = sin,
                mask_w = mask_w,
                mask_b = mask_b,
        )

class MistralImpl(Enum):
    TorchBuiltin = enum.auto()
    Debug = enum.auto()

    Default = TorchBuiltin

class MistralTokenEmbedding(nn.Module):
    def __init__(self, cfg, device = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.weight = nn.parameter.Parameter(torch.empty((cfg.tok_dim, self.inner_dim), dtype=f16, device=device))

    def forward(self, tok):
        batch_size, seq_len = tok.shape
        idx = torch.ravel(tok).unsqueeze(1)
        idx = torch.tile(idx, (1, self.inner_dim))
        stm = torch.gather(self.weight, 0, idx.to(dtype=i64))
        stm = stm.reshape((batch_size, seq_len, self.inner_dim))
        return stm

class MistralRMSNorm(nn.Module):
    def __init__(self, cfg, device = None, layer_idx = None, label = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.eps = torch.asarray(cfg.rms_norm_eps, dtype=f32, device=device)
        self.weight = nn.parameter.Parameter(torch.empty((self.inner_dim,), dtype=f16, device=device))

    def forward(self, stm):
        stmshape = stm.shape
        x = stm
        x = x.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        x = x.to(dtype=f32)
        v = torch.mean(x * x, 1, keepdim=True)
        # NB: eps inside sqrt.
        t = x * torch.rsqrt(v + self.eps)
        w = self.weight.unsqueeze(0)
        y = t * w
        stm = y
        stm = stm.to(dtype=f16)
        stm = stm.reshape(stmshape)
        return stm

class MistralRotaryEmbedding(nn.Module):
    def __init__(self, cfg, consts, device = None, layer_idx = None):
        super().__init__()
        self.register_buffer("cos", consts.cos, persistent=False)
        self.register_buffer("sin", consts.sin, persistent=False)

    def forward(self, stm):
        inner = stm.ndim - 1
        split = stm.shape[inner] // 2
        lstm, rstm = torch.split(stm, [split, split], inner)
        stm2 = torch.cat([-rstm, lstm], inner)
        stm = (stm * self.cos) + (stm2 * self.sin)
        return stm

def _make_inv_linear_scale(cfg):
    if cfg.linear_scale is not None:
        return torch.reciprocal(torch.asarray(cfg.linear_scale, dtype=f16))
    else:
        return None

class MistralSelfAttention(nn.Module):
    def __init__(self, cfg, consts, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.impl = impl
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.num_kv_head = cfg.num_head // cfg.q_group
        self.q_group = cfg.q_group
        self.kv_inner_dim = self.num_kv_head * cfg.head_dim
        self.attn_scale = torch.reciprocal(torch.sqrt(torch.asarray(cfg.head_dim, dtype=f32, device=device)))
        self.inv_linear_scale = _make_inv_linear_scale(cfg)
        self.register_buffer("mask_w", consts.mask_w, persistent=False)
        self.register_buffer("mask_b", consts.mask_b, persistent=False)
        self.rot = MistralRotaryEmbedding(cfg, consts)
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=f16, device=device)
        self.k_proj = nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=f16, device=device)
        self.v_proj = nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=f16, device=device)
        self.o_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=f16, device=device)

    def forward(self, stm):
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        q_stm = self.q_proj(stm)
        k_stm = self.k_proj(stm)
        v_stm = self.v_proj(stm)
        if self.inv_linear_scale is not None:
            q_stm *= self.inv_linear_scale
            k_stm *= self.inv_linear_scale
            v_stm *= self.inv_linear_scale
        if self.q_group > 1:
            k_stm = k_stm.reshape((stmshape[0] * stmshape[1], self.num_kv_head, 1, stmshape[3]))
            v_stm = v_stm.reshape((stmshape[0] * stmshape[1], self.num_kv_head, 1, stmshape[3]))
            k_stm = torch.repeat_interleave(k_stm, self.q_group, 2)
            v_stm = torch.repeat_interleave(v_stm, self.q_group, 2)
        q_stm = q_stm.reshape(stmshape)
        k_stm = k_stm.reshape(stmshape)
        v_stm = v_stm.reshape(stmshape)
        q_stm = self.rot(q_stm)
        k_stm = self.rot(k_stm)
        q_stm = q_stm.transpose(1, 2)
        k_stm = k_stm.transpose(1, 2).transpose(2, 3)
        attn = torch.matmul(q_stm, k_stm).to(dtype=f32) * self.attn_scale
        attn = attn * self.mask_w + self.mask_b
        if self.impl == MistralImpl.TorchBuiltin:
            attn = torch.softmax(attn, 3)
        elif self.impl == MistralImpl.Debug:
            max_attn, _ = torch.max(attn, 3, keepdim=True)
            exp_attn = torch.exp(attn - max_attn)
            sum_exp_attn = torch.sum(exp_attn, 3, keepdim=True)
            attn = (exp_attn / sum_exp_attn)
        else:
            raise NotImplementedError
        attn = attn.to(dtype=f16)
        v_stm = v_stm.transpose(1, 2)
        o_stm = torch.matmul(attn, v_stm)
        o_stm = o_stm.transpose(1, 2)
        o_stm = o_stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        o_stm = self.o_proj(o_stm)
        if self.inv_linear_scale is not None:
            o_stm *= self.inv_linear_scale
        stm = o_stm.reshape(stmshape)
        return stm

def silu(x, impl = MistralImpl.Default):
    if impl == MistralImpl.TorchBuiltin:
        return nn.functional.silu(x)
    elif impl == MistralImpl.Debug:
        return x / (torch.exp(-x) + 1.0)
    else:
        raise NotImplementedError

class MistralMLP(nn.Module):
    def __init__(self, cfg, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.impl = impl
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.inv_linear_scale = _make_inv_linear_scale(cfg)
        self.gate_proj = nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=f16, device=device)
        self.up_proj = nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=f16, device=device)
        self.down_proj = nn.Linear(cfg.mlp_inner_dim, self.inner_dim, bias=False, dtype=f16, device=device)

    def forward(self, stm):
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        gate_stm = self.gate_proj(stm)
        up_stm = self.up_proj(stm)
        if self.inv_linear_scale is not None:
            gate_stm *= self.inv_linear_scale
            up_stm *= self.inv_linear_scale
        gate_stm = silu(gate_stm, self.impl)
        down_stm = gate_stm * up_stm
        down_stm = self.down_proj(down_stm)
        if self.inv_linear_scale is not None:
            down_stm *= self.inv_linear_scale
        stm = down_stm.reshape(stmshape)
        return stm

class MistralLayer(nn.Module):
    def __init__(self, cfg, consts, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg, device, layer_idx, label = "pre_attn")
        self.self_attn = MistralSelfAttention(cfg, consts, device, impl, layer_idx)
        self.post_attention_layernorm = MistralRMSNorm(cfg, device, layer_idx, label = "postattn")
        self.mlp = MistralMLP(cfg, device, impl, layer_idx)

    def forward(self, stm):
        residual_stm = stm
        stm = self.input_layernorm(stm)
        stm = self.self_attn(stm)
        stm = residual_stm + stm
        residual_stm = stm
        stm = self.post_attention_layernorm(stm)
        stm = self.mlp(stm)
        stm = residual_stm + stm
        return stm

class Mistral(nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len, device = None, impl = MistralImpl.Default):
        super().__init__()
        self.tok_dim = cfg.tok_dim
        self.num_head = cfg.num_head
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.inv_linear_scale = _make_inv_linear_scale(cfg)
        consts = cfg.constants(batch_size, max_seq_len, device)
        if impl == MistralImpl.TorchBuiltin:
            self.embed_tokens = nn.Embedding(cfg.tok_dim, self.inner_dim, dtype=f16, device=device)
        elif impl == MistralImpl.Debug:
            self.embed_tokens = MistralTokenEmbedding(cfg, device)
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList()
        for layer_idx in range(cfg.num_layer):
            self.layers.append(MistralLayer(cfg, consts, device, impl, layer_idx))
        self.norm = MistralRMSNorm(cfg, device)
        self.lm_head = nn.Linear(self.inner_dim, cfg.tok_dim, bias=False, dtype=f16, device=device)

    def forward(self, tok):
        batch_size, seq_len = tok.shape
        stm = self.embed_tokens(tok)
        stm = stm.reshape((batch_size, seq_len, self.num_head, self.head_dim))
        for layer in self.layers:
            stm = layer(stm)
        stm = self.norm(stm)
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        logit = self.lm_head(stm)
        logit = logit.to(dtype=f32)
        if self.inv_linear_scale is not None:
            logit *= self.inv_linear_scale
        return logit.reshape((batch_size, seq_len, self.tok_dim))

LlamaConfig = MistralConfig
Llama = Mistral
