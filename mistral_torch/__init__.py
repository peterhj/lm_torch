__all__ = [
    "MistralConfig",
    "Mistral",
    "LlamaConfig",
    "Llama",
]

from torch_prelude import f16, f32, i64
import torch

from dataclasses import dataclass
import enum
from enum import Enum
from typing import Optional

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

class MistralLazyConstants:
    def __init__(self, cfg, max_seq_len):
        self.cfg = cfg
        self.max_seq_len = max_seq_len

    def _register_rot(self, mod):
        if not (hasattr(self, "cos") and hasattr(self, "sin")):
            # cos, sin for rotary embedding.
            dtype = mod.dtype
            device = None
            max_seq_len = self.max_seq_len
            base = torch.asarray(10000.0, dtype=f32, device=device)
            exp_factor = torch.asarray(-2.0 / self.cfg.head_dim, dtype=f32, device=device)
            exp = torch.arange(0, self.cfg.head_dim // 2, dtype=f32, device=device) * exp_factor
            inv_freq = torch.pow(base, exp)
            pos = torch.arange(0, max_seq_len, dtype=f32, device=device)
            freq = torch.outer(pos, inv_freq)
            freq2 = torch.tile(freq, (1, 2))
            cos = torch.cos(freq2).to(dtype=dtype)
            sin = torch.sin(freq2).to(dtype=dtype)
            cos = cos.reshape((1, max_seq_len, 1, self.cfg.head_dim))
            sin = sin.reshape((1, max_seq_len, 1, self.cfg.head_dim))
            self.cos = cos
            self.sin = sin
        if not (hasattr(mod, "cos") and hasattr(mod, "sin")):
            mod.register_buffer("cos", self.cos, persistent=False)
            mod.register_buffer("sin", self.sin, persistent=False)

    def _register_self_attn(self, mod):
        if not (hasattr(self, "mask_w") and hasattr(self, "mask_b")):
            # Causal attention mask, multiply-add style.
            device = None
            max_seq_len = self.max_seq_len
            mask_w = torch.tril(torch.full((max_seq_len, max_seq_len), 1.0, dtype=f32, device=device), 0)
            mask_b = torch.triu(torch.full((max_seq_len, max_seq_len), -torch.inf, dtype=f32, device=device), 1)
            mask_w = mask_w.reshape((1, 1, max_seq_len, max_seq_len))
            mask_b = mask_b.reshape((1, 1, max_seq_len, max_seq_len))
            self.attn_scale = torch.reciprocal(torch.sqrt(torch.asarray(self.cfg.head_dim, dtype=f32, device=device)))
            self.mask_w = mask_w
            self.mask_b = mask_b
        if not (hasattr(mod, "mask_w") and hasattr(mod, "mask_b")):
            mod.register_buffer("attn_scale", self.attn_scale, persistent=False)
            mod.register_buffer("mask_w", self.mask_w, persistent=False)
            mod.register_buffer("mask_b", self.mask_b, persistent=False)

    def _register_linear(self, mod):
        if not hasattr(self, "inv_linear_scale"):
            if self.cfg.linear_scale is not None:
                dtype = mod.dtype
                device = None
                self.inv_linear_scale = torch.reciprocal(torch.asarray(self.cfg.linear_scale, dtype=dtype, device=device))
            else:
                self.inv_linear_scale = None
        if not hasattr(mod, "inv_linear_scale"):
            if self.cfg.linear_scale is not None:
                mod.register_buffer("inv_linear_scale", self.inv_linear_scale, persistent=False)
            else:
                mod.inv_linear_scale = None

class MistralImpl(Enum):
    TorchBuiltin = enum.auto()
    Debug = enum.auto()

    Default = TorchBuiltin

class MistralTokenEmbedding(torch.nn.Module):
    def __init__(self, cfg, dtype = f16, device = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.weight = torch.nn.parameter.Parameter(torch.empty((cfg.tok_dim, self.inner_dim), dtype=dtype, device=device))

    def forward(self, tok):
        batch_size, seq_len = tok.shape
        idx = torch.ravel(tok).unsqueeze(1)
        idx = torch.tile(idx, (1, self.inner_dim))
        stm = torch.gather(self.weight, 0, idx.to(dtype=i64))
        stm = stm.reshape((batch_size, seq_len, self.inner_dim))
        return stm

class MistralRMSNorm(torch.nn.Module):
    def __init__(self, cfg, dtype = f16, device = None, layer_idx = None, label = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.eps = cfg.rms_norm_eps
        #self.eps = torch.asarray(cfg.rms_norm_eps, dtype=f32, device=device)
        self.dtype = dtype
        self.weight = torch.nn.parameter.Parameter(torch.empty((self.inner_dim,), dtype=dtype, device=device))

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
        stm = stm.to(dtype=self.dtype)
        stm = stm.reshape(stmshape)
        return stm

class MistralRotaryEmbedding(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.dtype = dtype
        self.consts = consts

    def forward(self, stm, seq_start = 0):
        self.consts._register_rot(self)
        seq_len = stm.shape[1]
        seq_end = seq_start + seq_len
        inner = stm.ndim - 1
        split = stm.shape[inner] // 2
        lstm, rstm = torch.split(stm, [split, split], inner)
        stm2 = torch.cat([-rstm, lstm], inner)
        stm = (stm * self.cos[:,seq_start:seq_end,:,:]) + (stm2 * self.sin[:,seq_start:seq_end,:,:])
        return stm

class MistralSelfAttention(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.impl = impl
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.num_kv_head = cfg.num_head // cfg.q_group
        self.q_group = cfg.q_group
        self.kv_inner_dim = self.num_kv_head * cfg.head_dim
        self.dtype = dtype
        self.consts = consts
        self.rot = MistralRotaryEmbedding(cfg, consts, dtype, device, layer_idx)
        self.q_proj = torch.nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = torch.nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = torch.nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = torch.nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)

    def forward(self, stm):
        self.consts._register_self_attn(self)
        self.consts._register_linear(self)
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
            attn = torch.nn.functional.softmax(attn, 3)
        elif self.impl == MistralImpl.Debug:
            max_attn, _ = torch.max(attn, 3, keepdim=True)
            exp_attn = torch.exp(attn - max_attn)
            sum_exp_attn = torch.sum(exp_attn, 3, keepdim=True)
            attn = (exp_attn / sum_exp_attn)
        else:
            raise NotImplementedError
        attn = attn.to(dtype=self.dtype)
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
        return torch.nn.functional.silu(x)
    elif impl == MistralImpl.Debug:
        return x / (torch.exp(-x) + 1.0)
    else:
        raise NotImplementedError

class MistralMLP(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.impl = impl
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.dtype = dtype
        self.consts = consts
        self.gate_proj = torch.nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=dtype, device=device)
        self.up_proj = torch.nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=dtype, device=device)
        self.down_proj = torch.nn.Linear(cfg.mlp_inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)

    def forward(self, stm):
        self.consts._register_linear(self)
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

class MistralLayer(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "pre_attn")
        self.self_attn = MistralSelfAttention(cfg, consts, dtype, device, impl, layer_idx)
        self.post_attention_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "postattn")
        self.mlp = MistralMLP(cfg, consts, dtype, device, impl, layer_idx)

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

class Mistral(torch.nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len, head = "lm", dtype = f16, device = None):
        super().__init__()
        self.tok_dim = cfg.tok_dim
        self.num_head = cfg.num_head
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.single_head = False
        if head is None:
            head = "stream"
        if isinstance(head, str):
            self.single_head = True
            head = (head,)
        self.head = tuple(head)
        self.dtype = dtype
        consts = MistralLazyConstants(cfg, max_seq_len)
        self.consts = consts
        impl = MistralImpl.Default
        if impl == MistralImpl.TorchBuiltin:
            self.embed_tokens = torch.nn.Embedding(cfg.tok_dim, self.inner_dim, dtype=dtype, device=device)
        elif impl == MistralImpl.Debug:
            self.embed_tokens = MistralTokenEmbedding(cfg, dtype, device)
        else:
            raise NotImplementedError
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(cfg.num_layer):
            self.layers.append(MistralLayer(cfg, consts, dtype, device, impl, layer_idx))
        self.norm = MistralRMSNorm(cfg, dtype, device)
        self.lm_head = torch.nn.Linear(self.inner_dim, cfg.tok_dim, bias=False, dtype=dtype, device=device)

    def forward(self, tok):
        self.consts._register_linear(self)
        batch_size, seq_len = tok.shape
        stm = self.embed_tokens(tok)
        stm = stm.reshape((batch_size, seq_len, self.num_head, self.head_dim))
        for layer in self.layers:
            stm = layer(stm)
        stm = self.norm(stm)
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0], stmshape[1], stmshape[2] * stmshape[3]))
        out = []
        for head in self.head:
            if head == "stream":
                out.append(stm)
            elif head == "lm":
                stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
                logit = self.lm_head(stm)
                logit = logit.to(dtype=f32)
                if self.inv_linear_scale is not None:
                    logit *= self.inv_linear_scale
                logit = logit.reshape((batch_size, seq_len, self.tok_dim))
                out.append(logit)
            else:
                raise ValueError("Mistral: not a valid output head: {}".format(head))
        if len(out) < 1:
            raise RuntimeError("Mistral: no output heads")
        if len(out) == 1 and self.single_head:
            return out[0]
        return tuple(out)

class CachedMistralSelfAttention(torch.nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.head_dim = cfg.head_dim
        self.num_head = cfg.num_head
        self.num_kv_head = cfg.num_head // cfg.q_group
        self.q_group = cfg.q_group
        self.kv_inner_dim = self.num_kv_head * cfg.head_dim
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.consts = consts
        self.rot = MistralRotaryEmbedding(cfg, consts, dtype, device, layer_idx)
        self.q_proj = torch.nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = torch.nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = torch.nn.Linear(self.inner_dim, self.kv_inner_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = torch.nn.Linear(self.inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.default_k_cache = None
        self.default_v_cache = None
        self.tagged_k_caches = dict()
        self.tagged_v_caches = dict()

    def fresh_cache_buf(self):
        bufshape = (self.batch_size, self.max_seq_len, self.num_head, self.head_dim)
        return torch.zeros(bufshape, dtype=self.dtype)

    def k_cache(self, cache_tag = None):
        if cache_tag is not None:
            if cache_tag not in self.tagged_k_caches:
                self.tagged_k_caches[cache_tag] = self.fresh_cache_buf()
            return self.tagged_k_caches[cache_tag]
        else:
            if self.default_k_cache is None:
                self.default_k_cache = self.fresh_cache_buf()
            return self.default_k_cache

    def v_cache(self, cache_tag = None):
        if cache_tag is not None:
            if cache_tag not in self.tagged_v_caches:
                self.tagged_v_caches[cache_tag] = self.fresh_cache_buf()
            return self.tagged_v_caches[cache_tag]
        else:
            if self.default_v_cache is None:
                self.default_v_cache = self.fresh_cache_buf()
            return self.default_v_cache

    def forward(self, stm, seq_start = 0, cache_tag = None):
        self.consts._register_self_attn(self)
        self.consts._register_linear(self)
        stmshape = stm.shape
        seq_len = stmshape[1]
        seq_end = seq_start + seq_len
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
        q_stm = self.rot(q_stm, seq_start)
        k_stm = self.rot(k_stm, seq_start)
        k_cache = self.k_cache(cache_tag)
        v_cache = self.v_cache(cache_tag)
        k_cache[:,seq_start:seq_end,:,:] = k_stm
        v_cache[:,seq_start:seq_end,:,:] = v_stm
        q_stm = q_stm.transpose(1, 2)
        k_stm = k_cache[:,:seq_end,:,:]
        k_stm = k_stm.transpose(1, 2).transpose(2, 3)
        attn = torch.matmul(q_stm, k_stm).to(dtype=f32) * self.attn_scale
        attn = attn * self.mask_w[:,:,seq_start:seq_end,:seq_end] + self.mask_b[:,:,seq_start:seq_end,:seq_end]
        attn = torch.nn.functional.softmax(attn, 3).to(dtype=self.dtype)
        v_stm = v_cache[:,:seq_end,:,:]
        v_stm = v_stm.transpose(1, 2)
        o_stm = torch.matmul(attn, v_stm)
        o_stm = o_stm.transpose(1, 2)
        o_stm = o_stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        o_stm = self.o_proj(o_stm)
        if self.inv_linear_scale is not None:
            o_stm *= self.inv_linear_scale
        stm = o_stm.reshape(stmshape)
        return stm

class CachedMistralLayer(torch.nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len, consts, dtype = f16, device = None, impl = MistralImpl.Default, layer_idx = None):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "pre_attn")
        self.self_attn = CachedMistralSelfAttention(cfg, batch_size, max_seq_len, consts, dtype, device, layer_idx)
        self.post_attention_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "postattn")
        self.mlp = MistralMLP(cfg, consts, dtype, device, impl, layer_idx)

    def forward(self, stm, seq_start = 0, cache_tag = None):
        residual_stm = stm
        stm = self.input_layernorm(stm)
        stm = self.self_attn(stm, seq_start, cache_tag)
        stm = residual_stm + stm
        residual_stm = stm
        stm = self.post_attention_layernorm(stm)
        stm = self.mlp(stm)
        stm = residual_stm + stm
        return stm

class CachedMistral(torch.nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len, head = "lm", dtype = f16, device = None):
        super().__init__()
        self.tok_dim = cfg.tok_dim
        self.num_head = cfg.num_head
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.single_head = False
        if head is None:
            head = "stream"
        if isinstance(head, str):
            self.single_head = True
            head = (head,)
        self.head = tuple(head)
        self.dtype = dtype
        consts = MistralLazyConstants(cfg, max_seq_len)
        self.consts = consts
        impl = MistralImpl.Default
        self.embed_tokens = torch.nn.Embedding(cfg.tok_dim, self.inner_dim, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(cfg.num_layer):
            self.layers.append(CachedMistralLayer(cfg, batch_size, max_seq_len, consts, dtype, device, impl, layer_idx))
        self.norm = MistralRMSNorm(cfg, dtype, device)
        self.lm_head = torch.nn.Linear(self.inner_dim, cfg.tok_dim, bias=False, dtype=dtype, device=device)

    def forward(self, tok, seq_start = 0, cache_tag = None):
        self.consts._register_linear(self)
        batch_size, seq_len = tok.shape
        stm = self.embed_tokens(tok)
        stm = stm.reshape((batch_size, seq_len, self.num_head, self.head_dim))
        for layer in self.layers:
            stm = layer(stm, seq_start, cache_tag)
        stm = self.norm(stm)
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0], stmshape[1], stmshape[2] * stmshape[3]))
        out = []
        for head in self.head:
            if head == "stream":
                out.append(stm)
            elif head == "lm":
                stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
                logit = self.lm_head(stm)
                logit = logit.to(dtype=f32)
                if self.inv_linear_scale is not None:
                    logit *= self.inv_linear_scale
                logit = logit.reshape((batch_size, seq_len, self.tok_dim))
                out.append(logit)
            else:
                raise ValueError("Mistral: not a valid output head: {}".format(head))
        if len(out) < 1:
            raise RuntimeError("Mistral: no output heads")
        if len(out) == 1 and self.single_head:
            return out[0]
        return tuple(out)

LlamaConfig = MistralConfig
Llama = Mistral
CachedLlama = CachedMistral
