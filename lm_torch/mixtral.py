__all__ = [
    "MixtralConfig",
    "Mixtral",
    "CachedMixtral",
]

from .prelude import f16, f32, i64, cpu
from .mistral import (
    MistralLazyConstants,
    MistralRMSNorm,
    MistralSelfAttention,
    CachedMistralSelfAttention,
)

import torch

from dataclasses import dataclass
import enum
from enum import Enum
from typing import Optional

@dataclass
class MixtralConfig:
    tok_dim: int
    num_layer: int
    num_head: int
    head_dim: int
    num_expert_mlp: int
    mlp_inner_dim: int
    q_group: int
    rms_norm_eps: float = 1.0e-5
    rope_base: float = 1.0e6
    linear_scale: Optional[float] = None

    @staticmethod
    def mixtral_8x7b():
        return MixtralConfig(
                tok_dim = 32000,
                num_layer = 32,
                num_head = 32,
                head_dim = 128,
                num_expert_mlp = 8,
                mlp_inner_dim = 14336,
                q_group = 4,
        )

    @staticmethod
    def mixtral_8x22b():
        return MixtralConfig(
                tok_dim = 32000,
                num_layer = 56,
                num_head = 48,
                head_dim = 128,
                num_expert_mlp = 8,
                mlp_inner_dim = 16384,
                q_group = 6,
        )

class MixtralMLP(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.dtype = dtype
        self.consts = consts
        self.w1 = torch.nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=dtype, device=device)
        self.w2 = torch.nn.Linear(cfg.mlp_inner_dim, self.inner_dim, bias=False, dtype=dtype, device=device)
        self.w3 = torch.nn.Linear(self.inner_dim, cfg.mlp_inner_dim, bias=False, dtype=dtype, device=device)

    def forward(self, stm):
        self.consts._register_linear(self)
        gate_stm = self.w1(stm)
        up_stm = self.w3(stm)
        if self.inv_linear_scale is not None:
            gate_stm *= self.inv_linear_scale
            up_stm *= self.inv_linear_scale
        gate_stm = torch.nn.functional.silu(gate_stm)
        down_stm = gate_stm * up_stm
        down_stm = self.w2(down_stm)
        if self.inv_linear_scale is not None:
            down_stm *= self.inv_linear_scale
        stm = down_stm
        return stm

class MixtralMoE(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.inner_dim = cfg.num_head * cfg.head_dim
        self.num_expert_mlp = cfg.num_expert_mlp
        self.dtype = dtype
        self.device = device
        self.gate = torch.nn.Linear(self.inner_dim, cfg.num_expert_mlp, bias=False, dtype=dtype, device=device)
        self.experts = torch.nn.ModuleList()
        for _ in range(cfg.num_expert_mlp):
            self.experts.append(MixtralMLP(cfg, consts, dtype, device, layer_idx))

    def forward(self, stm):
        stmshape = stm.shape
        stm = stm.reshape((stmshape[0] * stmshape[1], stmshape[2] * stmshape[3]))
        gate_stm = self.gate(stm)
        gate_logit, gate_index = torch.topk(gate_stm, 2, 1)
        gate_w = torch.nn.functional.softmax(gate_logit.to(dtype=f32), 1)
        gate_w = gate_w.to(dtype=self.dtype)
        out_stm = torch.zeros(stm.shape, dtype=self.dtype)
        for e_rank in range(self.num_expert_mlp):
            for k in range(2):
                e_index = torch.nonzero(gate_index[:,k] == e_rank).squeeze()
                if e_index.ndim == 0:
                    e_index = e_index.unsqueeze(0)
                assert e_index.ndim == 1
                if e_index.shape[0] <= 0:
                    continue
                e_w = gate_w[e_index,k].unsqueeze(1)
                e_stm = stm[e_index,:]
                e_stm = self.experts[e_rank](e_stm)
                #out_stm[e_index,:] += e_w * e_stm
                out_stm.index_add_(0, e_index, e_w * e_stm)
        stm = out_stm.reshape(stmshape)
        return stm

class MixtralLayer(torch.nn.Module):
    def __init__(self, cfg, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "pre_attn")
        self.self_attn = MistralSelfAttention(cfg, consts, dtype, device, layer_idx)
        self.post_attention_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "postattn")
        self.block_sparse_moe = MixtralMoE(cfg, consts, dtype, device, layer_idx)

    def forward(self, stm):
        residual_stm = stm
        stm = self.input_layernorm(stm)
        stm = self.self_attn(stm)
        stm = residual_stm + stm
        residual_stm = stm
        stm = self.post_attention_layernorm(stm)
        stm = self.block_sparse_moe(stm)
        stm = residual_stm + stm
        return stm

class Mixtral(torch.nn.Module):
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
        self.embed_tokens = torch.nn.Embedding(cfg.tok_dim, self.inner_dim, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(cfg.num_layer):
            self.layers.append(MixtralLayer(cfg, consts, dtype, device, layer_idx))
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
                raise ValueError("Mixtral: not a valid output head: {}".format(head))
        if len(out) < 1:
            raise RuntimeError("Mixtral: no output heads")
        if len(out) == 1 and self.single_head:
            return out[0]
        return tuple(out)

class CachedMixtralLayer(torch.nn.Module):
    def __init__(self, cfg, max_batch_size, max_seq_len, consts, dtype = f16, device = None, layer_idx = None):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "pre_attn")
        self.self_attn = CachedMistralSelfAttention(cfg, max_batch_size, max_seq_len, consts, dtype, device, layer_idx)
        self.post_attention_layernorm = MistralRMSNorm(cfg, dtype, device, layer_idx, label = "postattn")
        self.block_sparse_moe = MixtralMoE(cfg, consts, dtype, device, layer_idx)

    def forward(self, stm, seq_start = 0, cache_tag = None):
        residual_stm = stm
        stm = self.input_layernorm(stm)
        stm = self.self_attn(stm, seq_start, cache_tag)
        stm = residual_stm + stm
        residual_stm = stm
        stm = self.post_attention_layernorm(stm)
        stm = self.block_sparse_moe(stm)
        stm = residual_stm + stm
        return stm

class CachedMixtral(torch.nn.Module):
    def __init__(self, cfg, max_batch_size, max_seq_len, head = "lm", dtype = f16, device = None):
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
        self.embed_tokens = torch.nn.Embedding(cfg.tok_dim, self.inner_dim, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(cfg.num_layer):
            self.layers.append(CachedMixtralLayer(cfg, max_batch_size, max_seq_len, consts, dtype, device, layer_idx))
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
                raise ValueError("Mixtral: not a valid output head: {}".format(head))
        if len(out) < 1:
            raise RuntimeError("Mixtral: no output heads")
        if len(out) == 1 and self.single_head:
            return out[0]
        return tuple(out)
