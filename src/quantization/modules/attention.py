import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import QLinear
from src.utils import Attention


class QAttention(Attention):
    def __init__(self, m: Attention, quan_w_fn=None, quan_a_fn=None, quan_attn_fn=None, **kwargs):
        assert type(m) == Attention
        super().__init__(
            dim=m.dim,
            num_heads=m.num_heads,
            attention_dropout=m.attention_dropout,
            projection_dropout=m.projection_dropout,
        )
        self.qkv = QLinear(
            self.qkv,
            quan_w_fn=copy.deepcopy(quan_w_fn),
            quan_a_fn=copy.deepcopy(quan_a_fn),
        )
        self.proj = QLinear(
            self.proj,
            quan_w_fn=copy.deepcopy(quan_w_fn),
            quan_a_fn=copy.deepcopy(quan_a_fn),
        )
        quan_attn_fn = quan_attn_fn if quan_attn_fn is not None else quan_a_fn
        self.quan_a_q_fn = copy.deepcopy(quan_attn_fn)
        self.quan_a_k_fn = copy.deepcopy(quan_attn_fn)
        self.quan_a_v_fn = copy.deepcopy(quan_attn_fn)
        # self.quan_a_softmax_fn = copy.deepcopy(quan_attn_fn)
        self.quan_a_softmax_fn = type(quan_attn_fn)(
            bit = self.quan_a_q_fn.bit,
            all_positive = True,
            symmetric = getattr(self.quan_a_q_fn, "symmetric", False),
            normalize_first = getattr(self.quan_a_q_fn, "normalize_first", False),
            per_channel = getattr(self.quan_a_q_fn, "per_channel", False),
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.quan_a_q_fn(q)
        k = self.quan_a_k_fn(k)
        v = self.quan_a_v_fn(v)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)
        # attn_prob = nn.Softmax(dim=-1)(attn_weights)
        # attn = attn.softmax(dim=-1)
        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # return x, attn_weights
        return x
