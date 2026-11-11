"""Multi-head attention for self- and cross-attention, with optional QK-norm, mask, causal mask and dropout,
and a swappable attention kernel.

Masks are boolean, True = the query may attend to the key, broadcastable to (B, heads, L_query, L_key).
mask=None means every query sees every key: that is what a diffusion/flow policy needs, and the
only case the flash kernels run (any mask tensor, even all True, makes SDPA use a slower kernel).

Kernels: any function attention_fn(q, k, v, mask, dropout_p, is_causal, scale) -> out, the arguments of
F.scaled_dot_product_attention, with q (B, heads, Lq, head_dim), k and v (B, heads, Lk, head_dim),
out (B, heads, Lq, head_dim).
    dropout_p: the probability of dropping each attention weight, applied on every call, so Attention
        passes 0 outside training.
    is_causal: query i sees keys 0..i only, without a mask tensor (so the flash kernels still run).
    scale: of the logits q k^T, None = 1 / sqrt(head_dim).
Hydra picks one with `attention_fn=<file in configs/attention_fn>`.
"""

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from diff_style.models.layers.init import INIT_STD, init_linear

try:  # FlashAttention 3: a separate package, installed only on Hopper machines
    from flash_attn_interface import flash_attn_func  # ty: ignore[import-not-found]
except ImportError:
    flash_attn_func = None


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None,
         dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None,
         kernels: Sequence[str] | None = None) -> torch.Tensor:
    """PyTorch's scaled_dot_product_attention. It picks the kernel: FlashAttention 2 (fp16/bf16, no mask,
    Ampere or newer), cuDNN, memory-efficient (any mask) or math (anything, slowest).
    kernels: allow only these, e.g. ["FLASH_ATTENTION"] (names of torch.nn.attention.SDPBackend);
    a call no allowed kernel can run raises an error instead of falling back. None = PyTorch picks."""
    # a C++ enum: no SDPBackend[name]
    allowed = sdpa_kernel([getattr(SDPBackend, name) for name in kernels]) if kernels is not None else nullcontext()
    with allowed:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal,
                                              scale=scale)


def flash3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None,
           dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """FlashAttention 3 (`flash_attn_interface`, Hopper GPUs only). No masks, no dropout."""
    if flash_attn_func is None:
        raise ImportError("flash3 needs FlashAttention 3 (the flash_attn_interface package): install it or use sdpa")
    if mask is not None:
        raise ValueError("flash3 supports no mask: pass mask=None or use sdpa")
    if dropout_p > 0:
        raise ValueError("flash3 supports no dropout: set the attention dropout to 0 or use sdpa")
    out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),  # takes (B, L, heads, head_dim)
                          softmax_scale=scale, causal=is_causal)
    out = out[0] if isinstance(out, tuple) else out  # some FA3 versions also return the log-sum-exp
    return out.transpose(1, 2)


def math_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None,
                   dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """softmax(q k^T / sqrt(head_dim)) v, written out: the readable reference the kernels are tested against."""
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = q @ k.transpose(-2, -1) * scale  # (B, heads, Lq, Lk)
    if is_causal:
        scores = scores.masked_fill(~causal_mask(q.shape[2], k.shape[2], q.device), float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.dropout(scores.softmax(dim=-1), dropout_p)  # zeroes weights, scales the rest by 1 / (1 - p)
    return weights @ v


def causal_mask(query_length: int, key_length: int, device: torch.device | None = None) -> torch.Tensor:
    """(Lq, Lk) boolean, True on and below the diagonal: query i sees keys 0..i (torch's is_causal)."""
    return torch.ones(query_length, key_length, dtype=torch.bool, device=device).tril()


class Attention(nn.Module):
    """Multi-head attention, parameters named as torchtitan's Llama 3 Attention.

    qk_norm: RMSNorm on each head's queries and keys before the dot product (as Qwen3, OLMo 2, SD3),
    which keeps the attention logits bounded and training stable.
    bias: biases on the projections (ViT/DiT have them, Llama doesn't).
    dropout: probability of dropping each attention weight, in training only (as nn.MultiheadAttention's).
    is_causal: query i sees keys 0..i only (self-attention only: in cross-attention the queries and keys
    have no shared position axis). Combined with a mask, both apply.
    scale: of the logits q k^T, None = 1 / sqrt(head_dim).
    attention_fn: the kernel, see the top of this file.
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = False, bias: bool = False, dropout: float = 0.0,
                 is_causal: bool = False, scale: float | None = None,
                 attention_fn: Callable[..., torch.Tensor] = sdpa) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.scale = scale
        self.attention_fn = attention_fn
        self.wq = nn.Linear(dim, dim, bias=bias)
        self.wk = nn.Linear(dim, dim, bias=bias)
        self.wv = nn.Linear(dim, dim, bias=bias)
        self.wo = nn.Linear(dim, dim, bias=bias)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-5) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, Lq, dim), the queries. context: (B, Lk, dim), the keys and values; None = x (self-attention)."""
        if context is None:
            context = x
        elif self.is_causal:
            raise ValueError("is_causal needs queries and keys on the same positions: use it in self-attention only")
        is_causal = self.is_causal
        if is_causal and mask is not None:  # torch's kernels take one or the other: fold the causal mask in
            mask, is_causal = mask & causal_mask(x.shape[1], x.shape[1], x.device), False
        q = self.q_norm(self.wq(x).unflatten(-1, (self.num_heads, -1)))  # (B, Lq, heads, head_dim)
        k = self.k_norm(self.wk(context).unflatten(-1, (self.num_heads, -1)))  # (B, Lk, heads, head_dim)
        v = self.wv(context).unflatten(-1, (self.num_heads, -1))  # (B, Lk, heads, head_dim)
        dropout_p = self.dropout if self.training else 0.0
        out = self.attention_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask, dropout_p,
                                is_causal, self.scale)
        return self.wo(out.transpose(1, 2).flatten(2))  # (B, Lq, dim)

    def init_weights(self, output_std: float) -> None:
        """torchtitan's Llama 3 Attention.init_weights: wq, wk, wv at 0.02, then wo (back into the stream) at
        output_std. Biases 0."""
        for linear in (self.wq, self.wk, self.wv):
            init_linear(linear, INIT_STD)
        init_linear(self.wo, output_std)
