"""Dropout masks fixed by a hash, so a captured reference is reproducible in every torch version.

torch draws other Bernoulli masks in other versions, even with the same seed (torch 2.8 than 2.5.1): the
seeded output of a dropout run matches only the torch that captured it. Inside `deterministic_dropout()`
every draw is a hash of the element index and of the draw's number instead, and an element is kept where
that hash, as a uniform value in [0, 1), is >= p:

    u[i] = splitmix32(i + draw * 2**32 / the golden ratio) / 2**32

Thus the reference capture and the tests get the same masks whatever their torch version. The masks are
still drawn in the order of the run, each of density 1 - p: a dropout at another site, or with another
probability, gives another mask.

Patched for the duration, each with the shape and probability its caller passes:
    torch.nn.functional.dropout      x * mask / (1 - p), torch's definition with a fixed mask (nn.Dropout)
    torch.rand                       the per-sample draws of NullTokenDropout and of diffusers' token_drop
    torch.Tensor.bernoulli_          the per-sample masks of DropPath, ours and timm's
    F.scaled_dot_product_attention   the attention dropout, which torch draws inside its kernel: the same
                                     math with the same fixed mask (math_attention), so both sides of a
                                     comparison run the same kernel.
"""

from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from diff_style.models.layers import math_attention

_GOLDEN = 0x9E3779B1  # 2**32 / the golden ratio: spreads consecutive draws over the whole range
_MASK_32 = 0xFFFFFFFF
_RANGE = 2 ** 32


def _hashed_uniform(index: torch.Tensor) -> torch.Tensor:
    """SplitMix32 of each index, divided by 2**32: a value in [0, 1) that looks random and never changes."""
    value = index & _MASK_32
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & _MASK_32
    value = ((value ^ (value >> 15)) * 0x846CA68B) & _MASK_32
    return (value ^ (value >> 16)).to(torch.float64) / _RANGE


class _Draws:
    """The draws of one run, numbered in the order the run makes them."""

    def __init__(self) -> None:
        self.number = 0

    def uniform(self, shape, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        index = torch.arange(torch.Size(shape).numel(), dtype=torch.int64) + self.number * _GOLDEN
        self.number += 1
        return _hashed_uniform(index).reshape(torch.Size(shape)).to(device=device, dtype=dtype)


def _hashed_dropout(draws: _Draws):
    """F.dropout's signature: the same x * mask / (1 - p) of torch, with the fixed mask in place of the draw."""
    def dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> torch.Tensor:
        if not training or p == 0:
            return input
        if p == 1:
            return torch.zeros_like(input)
        keep = draws.uniform(input.shape, input.dtype, input.device) >= p
        return input * keep / (1 - p)

    return dropout


def _hashed_rand(draws: _Draws):
    """torch.rand's signature: one fixed uniform draw, on the asked device and dtype."""
    def rand(*size, generator=None, dtype=None, device=None, **options) -> torch.Tensor:
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        return draws.uniform(shape, dtype or torch.get_default_dtype(), device)

    return rand


def _hashed_bernoulli(draws: _Draws):
    """Tensor.bernoulli_'s signature: each element 1 with probability p, from the fixed draw."""
    def bernoulli_(self: torch.Tensor, p: float = 0.5, *, generator=None) -> torch.Tensor:
        self.copy_((draws.uniform(self.shape, self.dtype, self.device) < p).to(self.dtype))
        return self

    return bernoulli_


def _masked_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None,
                      dropout_p: float = 0.0, is_causal: bool = False, scale=None,
                      enable_gqa: bool = False) -> torch.Tensor:
    """SDPA's signature, computed by math_attention: its dropout is F.dropout, which takes the fixed masks."""
    if enable_gqa:
        raise ValueError("deterministic_dropout supports no grouped-query attention")
    return math_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)


@contextmanager
def deterministic_dropout() -> Generator[None, None, None]:
    """In the block, every dropout takes its masks from the same fixed draws, in the order of the run."""
    draws = _Draws()
    patched = [(F, "dropout", _hashed_dropout(draws)), (torch, "rand", _hashed_rand(draws)),
               (torch.Tensor, "bernoulli_", _hashed_bernoulli(draws)),
               (F, "scaled_dot_product_attention", _masked_attention)]
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in patched]
    for owner, name, replacement in patched:
        setattr(owner, name, replacement)
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
