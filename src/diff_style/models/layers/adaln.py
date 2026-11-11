"""adaLN-Zero (DiT, Peebles & Xie 2023): a conditioning vector gives a shift and a scale after a norm, and a gate on
a residual branch output. The gates start at zero, so a conditioned block starts as the identity.

AdaLNModulation calculates the settings from the condition; Modulation holds those of one branch and applies them."""

from dataclasses import dataclass

import torch
from torch import nn


class AdaLNModulation(nn.Module):
    """The shifts, scales and gates that a condition gives a block: cond (B, cond_dim) -> SiLU -> Linear ->
    num_chunks vectors (B, 1, dim), one value per channel, shared by every token of a sample.
    The Linear starts at zero (DiT), so every shift, scale and gate starts at 0."""

    def __init__(self, cond_dim: int, dim: int, num_chunks: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, num_chunks * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.num_chunks = num_chunks

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        modulations = self.linear(self.silu(cond))  # (B, num_chunks * dim)
        per_sample = modulations.unsqueeze(1)  # (B, 1, num_chunks * dim): the same for every token
        return per_sample.chunk(self.num_chunks, dim=-1)  # num_chunks x (B, 1, dim)


@dataclass(frozen=True)
class Modulation:
    """The adaLN settings of one branch, each (B, 1, dim): the same for every token of a sample.
    A missing setting leaves the tokens unchanged: no gate for the final norm, nothing without adaLN.
    Not a gate of 1: torch 2.5's compiler crashes on a matmul times a Python number."""

    shift: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    gate: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.shift is None) != (self.scale is None):
            raise ValueError("a shift and a scale go together: give the two or neither")

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """DiT's modulation of the normalized tokens, before the branch."""
        return x if self.shift is None or self.scale is None else x * (1 + self.scale) + self.shift

    def gated(self, x: torch.Tensor) -> torch.Tensor:
        """adaLN-Zero's gate on the branch output."""
        return x if self.gate is None else self.gate * x
