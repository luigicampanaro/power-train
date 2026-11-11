import math

import torch
from torch import nn

from diff_style.models.layers.init import INIT_STD
from diff_style.models.layers.mlp import MLP


class TimestepEmbedding(nn.Module):
    """Flow time tau (B,) in [0, 1] -> (B, dim): sinusoidal features, then an MLP, as DiT's TimestepEmbedder.

    DiT's frequencies (periods 2π to 2π·max_period) are made for diffusion steps in [0, 1000], so tau
    is multiplied by `scale` first: with 1000, nearby flow times still get distinct features.
    """

    def __init__(self, dim: int, frequency_dim: int = 256, scale: float = 1000.0, max_period: float = 10000.0) -> None:
        super().__init__()
        if frequency_dim % 2:
            raise ValueError(f"frequency_dim must be even (a cos and a sin per frequency), got {frequency_dim}")
        self.scale = scale
        self.mlp = MLP(frequency_dim, dim, activation=nn.SiLU, out_dim=dim)  # DiT's: Linear -> SiLU -> Linear
        self.mlp.init_weights(INIT_STD)  # DiT's: the two Linears from N(0, 0.02), biases 0
        # Geometric frequencies from 1 down to 1/max_period, computed once in float32. A plain tensor, not a buffer:
        # .half() would round a buffer (0.93057 -> 0.92969 in bf16, 0.9 rad off at tau = 1), the EMA would average it
        # (float32 drift), and it would be saved in checkpoints. It moves to the module's device on first use.
        half = frequency_dim // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        if self.freqs.device != tau.device:
            self.freqs = self.freqs.to(tau.device)  # once per device: the same values everywhere
        dtype = torch.promote_types(tau.dtype, torch.float32)  # at least float32: bf16 would blur nearby flow times
        args = (tau.to(dtype) * self.scale)[:, None] * self.freqs.to(dtype)  # (B, frequency_dim / 2)
        features = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(features.to(self.mlp.fc1.weight.dtype))  # the MLP's dtype, also in a .half() or .double() model
