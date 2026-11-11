"""Rotary position embedding (RoPE, Su et al. 2021), with the layout of torchtitan's Llama 3:
dimensions (0, 1), (2, 3), ... of each head are rotated together."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotates each query/key by an angle proportional to its position, so the attention score
    q·k depends only on the distance between the two tokens."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE rotates pairs of dimensions: head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, num_heads, head_dim). positions: (L,), default 0, 1, ..., L-1."""
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        # One frequency per pair of dimensions, from 1 down to ~1/theta. Computed here, in at least float32,
        # rather than stored in a buffer that model.half() would round
        dtype = torch.promote_types(x.dtype, torch.float32)
        inv_freq = 1.0 / self.theta ** (torch.arange(0, self.head_dim, 2, device=x.device).to(dtype) / self.head_dim)
        angles = positions.to(dtype)[:, None] * inv_freq  # (L, head_dim / 2)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]  # (L, 1, head_dim / 2): the same for every head
        x0, x1 = x.to(dtype).unflatten(-1, (-1, 2)).unbind(-1)  # even and odd dimensions, (B, L, heads, head_dim / 2)
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)  # 2D rotation of each pair
        return rotated.flatten(-2).type_as(x)
