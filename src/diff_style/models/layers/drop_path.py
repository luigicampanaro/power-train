import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth (Huang et al. 2016), on a residual branch: in training, each sample's whole branch
    output is zeroed with probability p, so for that sample the block is skipped; the kept outputs are
    scaled by 1 / (1 - p), so the expectation is unchanged. Unlike nn.Dropout, one draw per sample, not
    per element. timm's DropPath (ViT, DINOv2)."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"drop path probability {p} is not in [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, ...), the branch output."""
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty(len(x), *[1] * (x.ndim - 1)).bernoulli_(1 - self.p)  # (B, 1, ...): one draw per sample
        return x * keep / (1 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"
