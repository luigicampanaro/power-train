import torch
from torch import nn

from diff_style.models.layers.init import INIT_STD


class NullTokenDropout(nn.Module):
    """In training, replaces all the tokens of a sample with one learned null token, with probability p
    per sample: the label/caption dropout of DiT and PixArt-α for classifier-free guidance. With one
    instance per input (a camera, the qpos), it is modality dropout. Kept samples are unchanged (no
    1 / (1 - p) scaling): a dropped sample holds no information, not a scaled-down one.

    The null token exists only with p > 0: an unused parameter would stop DDP.
    """

    def __init__(self, dim: int, p: float = 0.0) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"dropout probability {p} is not in [0, 1]")
        self.p = p
        self.null = nn.Parameter(torch.empty(dim)) if p > 0 else None
        if self.null is not None:
            nn.init.trunc_normal_(self.null, std=INIT_STD)  # a learned embedding, as the Perceiver's latents

    def forward(self, x: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, ..., dim). drop: (B,) bool, the samples to replace, also outside training (e.g. the
        unconditional pass of classifier-free guidance, as DiT's force_drop_ids); None = random in training."""
        if drop is None:
            if not self.training or self.p == 0:
                return x
            drop = torch.rand(len(x), device=x.device) < self.p
        elif drop.shape != (len(x),):
            raise ValueError(f"drop has shape {tuple(drop.shape)}, not ({len(x)},): one value per sample")
        if self.null is None:
            raise ValueError("NullTokenDropout with p = 0 has no null token to drop to")
        return torch.where(drop.view(-1, *[1] * (x.ndim - 1)), self.null.to(x.dtype), x)

    def extra_repr(self) -> str:
        return f"p={self.p}"
