import torch
from torch import nn


class Normalizer(nn.Module):
    """Per-dimension standardization with stats stored as buffers (saved in checkpoints, moved with .to())."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Placeholders with the right shape, so load_state_dict can fill them when loading a checkpoint
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_stats(self, stats: dict[str, torch.Tensor]) -> None:
        # copy_ writes into the existing buffers; `self.mean = ...` would replace them
        self.mean.copy_(stats["mean"])
        self.std.copy_(stats["std"])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
