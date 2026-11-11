"""VisionEncoder: the contract between every vision backbone and the flow-matching network, checked on each call."""

import pytest
import torch
from torch import nn

from diff_style.models.vision import VisionEncoder


class FixedTokens(VisionEncoder):
    """A backbone that gives zeros of a fixed shape, whatever the frames: it can break the contract on purpose."""

    def __init__(self, shape: tuple[int, ...], dim: int = 8) -> None:
        super().__init__(dim)
        self.shape = shape

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.shape)


class FrameMergingEncoder(VisionEncoder):
    """Video encoder stand-in: each pair of consecutive frames -> one output, one token per pixel.
    (N, T, 3, H, W) -> (N, T / 2, H*W, dim): the two frames of a pair are processed together. It tests the contract
    and the shapes only: the network does not support the time offsets of such an encoder yet (see VisionEncoder)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(dim)
        self.proj = nn.Linear(2 * 3, dim)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        pairs = frames.unflatten(1, (-1, 2)).flatten(2, 3)  # (N, T / 2, 2 * 3, H, W)
        return self.proj(pairs.flatten(-2).transpose(-1, -2))  # (N, T / 2, H*W, dim)


class NestedEncoder(VisionEncoder):
    """A backbone with a nested submodule (as a wrapped model), one token per pixel: (N, T, 3, H, W) -> (N, T, H*W, 8)."""

    def __init__(self) -> None:
        super().__init__(8)
        self.model = nn.Sequential(nn.Linear(3, 8), nn.Dropout(0.5))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model(frames.flatten(-2).transpose(-1, -2))


def test_a_new_backbone_trains_and_follows_its_parents_mode():
    encoder = NestedEncoder()
    assert all(p.requires_grad for p in encoder.parameters())
    assert all(module.training for module in encoder.modules())
    encoder.eval()
    assert not any(module.training for module in encoder.modules())
    encoder.train()
    assert all(module.training for module in encoder.modules())


def test_a_backbone_must_implement_encode():
    class NoEncode(VisionEncoder):
        pass

    with pytest.raises(TypeError, match="encode"):
        NoEncode(8)


def test_tokens_that_follow_the_contract_come_out_unchanged():
    tokens = FrameMergingEncoder()(torch.rand(2, 4, 3, 5, 5))
    assert tokens.shape == (2, 2, 25, 8)  # T_out = T / 2 <= T: a video encoder merges frames


@pytest.mark.parametrize("shape", [(2, 3, 4, 4), (2, 1, 1, 4, 4), (2, 1, 4, 4, 4)],
                         ids=["no time axis", "1 channel", "4 channels"])
def test_frames_must_be_n_t_3_h_w(shape):
    with pytest.raises(ValueError, match=r"frames must be \(N, T, 3, H, W\)"):
        FixedTokens((2, 3, 5, 8))(torch.rand(shape))


@pytest.mark.parametrize("shape", [(2, 3, 8), (4, 3, 5, 8), (2, 4, 5, 8), (2, 0, 5, 8), (2, 3, 5, 16)],
                         ids=["3D", "N not kept", "T_out > T", "no output frame", "width not dim"])
def test_tokens_that_break_the_contract_raise(shape):
    """Frames (2, 3, 3, 4, 4) must give (2, T_out in 1..3, L, 8)."""
    with pytest.raises(ValueError, match=r"FixedTokens gave tokens"):
        FixedTokens(shape)(torch.rand(2, 3, 3, 4, 4))
