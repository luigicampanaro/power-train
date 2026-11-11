"""The contract between a vision backbone and the flow-matching network.

Every backbone is a VisionEncoder: it implements encode(frames) and gives its token width to __init__. forward checks
the frames and the tokens on every call, so a backbone that breaks the contract fails at its own output, with a
clear error, and not later inside the network.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class VisionEncoder(nn.Module, ABC):
    """frames (N, T, 3, H, W) float in [0, 1] -> tokens (N, T_out, L, dim).

    N: independent videos, batch x cameras. The encoder never mixes them: each camera has its own sequence.
    T: the frames of one video, in time order. An image encoder encodes each frame on its own, so T_out = T. The
        contract permits 1 <= T_out <= T, but see the caution below.
    L: the tokens of one output frame, any number. Each encoder gives its tokens their spatial position and
        normalizes its own pixels.
    dim: the token width, read by the configs (the Perceiver's input_dim) and checked on every output.
    A backbone trains as any module (it follows its parent's mode). The policy freezes it (submodule_lrs).

    CAUTION: video encoders that merge frames (T_out < T) are not supported yet. The network gives the T_out
    outputs the consecutive time offsets -(T_out - 1) .. 0, one step apart. These offsets are correct only for an
    encoder with one output per input frame (T_out = T, as DINOv3). An encoder that merges frames gives outputs
    that are more than one step apart (2 steps for pairs of frames), so their offsets are incorrect: they no longer
    agree with the offsets of the qpos and past actions. The network warns, but it does not stop. To support such an
    encoder, the encoder must give its temporal stride, and the network must use the offsets -(T_out - 1) * stride
    .. 0. The current method cannot be extended to video models without this change.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self._check_frames(frames)
        tokens = self.encode(frames)
        self._check_tokens(tokens, frames)
        return tokens

    @abstractmethod
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (N, T, 3, H, W) float in [0, 1] -> tokens (N, T_out, L, dim): the backbone's own computation."""

    @staticmethod
    def _check_frames(frames: torch.Tensor) -> None:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"frames must be (N, T, 3, H, W), got {tuple(frames.shape)}")

    def _check_tokens(self, tokens: torch.Tensor, frames: torch.Tensor) -> None:
        num_videos, num_frames = frames.shape[:2]
        if (tokens.ndim != 4 or tokens.shape[0] != num_videos or not 1 <= tokens.shape[1] <= num_frames
                or tokens.shape[3] != self.dim):
            raise ValueError(f"{type(self).__name__} gave tokens {tuple(tokens.shape)} for frames "
                             f"{tuple(frames.shape)}: expected (N={num_videos}, T_out in 1..{num_frames}, L, "
                             f"dim={self.dim})")
