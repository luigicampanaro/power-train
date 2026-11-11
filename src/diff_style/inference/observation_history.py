"""The policy's input at the current step, from the observations and the actions of the steps so far.

The windows are EpisodeDataset's, so the policy sees on the robot what it saw in training:
    images:       the last obs_horizon_2d frames; before the episode start, copies of the first frame
    qpos:         the last obs_horizon_1d joint positions; before the start, copies of the first
    past_actions: the obs_horizon_1d actions sent before this step; before the start, zeros
Call add() with the observation of the step, then batch(), then record_action() with the action sent.
"""

from collections import deque

import numpy as np
import torch

from diff_style.inference.spec import PolicySpec


class ObservationHistory:
    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        """Forget the past steps: at the start of every episode."""
        self.frames: deque[torch.Tensor] = deque(maxlen=self.spec.obs_horizon_2d)  # each (C, 3, H, W) uint8
        self.qpos: deque[torch.Tensor] = deque(maxlen=self.spec.obs_horizon_1d)  # each (Dq,) float32
        self.actions: deque[torch.Tensor] = deque(maxlen=self.spec.obs_horizon_1d)  # each (Da,) float32

    def add(self, frames: np.ndarray, qpos: np.ndarray) -> None:
        """frames: (C, H, W, 3) uint8 RGB, in the training camera order. qpos: (Dq,) raw joint positions."""
        if len(frames) != self.spec.num_cameras:
            raise ValueError(f"expected frames of {self.spec.num_cameras} cameras, got shape {frames.shape}")
        self.frames.append(torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2))
        self.qpos.append(torch.as_tensor(qpos, dtype=torch.float32))

    def record_action(self, action: np.ndarray | torch.Tensor) -> None:
        """action: (Da,) the action sent at this step, raw units. The past action of the next steps."""
        self.actions.append(torch.as_tensor(action, dtype=torch.float32).cpu())

    def batch(self) -> dict[str, torch.Tensor]:
        """The policy's input, a batch of one sample: images (1, T2, C, 3, H, W) uint8, qpos (1, T1, Dq),
        past_actions (1, T1, Da), raw units."""
        if not self.qpos:
            raise RuntimeError("no observation yet: add() the observation of this step first")
        past_actions = pad_front(list(self.actions), torch.zeros(self.spec.action_dim), self.spec.obs_horizon_1d)
        return {
            "images": torch.stack(pad_front(list(self.frames), self.frames[0], self.spec.obs_horizon_2d))[None],
            "qpos": torch.stack(pad_front(list(self.qpos), self.qpos[0], self.spec.obs_horizon_1d))[None],
            "past_actions": torch.stack(past_actions)[None],
        }


def pad_front(steps: list[torch.Tensor], padding: torch.Tensor, length: int) -> list[torch.Tensor]:
    """steps, oldest first, with copies of padding in front up to length."""
    return [padding] * (length - len(steps)) + steps
