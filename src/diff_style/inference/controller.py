"""One control step on the robot: the observation of the step in, the action to send out.

    observation -> ObservationHistory -> policy (a chunk of action_horizon actions) -> TemporalEnsemble -> action
The action sent is recorded in the history: the past action of the next steps, as in the training windows.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from diff_style.inference.observation_history import ObservationHistory
from diff_style.inference.spec import PolicySpec
from diff_style.inference.temporal_ensemble import TemporalEnsemble


@dataclass(frozen=True)
class ControlSettings:
    """spec: the policy's sizes. ensemble_offset, ensemble_decay: TemporalEnsemble's offset and decay."""

    spec: PolicySpec
    ensemble_offset: int = 0
    ensemble_decay: float = 0.05


class Controller:
    def __init__(self, predict_chunk: Callable[[dict[str, torch.Tensor]], torch.Tensor],
                 settings: ControlSettings) -> None:
        """predict_chunk: ObservationHistory.batch -> (action_horizon, action_dim) raw actions, e.g. a GraphedPolicy."""
        self.predict_chunk = predict_chunk
        self.history = ObservationHistory(settings.spec)
        self.ensemble = TemporalEnsemble(settings.spec.action_horizon, settings.spec.action_dim,
                                         settings.ensemble_offset, settings.ensemble_decay)  # on the CPU

    def reset(self) -> None:
        """Forget the past observations, actions and chunks: at the start of every episode."""
        self.history.reset()
        self.ensemble.reset()

    def step(self, frames: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """frames: (C, H, W, 3) uint8 RGB, in the training camera order. qpos: (Dq,) raw. Returns the (Da,) action
        to send now, raw units, float32."""
        self.history.add(frames, qpos)
        chunk = self.predict_chunk(self.history.batch()).float().cpu()  # (H, Da): a few numbers, the ensemble's device
        action = self.ensemble(chunk)
        self.history.record_action(action)
        return action.numpy()
