"""Controller: each step gives the policy the history with the new observation, sends the ensembled action of the
predicted chunks, and records it as the past action of the next steps. A stand-in policy replaces the real one."""

import numpy as np
import torch

from diff_style.inference.controller import ControlSettings, Controller
from diff_style.inference.spec import PolicySpec
from diff_style.inference.temporal_ensemble import TemporalEnsemble

SPEC = PolicySpec(num_cameras=2, obs_horizon_1d=2, obs_horizon_2d=1, action_horizon=4, qpos_dim=7, action_dim=3,
                  image_size=(2, 2))
SETTINGS = ControlSettings(SPEC, ensemble_offset=1, ensemble_decay=0.5)
STEPS = 6


class RecordingPolicy:
    """Returns the chunk of each step, (H, Da) float32, and records the batches it got."""

    def __init__(self, chunks: torch.Tensor) -> None:
        self.chunks = iter(chunks)
        self.batches = []

    def __call__(self, batch: dict) -> torch.Tensor:
        self.batches.append(batch)
        return next(self.chunks)


def chunks() -> torch.Tensor:
    return torch.randn(2 * STEPS, SPEC.action_horizon, SPEC.action_dim, generator=torch.Generator().manual_seed(0))


def observation(step: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((2, 2, 2, 3), step, dtype=np.uint8), np.full(7, float(step))


def run(controller: Controller, steps: int) -> list[np.ndarray]:
    return [controller.step(*observation(t)) for t in range(steps)]


def test_each_step_sends_the_ensembled_action_of_the_chunks():
    actions = run(Controller(RecordingPolicy(chunks()), SETTINGS), STEPS)
    ensemble = TemporalEnsemble(SPEC.action_horizon, SPEC.action_dim, offset=1, decay=0.5)
    expected = [ensemble(chunk).numpy() for chunk in chunks()[:STEPS]]
    np.testing.assert_array_equal(np.stack(actions), np.stack(expected))
    assert actions[0].dtype == np.float32


def test_the_policy_gets_the_new_observation_and_the_actions_sent_before():
    policy = RecordingPolicy(chunks())
    actions = run(Controller(policy, SETTINGS), STEPS)
    for t, batch in enumerate(policy.batches):
        assert batch["qpos"][0, -1, 0] == t and batch["images"][0, -1].unique().tolist() == [t]
        sent_before = [np.zeros(SPEC.action_dim, dtype=np.float32)] * 2 + actions[:t]
        np.testing.assert_array_equal(batch["past_actions"][0].numpy(), np.stack(sent_before[-2:]))


def test_reset_starts_a_new_episode():
    controller = Controller(RecordingPolicy(chunks()), SETTINGS)
    run(controller, STEPS)
    controller.reset()
    after_reset = run(controller, STEPS)
    fresh = run(Controller(RecordingPolicy(chunks()[STEPS:]), SETTINGS), STEPS)
    np.testing.assert_array_equal(np.stack(after_reset), np.stack(fresh))
