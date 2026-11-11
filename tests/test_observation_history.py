"""ObservationHistory against EpisodeDataset: fed an episode one step at a time, it gives the policy the windows the
dataset gave it in training, with the same padding at the episode start (the first qpos and frame repeated, zero past
actions). The actions recorded are the episode's, as a robot that sends the demonstrated actions."""

import h5py
import numpy as np
import pytest
import torch

from diff_style.datasets.episode_dataset import EpisodeDataset
from diff_style.inference.observation_history import ObservationHistory
from diff_style.inference.spec import PolicySpec
from test_episode_dataset import write_episodes

T1, T2 = 3, 2  # longer than the start of the episode: the padding counts


def spec(obs_horizon_1d=T1, obs_horizon_2d=T2) -> PolicySpec:
    return PolicySpec(num_cameras=2, obs_horizon_1d=obs_horizon_1d, obs_horizon_2d=obs_horizon_2d,
                      action_horizon=4, qpos_dim=7, action_dim=7, image_size=(2, 4))


def read_steps(path) -> list[dict]:
    """Each step of the episode as the robot gives it: the frames (C, H, W, 3) uint8 in the dataset's camera order,
    the qpos (Dq,) and the action sent."""
    with h5py.File(path, "r") as f:
        cameras = sorted(f["observations/images"])
        frames = np.stack([f[f"observations/images/{cam}"][()] for cam in cameras], axis=1)  # (T, C, H, W, 3)
        return [{"frames": frames[t], "qpos": f["observations/qpos"][t], "action": f["action"][t]}
                for t in range(len(frames))]


@pytest.mark.parametrize("obs_horizon_1d, obs_horizon_2d", [(T1, T2), (1, 1)])
def test_the_windows_are_the_dataset_samples_of_each_step(tmp_path, obs_horizon_1d, obs_horizon_2d):
    dataset = EpisodeDataset(write_episodes(tmp_path), obs_horizon_1d, obs_horizon_2d, episodes=[1])  # 4 steps
    history = ObservationHistory(spec(obs_horizon_1d, obs_horizon_2d))
    for t, step in enumerate(read_steps(dataset.paths[0])):
        history.add(step["frames"], step["qpos"])
        batch, sample = history.batch(), dataset[t]
        for key in ["images", "qpos", "past_actions"]:
            assert batch[key].dtype == sample[key].dtype, key
            torch.testing.assert_close(batch[key], sample[key].unsqueeze(0), rtol=0, atol=0, msg=key)
        history.record_action(step["action"])


def test_reset_starts_a_new_episode(tmp_path):
    steps = read_steps(write_episodes(tmp_path) / "episode_00010.hdf5")
    history, fresh = ObservationHistory(spec()), ObservationHistory(spec())
    for step in steps:
        history.add(step["frames"], step["qpos"])
        history.record_action(step["action"])
    history.reset()
    for h in [history, fresh]:
        h.add(steps[0]["frames"], steps[0]["qpos"])
    for key, value in fresh.batch().items():
        torch.testing.assert_close(history.batch()[key], value, rtol=0, atol=0, msg=key)


def test_a_batch_before_the_first_observation_raises():
    with pytest.raises(RuntimeError, match="observation"):
        ObservationHistory(spec()).batch()


def test_frames_of_another_number_of_cameras_raise():
    with pytest.raises(ValueError, match="cameras"):
        ObservationHistory(spec()).add(np.zeros((3, 2, 4, 3), dtype=np.uint8), np.zeros(7))
