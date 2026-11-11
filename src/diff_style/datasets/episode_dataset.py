"""Sliding-window dataset over ACT-style HDF5 episodes.

One sample per timestep `t` of every episode: the last `obs_horizon_2d` image frames, the last
`obs_horizon_1d` non-image observation steps (qpos and past actions), and the next
`action_horizon` actions, in raw units. Images are much heavier than 1D observations, so the two
histories can have different lengths. Windows running past the episode edges repeat the
first/last step, so the policy learns to hold still at the end. Past actions are the rows before
`t`, because row `t` holds the next action for observation `t`. Rows before the episode start
are zeros, not copies of `action[0]`, because `action[0]` is a future target at `t=0`.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def list_episodes(dataset_dir: str | Path) -> list[Path]:
    """Paths of all `episode_<number>.hdf5` files in `dataset_dir`, in numeric order.

    `int()` ignores zero padding: episode_2 < episode_00010 < episode_11.
    """
    paths = Path(dataset_dir).glob("episode_*.hdf5")
    return sorted(paths, key=lambda p: (int(p.stem.removeprefix("episode_")), p.name))


def _timesteps(start: int, stop: int, last: int) -> list[int]:
    """Indices from `start` to `stop` (exclusive), clamped to `0 ... last`."""
    return [min(max(i, 0), last) for i in range(start, stop)]


class EpisodeDataset(Dataset):
    """Sliding-window dataset over ACT-style HDF5 episodes.

    `dataset_dir`: task folder with one `episode_<number>.hdf5` per episode.
    `episodes`: positions in that sorted list to use (e.g. the train split); None = all.
    `obs_horizon_1d`: past steps of non-image observations (qpos and past actions).
    `obs_horizon_2d`: past image frames; images are heavy, so keep this history short.

    Returns dicts with (raw units, no normalization)
        images:       (obs_horizon_2d, num_cams, 3, H, W) uint8
        qpos:         (obs_horizon_1d, D)
        past_actions: (obs_horizon_1d, D), zeros before the episode start
        actions:      (action_horizon, D), starting at t
    """

    def __init__(self, dataset_dir: str | Path, obs_horizon_1d: int = 2, obs_horizon_2d: int = 1,
                 action_horizon: int = 16, episodes: list[int] | None = None) -> None:
        if min(obs_horizon_1d, obs_horizon_2d, action_horizon) < 1:
            raise ValueError(
                f"horizons must be at least 1, got 1D={obs_horizon_1d}, 2D={obs_horizon_2d}, "
                f"action={action_horizon}"
            )
        self.paths = list_episodes(dataset_dir)
        if episodes is not None:
            self.paths = [self.paths[ep] for ep in episodes]
        self.obs_horizon_1d = obs_horizon_1d
        self.obs_horizon_2d = obs_horizon_2d
        self.action_horizon = action_horizon

        # qpos/actions are small: keep them in memory. Images are read on demand.
        self.qpos, self.actions = [], []  # one (T, D) float32 tensor per episode
        for ep in range(len(self.paths)):
            with self._open(ep) as f:
                self.qpos.append(torch.from_numpy(f["observations/qpos"][()]).float())
                self.actions.append(torch.from_numpy(f["action"][()]).float())
                self.camera_names = sorted(f["observations/images"])
        # actions[t] follows observation t. The zero row is the action before t=0, so
        # padded_actions[t] is actions[t-1], and past actions are [t-obs_horizon_1d+1 : t+1].
        self.padded_actions = [
            torch.cat([torch.zeros(1, actions.shape[1]), actions]) for actions in self.actions
        ]
        # (episode, t) pairs as one numpy array: a Python list of tuples this long gets
        # copied into every DataLoader worker
        self.index = np.array(
            [(ep, t) for ep in range(len(self.paths)) for t in range(len(self.actions[ep]))]
        )

    def _open(self, ep: int) -> h5py.File:
        # Read-only, so no HDF5 file locks: on NFS/Lustre they fail or serialize readers
        # across ranks/nodes. Never write to the files (e.g. value targets) while training.
        return h5py.File(self.paths[ep], "r", locking=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ep, t = self.index[index].tolist()
        last = len(self.actions[ep]) - 1
        obs_1d_start = t - self.obs_horizon_1d + 1
        obs_1d_ts = _timesteps(obs_1d_start, t + 1, last)
        obs_2d_ts = _timesteps(t - self.obs_horizon_2d + 1, t + 1, last)
        past_action_ts = _timesteps(obs_1d_start, t + 1, len(self.padded_actions[ep]) - 1)
        act_ts = _timesteps(t, t + self.action_horizon, last)

        # Opening is ~0.1 ms vs ~1-3 ms to read the frames, so no handle cache: scales
        # to any number of episodes and keeps the dataset picklable for workers
        with self._open(ep) as f:
            images = [[f[f"observations/images/{cam}"][i] for cam in self.camera_names] for i in obs_2d_ts]

        return {
            "images": torch.from_numpy(np.array(images)).permute(0, 1, 4, 2, 3),  # (T, K, H, W, C) -> (T, K, C, H, W)
            "qpos": self.qpos[ep][obs_1d_ts],
            "past_actions": self.padded_actions[ep][past_action_ts],
            "actions": self.actions[ep][act_ts],
        }
