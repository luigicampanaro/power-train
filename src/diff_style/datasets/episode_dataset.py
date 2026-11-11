"""Sliding-window dataset over ACT-style HDF5 episodes.

One sample per timestep `t` of every episode: the last `obs_horizon_2d` image frames, the last
`obs_horizon_1d` non-image observation steps (qpos and past actions), and the next
`action_horizon` actions, in raw units. Images are much heavier than 1D observations, so the two
histories can have different lengths. Windows running past the episode edges repeat the
first/last step, so the policy learns to hold still at the end. Past actions are the rows before
`t`, because row `t` holds the next action for observation `t`. Rows before the episode start
are zeros, not copies of `action[0]`, because `action[0]` is a future target at `t=0`.
"""

import os
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

        # qpos/actions are small: keep them in memory. Images are read on demand, straight from the file bytes
        # (see _read_frames): where each camera's frames start, and the shape of one frame, per episode.
        self.qpos, self.actions = [], []  # one (T, D) float32 tensor per episode
        self.camera_names: list[str] = []
        image_offsets, frame_shapes = [], []
        for ep in range(len(self.paths)):
            with self._open(ep) as f:
                self.qpos.append(torch.from_numpy(f["observations/qpos"][()]).float())
                self.actions.append(torch.from_numpy(f["action"][()]).float())
                camera_names = sorted(f["observations/images"])
                if ep > 0 and camera_names != self.camera_names:
                    raise ValueError(f"{self.paths[ep]}: cameras {camera_names}, but {self.paths[0].name} has "
                                     f"{self.camera_names}: every episode needs the same cameras")
                self.camera_names = camera_names
                cameras = [f[f"observations/images/{cam}"] for cam in camera_names]
                offsets, frame_shape = self._image_layout(self.paths[ep], cameras, len(self.actions[ep]))
                image_offsets.append(offsets)
                frame_shapes.append(frame_shape)
        # Plain int64 arrays, not lists of Python ints: like `index`, copied into every DataLoader worker
        self.image_offsets = np.array(image_offsets, dtype=np.int64).reshape(len(self.paths), len(self.camera_names))
        self.frame_shapes = np.array(frame_shapes, dtype=np.int64).reshape(len(self.paths), 3)  # (H, W, C) per episode
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

    @staticmethod
    def _image_layout(path: Path, cameras: list[h5py.Dataset], num_steps: int) -> tuple[list[int], tuple[int, ...]]:
        """Where each camera's frames start in the file (bytes), and the (H, W, C) of one frame.

        The frames must be one plain block of uint8 bytes per camera, frame after frame (HDF5's default when
        written as f[name] = array): then frame i is at offset + i * frame_bytes, and _read_frames reads it
        without HDF5. Anything else (chunked, compressed, filtered, external or not yet written data) can't be
        read that way: it raises here, at the start, instead of returning wrong frames.
        """
        offsets, shapes = [], set()
        for camera in cameras:
            plist = camera.id.get_create_plist()
            problems = [
                problem for problem, bad in [
                    ("not uint8", camera.dtype != np.uint8),
                    ("not (T, H, W, C)", camera.ndim != 4),
                    ("not contiguous (chunked or compact)", plist.get_layout() != h5py.h5d.CONTIGUOUS),
                    ("compressed or filtered", plist.get_nfilters() != 0),
                    ("stored in external files", plist.get_external_count() != 0),
                    ("not fully written", camera.id.get_storage_size() != camera.nbytes or camera.id.get_offset() is None),
                ] if bad
            ]
            if not problems and len(camera) < num_steps:
                problems.append(f"{len(camera)} frames for {num_steps} steps")
            if problems:
                raise ValueError(f"{path}: {camera.name}: {', '.join(problems)}. The dataset reads the frames as "
                                 "raw bytes: write them uncompressed, unchunked, uint8 (T, H, W, C), one per step")
            offsets.append(camera.id.get_offset())  # from the start of the file, any userblock included
            shapes.add(camera.shape[1:])
        if len(shapes) > 1:
            raise ValueError(f"{path}: the cameras have different frame shapes {sorted(shapes)}")
        return offsets, shapes.pop() if shapes else (0, 0, 0)

    def _read_frames(self, ep: int, timesteps: list[int]) -> np.ndarray:
        """Frames `timesteps` of every camera of episode `ep`: (T, K, H, W, C) uint8, read straight from the file.

        No HDF5 here: h5py 3.15 / HDF5 1.14 keep ~100-600 bytes for good at every h5py.File(), f["path"] lookup
        or open dataset, ~1.2 KB per sample when opened per sample: GBs per DataLoader worker over a long run
        (the OOM of 2026-11-11: ~2.5 GB in each of the 8 workers after ~90k steps). Plain open/pread/close keep
        nothing, hold no file open between samples (any number of episodes), and have no state to share across
        fork/spawn, ranks or nodes. The same bytes HDF5 would read: _image_layout checked the layout.
        """
        height, width, channels = self.frame_shapes[ep].tolist()
        frame_bytes = height * width * channels
        frames = np.empty((len(timesteps), len(self.camera_names), height, width, channels), dtype=np.uint8)
        fd = os.open(self.paths[ep], os.O_RDONLY)
        try:
            for i, t in enumerate(timesteps):
                for k, offset in enumerate(self.image_offsets[ep].tolist()):
                    # Straight into the output array, no extra copy
                    read = os.preadv(fd, [memoryview(frames[i, k]).cast("B")], offset + t * frame_bytes)
                    if read != frame_bytes:
                        raise OSError(f"{self.paths[ep]}: read {read} of {frame_bytes} bytes of frame {t} of "
                                      f"{self.camera_names[k]}: the file changed since the dataset was made")
        finally:
            os.close(fd)
        return frames

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

        images = self._read_frames(ep, obs_2d_ts)

        return {
            "images": torch.from_numpy(images).permute(0, 1, 4, 2, 3),  # (T, K, H, W, C) -> (T, K, C, H, W)
            "qpos": self.qpos[ep][obs_1d_ts],
            "past_actions": self.padded_actions[ep][past_action_ts],
            "actions": self.actions[ep][act_ts],
        }
