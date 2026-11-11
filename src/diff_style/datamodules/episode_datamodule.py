"""Train/val/test DataModule over the episodes of one task folder.

The split is by episodes, so there's no overlap between train/val/test. The normalization stats come from
the train dataset only; the LightningModule copies `datamodule.stats` into its
`Normalizer` buffers, so they are saved in the checkpoint.
"""

import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from diff_style.datasets.episode_dataset import EpisodeDataset, list_episodes
from diff_style.models.preprocess import preprocess_images

# {"qpos": {"mean": (D,), "std": (D,)}, "actions": {...}}
Stats = dict[str, dict[str, torch.Tensor]]


class EpisodeDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str | Path, stats_path: str | Path, obs_horizon_1d: int = 2,
                 obs_horizon_2d: int = 1, action_horizon: int = 16, image_size: tuple[int, int] | list[int] = (224, 224),
                 train_fraction: float = 0.8, val_fraction: float = 0.1, test_fraction: float = 0.1,
                 batch_size: int = 8, num_workers: int = 2, seed: int = 0) -> None:
        super().__init__()
        if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError(f"train/val/test fractions must sum to 1, got {train_fraction}/{val_fraction}/{test_fraction}")
        # Saved hparams must stay a plain config (re-runnable, loadable with weights_only=True): str, not Path
        dataset_dir, stats_path = str(object=dataset_dir), str(stats_path)
        image_size = list(image_size)  # a list, like the YAML config gives
        self.save_hyperparameters()
        self.stats_path = Path(stats_path)

    def split_episodes(self, num_episodes: int) -> dict[str, np.ndarray]:
        """{"train": episodes, "val": episodes, "test": episodes}, sorted episode numbers.

        val/test get round(fraction * num_episodes) episodes, at least 1 if their fraction
        is > 0; train gets the rest.
        """
        n_val, n_test = [max(1, round(f * num_episodes)) if f > 0 else 0
                         for f in [self.hparams["val_fraction"], self.hparams["test_fraction"]]]
        if n_val + n_test >= num_episodes:
            raise ValueError(f"{num_episodes} episodes: no train episodes left after {n_val} val + {n_test} test")
        # Own fixed-seed generator: the same split on every rank and every run, whatever the training seed
        perm = np.random.default_rng(self.hparams["seed"]).permutation(num_episodes)
        # Cut the shuffled episodes in three: the first n_val, the next n_test, the rest
        val, test, train = np.split(perm, [n_val, n_val + n_test])
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}

    @staticmethod
    def compute_stats(dataset: EpisodeDataset) -> Stats:
        # Pass the train dataset, so nothing leaks from val/test. Raw steps, not windows: the
        # repeated edge steps of the windows would count the first/last step extra times.
        stats = {}
        for key in ["qpos", "actions"]:
            x = torch.cat(getattr(dataset, key))  # every step of every episode in the dataset
            stats[key] = {"mean": x.mean(0), "std": x.std(0).clamp(min=1e-2)}
        return stats

    def save_stats(self, stats: Stats) -> None:
        # Writes stats.pt for inspection/export (ONNX, ROS, ...). Training never reads it:
        # the model gets the stats from the datamodule and stores them in the checkpoint.
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)  # create the run folder if missing
        # Write to stats.tmp first, then rename it to stats.pt: a rename is all-or-nothing,
        # so a crash while writing never leaves a corrupt stats.pt behind
        tmp_path = self.stats_path.with_suffix(".tmp")  # <run_dir>/stats.pt -> <run_dir>/stats.tmp
        torch.save(stats, tmp_path)
        tmp_path.replace(self.stats_path)  # rename stats.tmp to the stats_path name (stats.pt)

    def setup(self, stage: str) -> None:
        num_episodes = len(list_episodes(self.hparams["dataset_dir"]))
        self.split = self.split_episodes(num_episodes)

        # One dataset per split, each reading only its own episodes
        self.train_ds, self.val_ds, self.test_ds = [
            EpisodeDataset(
                self.hparams["dataset_dir"],
                obs_horizon_1d=self.hparams["obs_horizon_1d"],
                obs_horizon_2d=self.hparams["obs_horizon_2d"],
                action_horizon=self.hparams["action_horizon"],
                episodes=self.split[name].tolist(),
            )
            for name in ["train", "val", "test"]
        ]

        # Computed on every rank (cheap, and the same everywhere); written once
        self.stats = self.compute_stats(self.train_ds)
        if self.trainer is None or self.trainer.is_global_zero:
            self.save_stats(self.stats)
            for name, ds in [("train", self.train_ds), ("val", self.val_ds), ("test", self.test_ds)]:
                print(f"{name}: {len(ds.paths)} episodes, {len(ds)} samples {[p.stem for p in ds.paths]}")

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams["batch_size"],
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            persistent_workers=self.hparams["num_workers"] > 0,
        )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        # Every stage, on the GPU, once per batch: raw uint8 frames -> float [0, 1] at image_size.
        # The loaders send uint8 (4x smaller than float). Training augmentation runs after this, in the callback.
        batch["images"] = preprocess_images(batch["images"], self.hparams["image_size"])
        return batch

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
