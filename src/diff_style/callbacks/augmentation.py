"""PyTorch Lightning callback that augments the train batch on the GPU."""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch

from diff_style.augmentations import Augmentor


class AugmentationCallback(pl.Callback):
    """Augment ``batch[image_key]`` before each training step. Other stages are not changed.

    The images arrive as float [0, 1] at the model input size: Lightning runs
    ``EpisodeDataModule.on_after_batch_transfer`` before this hook.

    The augmentation seed of a batch depends only on (seed, global rank, epoch, batch index).
    A rerun with the same config and the same number of GPUs draws exactly the same
    augmentations, and each GPU draws different ones.
    """

    def __init__(self, augmentor: Augmentor, seed: int, image_key: str = "images") -> None:
        self.augmentor = augmentor
        self.seed = seed
        self.image_key = image_key

    def batch_seed(self, rank: int, epoch: int, batch_idx: int) -> int:
        # SeedSequence mixes the four numbers into one 64-bit seed: different batches get unrelated seeds
        return int(np.random.SeedSequence([self.seed, rank, epoch, batch_idx]).generate_state(1, dtype=np.uint64)[0])

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        images = batch[self.image_key]
        seed = self.batch_seed(trainer.global_rank, trainer.current_epoch, batch_idx)
        generator = torch.Generator(device=images.device).manual_seed(seed)
        batch[self.image_key] = self.augmentor(images, generator=generator)
