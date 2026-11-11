"""Lightning's Rich progress bar, with the train bar in optimizer steps for the full run, not in batches per epoch."""

from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import RichProgressBar


class StepProgressBar(RichProgressBar):
    """One train bar for the full run: global_step out of the run's optimizer steps (max_steps). It is not reset at
    the epoch ends, so its elapsed and remaining times and its speed are those of the run."""

    @property
    def total_train_batches(self) -> int | float:
        return self.trainer.estimated_stepping_batches

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.train_progress_bar_id is None:
            super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any,
                           batch_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._update(self.train_progress_bar_id, trainer.global_step)

    def _get_train_description(self, current_epoch: int) -> str:
        # Padded to the validation bar's label, as Lightning does: the train row does not move when that row shows
        return "Training".ljust(len(self.validation_description))
