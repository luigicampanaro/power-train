"""StepProgressBar in a real fit: one train bar for the full run, in optimizer steps out of the run's total, not
reset at the epoch ends; and train.yaml picks it."""

import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diff_style.callbacks.progress_bar import StepProgressBar
from test_flow_matching import small_train_config

MAX_STEPS = 5  # 2 batches per epoch: the run crosses 2 epoch ends


class TinyModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class TrainBarStates(pl.Callback):
    """(completed, total, start time, description) of the train bar after each training batch, after the bar's update
    (it comes first in the list)."""

    def __init__(self, bar: StepProgressBar) -> None:
        self.bar = bar
        self.states = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        task = self.bar.progress.tasks[self.bar.train_progress_bar_id]
        self.states.append((task.completed, task.total, task.start_time, task.description))


def test_the_train_bar_counts_the_optimizer_steps_of_the_run_across_epochs(tmp_path):
    bar = StepProgressBar()
    states = TrainBarStates(bar)
    trainer = pl.Trainer(accelerator="cpu", max_steps=MAX_STEPS, max_epochs=-1, callbacks=[bar, states], logger=False,
                         enable_checkpointing=False, default_root_dir=tmp_path)
    trainer.fit(TinyModel(), DataLoader(TensorDataset(torch.randn(4, 1)), batch_size=2))

    completed, totals, start_times, descriptions = zip(*states.states)
    assert list(completed) == list(range(1, MAX_STEPS + 1))
    assert set(totals) == {MAX_STEPS}
    assert len(set(start_times)) == 1  # no reset: the elapsed and remaining times are those of the run
    # padded to the validation bar's label: the train row does not move when the validation row shows
    assert set(descriptions) == {"Training".ljust(len(bar.validation_description))}


class TrainBarSpeedSamples(pl.Callback):
    """Rich's speed samples of the train bar (each one: the steps it counted as done at an update), after each batch."""

    def __init__(self, bar: StepProgressBar) -> None:
        self.bar = bar
        self.samples = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        task = self.bar.progress.tasks[self.bar.train_progress_bar_id]
        self.samples = [sample.completed for sample in task._progress]


def test_the_speed_and_remaining_time_count_each_step_once(tmp_path):
    """Rich computes it/s and the remaining time from what each update adds. The bar must only move forward, one
    step per optimizer step: never back to the batch of the epoch (Lightning's own update) and up again, which
    counted every jump as done steps (280,000 it/s, 0:00:01 remaining after 8 hours)."""
    bar = StepProgressBar()
    samples = TrainBarSpeedSamples(bar)
    trainer = pl.Trainer(accelerator="cpu", max_steps=MAX_STEPS, max_epochs=-1, callbacks=[bar, samples], logger=False,
                         enable_checkpointing=False, default_root_dir=tmp_path)
    trainer.fit(TinyModel(), DataLoader(TensorDataset(torch.randn(4, 1)), batch_size=2))
    assert samples.samples == [1] * MAX_STEPS


def test_the_config_picks_the_step_progress_bar(tmp_path):
    cfg = small_train_config([f"output_dir={tmp_path}"])
    trainer = pl.Trainer(accelerator="cpu", callbacks=[hydra.utils.instantiate(cfg.callbacks.progress_bar)])
    assert isinstance(trainer.progress_bar_callback, StepProgressBar)
