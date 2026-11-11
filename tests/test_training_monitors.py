"""The lr monitor of train.yaml in a real fit, logged to TensorBoard as scripts/train.py does: the lr of each
optimizer step."""

import hydra
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from test_episode_dataset import write_many_episodes
from test_flow_matching import run_logger, small_train_config

MAX_STEPS, WARMUP_STEPS = 4, 2


def test_the_config_logs_the_lr_of_each_step_to_tensorboard(tmp_path):
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=4", "datamodule.num_workers=0", "image_size=[8,8]",
        "trainer.accelerator=cpu", "trainer.precision=32", f"trainer.max_steps={MAX_STEPS}",
        "trainer.log_every_n_steps=1", "+trainer.limit_train_batches=2", "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false", "~callbacks.progress_bar", f"model.lr_scheduler.warmup_steps={WARMUP_STEPS}"])
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=run_logger(tmp_path / "run"),
                                      default_root_dir=tmp_path / "run")
    trainer.fit(hydra.utils.instantiate(cfg.model), datamodule=hydra.utils.instantiate(cfg.datamodule))

    events = EventAccumulator(str(tmp_path / "run"))
    events.Reload()
    lr = cfg.model.optimizer.lr
    # warmup_cosine: linear from 0 over 2 steps, then cosine to 0 at step 4
    assert [event.value for event in events.Scalars("lr-AdamW")] == pytest.approx([0.0, lr / 2, lr, lr / 2])
