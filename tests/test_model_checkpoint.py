"""The validation interval and the checkpoint callbacks of train.yaml in a real fit: validation every
val_check_interval optimizer steps, across epochs; the best checkpoints by val/action_mse, and last.ckpt with the
final state, in the run folder, each one with the EMA weights and loadable with weights_only=True."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from diff_style.policies.flow_matching import FlowMatchingPolicy
from test_episode_dataset import write_many_episodes
from test_flow_matching import run_logger, small_train_config

SAVE_TOP_K, VAL_CHECK_INTERVAL, MAX_STEPS = 3, 3, 15  # 2 steps per epoch: validations cross the epoch ends
VALIDATION_STEPS = [3, 6, 9, 12, 15]


def test_the_config_validates_and_checkpoints_every_val_check_interval_steps_not_every_epoch(tmp_path):
    """last.ckpt has the final step, also when it is not among the best: Lightning 2.6's save_last updates last.ckpt
    only together with a top-k checkpoint, so a separate callback keeps it. No save at the epoch ends."""
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=8", "datamodule.num_workers=0", "image_size=[8,8]", "trainer.accelerator=cpu",
        "trainer.precision=32", f"trainer.max_steps={MAX_STEPS}", f"trainer.val_check_interval={VAL_CHECK_INTERVAL}",
        "+trainer.limit_train_batches=2", "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false",
        "model.lr_scheduler.warmup_steps=1"])
    best, last = (hydra.utils.instantiate(cfg.callbacks[name]) for name in ["best_checkpoint", "last_checkpoint"])
    assert isinstance(best, pl.callbacks.ModelCheckpoint) and isinstance(last, pl.callbacks.ModelCheckpoint)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[hydra.utils.instantiate(cfg.callbacks.ema), best, last],
                                      logger=False, default_root_dir=tmp_path / "run")
    saves = record_saves(trainer)
    trainer.fit(hydra.utils.instantiate(cfg.model), datamodule=hydra.utils.instantiate(cfg.datamodule))

    assert [step for name, step in saves if name == "last.ckpt"] == VALIDATION_STEPS
    assert "val/action_mse" in trainer.progress_bar_metrics  # the last validation's score, shown in the bar
    folder = tmp_path / "run" / "checkpoints"
    best_paths = sorted(folder.glob("best-*.ckpt"))
    assert len(best_paths) == SAVE_TOP_K
    assert {path.name for path in folder.iterdir()} == {path.name for path in best_paths} | {"last.ckpt"}
    scores = {str(path): score.item() for path, score in best.best_k_models.items()}
    assert set(scores) == {str(path) for path in best_paths}
    assert best.best_model_score.item() == min(scores.values())  # lower is better
    assert {torch.load(path, weights_only=True)["global_step"] for path in best_paths} <= set(VALIDATION_STEPS)
    assert torch.load(folder / "last.ckpt", weights_only=True)["global_step"] == MAX_STEPS
    for path in [*best_paths, folder / "last.ckpt"]:
        assert "current_model_state" in torch.load(path, weights_only=True)  # the EMA's average is the state_dict
        FlowMatchingPolicy.load_from_checkpoint(path, weights_only=True)


def test_tensorboard_puts_each_validation_at_the_step_of_its_checkpoint(tmp_path):
    """Lightning logs at the index of the last training batch, one less than the optimizer steps done: without a
    logged step, the validation after step 3 would show at 2 in TensorBoard, next to best-step=0000003.ckpt."""
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=8", "datamodule.num_workers=0", "image_size=[8,8]", "trainer.accelerator=cpu",
        "trainer.precision=32", f"trainer.max_steps={MAX_STEPS}", f"trainer.val_check_interval={VAL_CHECK_INTERVAL}",
        "+trainer.limit_train_batches=2", "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false",
        "model.lr_scheduler.warmup_steps=1"])
    callbacks = [hydra.utils.instantiate(cfg.callbacks[name]) for name in ["ema", "best_checkpoint", "last_checkpoint"]]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=run_logger(tmp_path / "run"),
                                      default_root_dir=tmp_path / "run")
    trainer.fit(hydra.utils.instantiate(cfg.model), datamodule=hydra.utils.instantiate(cfg.datamodule))

    events = EventAccumulator(str(tmp_path / "run"))
    events.Reload()
    for tag in ["val/action_mse", "val/loss"]:
        assert [event.step for event in events.Scalars(tag)] == VALIDATION_STEPS
    checkpoint_steps = {int(path.stem.removeprefix("best-step=")) for path in (tmp_path / "run" / "checkpoints").glob("best-*")}
    assert checkpoint_steps <= set(VALIDATION_STEPS) and len(checkpoint_steps) == SAVE_TOP_K
    assert "step" not in events.Tags()["scalars"]  # the x axis, not a curve


def record_saves(trainer: pl.Trainer) -> list[tuple[str, int]]:
    """Wraps trainer.save_checkpoint to record (file name, global_step) of each save, in order."""
    saves = []
    save_checkpoint = trainer.save_checkpoint

    def recording_save_checkpoint(filepath, *args, **kwargs):
        saves.append((Path(filepath).name, trainer.global_step))
        save_checkpoint(filepath, *args, **kwargs)

    trainer.save_checkpoint = recording_save_checkpoint
    return saves
