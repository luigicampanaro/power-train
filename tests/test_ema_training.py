"""TrainableEMA in the real training config: the config picks it, a frozen vision encoder stays bit-identical through
a fit and its checkpoint, and after fit the weights that train are the EMA of their live values. The averaging
itself is in tests/test_ema.py."""

import hydra
import pytorch_lightning as pl
import torch

from diff_style.callbacks.ema import TrainableEMA
from test_episode_dataset import write_many_episodes
from test_flow_matching import small_train_config


class LiveWeights(pl.Callback):
    """The live weights after each training batch, after the EMA's update (it comes first in the list)."""

    def __init__(self) -> None:
        self.steps = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.steps.append({name: p.detach().clone() for name, p in pl_module.named_parameters()})


def train(tmp_path, overrides: list[str]):
    """train.yaml (small, the pixel stand-in) with configs/callbacks/ema, 3 steps on the CPU, then a checkpoint."""
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=4", "datamodule.num_workers=0", "image_size=[8,8]", "trainer.accelerator=cpu",
        "trainer.precision=32", "trainer.max_steps=3", "trainer.max_epochs=1", "+trainer.limit_val_batches=1",
        "+trainer.enable_progress_bar=false", "model.lr_scheduler.warmup_steps=1", *overrides])
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    policy = hydra.utils.instantiate(cfg.model)
    initial = {name: p.detach().clone() for name, p in policy.named_parameters()}
    ema = hydra.utils.instantiate(cfg.callbacks.ema)
    live = LiveWeights()
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[ema, live], logger=False, default_root_dir=tmp_path / "run")
    trainer.fit(policy, datamodule=datamodule)
    trainer.save_checkpoint(tmp_path / "policy.ckpt")
    checkpoint = torch.load(tmp_path / "policy.ckpt", weights_only=True)
    return policy, initial, live.steps, checkpoint


def test_the_config_uses_trainable_ema(tmp_path):
    cfg = small_train_config(["output_dir=unused"])
    assert isinstance(hydra.utils.instantiate(cfg.callbacks.ema), TrainableEMA)


def test_a_frozen_vision_encoder_stays_bit_identical_through_training_and_the_checkpoint(tmp_path):
    """lr 0: after fit (the EMA copied in) and in the checkpoint, as the EMA and as the live weights.
    The config's decay: Lightning's own EMA, 0.999 * a + 0.001 * a, rounds a frozen value away from a."""
    policy, initial, steps, checkpoint = train(tmp_path, ["model.submodule_lrs.vision_encoder=0.0"])
    vision = [name for name in initial if name.startswith("net.vision_encoder.")]
    assert vision
    for name in vision:
        assert all(torch.equal(step[name], initial[name]) for step in steps), name
        assert torch.equal(dict(policy.named_parameters())[name], initial[name]), name
        assert torch.equal(checkpoint["state_dict"][name], initial[name]), name
        assert torch.equal(checkpoint["current_model_state"][name], initial[name]), name


def test_after_fit_the_trained_weights_are_the_ema_of_the_live_weights(tmp_path):
    """lr > 0: every parameter that trains, the vision encoder's included, ends as the EMA of its live values after
    each optimizer step, computed here (decay 0.5)."""
    policy, initial, steps, _ = train(tmp_path, ["model.submodule_lrs.vision_encoder=1e-3", "callbacks.ema.decay=0.5"])
    assert len(steps) == 3
    for name, p in policy.named_parameters():
        expected = steps[0][name]
        for step in steps[1:]:
            expected = torch.lerp(expected, step[name], 0.5)
        assert torch.equal(p, expected), name
    vision = [name for name in initial if name.startswith("net.vision_encoder.")]
    assert all(not torch.equal(steps[-1][name], initial[name]) for name in vision)  # the encoder trained
