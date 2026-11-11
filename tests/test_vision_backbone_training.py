"""The vision backbone in the real training pipeline, on the GPU: train.yaml with the real DINOv3 at 224 x 224,
bf16-mixed, the EMA callback, gradient clipping and warmup_cosine, a few steps. A probe records, at every
optimizer step, the backbone's mode, its gradients and the lr of each parameter group.

lr 0 (model.submodule_lrs.vision_encoder): frozen. No gradients, not in the optimizer, eval mode, bit-identical
weights at every step, after fit (the EMA copied in) and in the checkpoint.
lr > 0: trained. Its own parameter group at that lr, on the schedule of the rest; finite gradients
on every parameter that the forward uses; training mode in training, eval mode in validation; every weight moves.

The compiled variants take minutes to compile: `uv run pytest -m slow`.
"""

import hydra
import pytest
import pytorch_lightning as pl
import torch

from diff_style.policies.flow_matching import FlowMatchingPolicy
from test_dinov3_encoder import DINOV3_CACHED
from test_episode_dataset import write_many_episodes
from test_flow_matching import CONFIG_DIR, run_logger

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
              pytest.mark.skipif(not DINOV3_CACHED, reason="DINOv3 is not in the Hugging Face cache")]
LR, VISION_LR, STEPS = 1e-4, 1e-5, 4  # train.yaml's lr; warmup_steps=1: the lr is 0 at step 0
MASK_TOKEN = "embeddings.mask_token"  # used only by DINOv3's masked pretraining: never trained
COMPILE = [False, pytest.param(True, marks=pytest.mark.slow)]


def backbone_of(policy: FlowMatchingPolicy) -> torch.nn.Module:
    return policy.net.vision_encoder.model  # DINOv3ViTModel


class Probe(pl.Callback):
    """Per optimizer step (after backward and clipping): the backbone's mode and gradients, the group lrs, which
    backbone parameters the optimizer has. Per batch: whether the live backbone weights still equal the initial."""

    def __init__(self) -> None:
        self.steps, self.unchanged_after_batch, self.validation_modes = [], [], []
        self.initial, self.last_live = {}, {}

    def on_train_start(self, trainer, pl_module) -> None:
        self.initial = {name: p.detach().clone() for name, p in backbone_of(pl_module).named_parameters()}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        backbone = dict(backbone_of(pl_module).named_parameters())
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.steps.append({
            "training": backbone_of(pl_module).training,
            "lrs": [group["lr"] for group in optimizer.param_groups],
            "in optimizer": {name for name, p in backbone.items() if id(p) in in_optimizer},
            "no grad": {name for name, p in backbone.items() if p.grad is None},
            "finite nonzero grad": {name for name, p in backbone.items()
                                    if p.grad is not None and p.grad.isfinite().all() and p.grad.any()},
        })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.unchanged_after_batch.append({name for name, p in backbone_of(pl_module).named_parameters()
                                           if torch.equal(p, self.initial[name])})
        self.last_live = {name: p.detach().clone() for name, p in backbone_of(pl_module).named_parameters()}

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self.validation_modes.append(backbone_of(pl_module).training)


def train(tmp_path, vision_encoder_lr: float, compile: bool):
    (tmp_path / "data").mkdir()
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose("train", overrides=[
            f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
            "datamodule.batch_size=4", "datamodule.num_workers=0", f"trainer.max_steps={STEPS}",
            "trainer.max_epochs=100", "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false",
            "~callbacks.progress_bar", "+trainer.enable_model_summary=false", "model.lr_scheduler.warmup_steps=1",
            f"model.submodule_lrs.vision_encoder={vision_encoder_lr}"])
    assert (cfg.trainer.accelerator, cfg.trainer.precision, cfg.image_size) == ("gpu", "bf16-mixed", [224, 224])
    assert cfg.model.optimizer.lr == LR
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    policy = hydra.utils.instantiate(cfg.model)
    if compile:  # as scripts/train.py
        policy.compile()
    probe = Probe()
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()] + [probe]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=run_logger(tmp_path / "run"),
                                      default_root_dir=tmp_path / "run")
    trainer.fit(policy, datamodule=datamodule)
    assert trainer.global_step == STEPS and len(probe.steps) == STEPS
    trainer.save_checkpoint(tmp_path / "policy.ckpt")
    return policy, probe, tmp_path / "policy.ckpt"


def backbone_tensors(state: dict) -> dict:
    prefix = "net.vision_encoder.model."
    return {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}


@pytest.mark.parametrize("compile", COMPILE)
def test_vision_encoder_lr_0_keeps_dinov3_frozen_in_training(tmp_path, compile):
    policy, probe, checkpoint_path = train(tmp_path, vision_encoder_lr=0.0, compile=compile)
    names = set(probe.initial)
    for step in probe.steps:
        assert step["in optimizer"] == set() and len(step["lrs"]) == 1
        assert step["no grad"] == names
        assert not step["training"]
    assert all(unchanged == names for unchanged in probe.unchanged_after_batch)  # live weights, every batch
    assert probe.validation_modes and not any(probe.validation_modes)
    after_fit = dict(backbone_of(policy).named_parameters())  # the EMA's weights, copied in at the end
    assert all(torch.equal(after_fit[name].cpu(), probe.initial[name].cpu()) for name in names)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    for state in ["state_dict", "current_model_state"]:  # the EMA's weights, and the live ones
        saved = backbone_tensors(checkpoint[state])
        assert all(torch.equal(saved[name].cpu(), probe.initial[name].cpu()) for name in names), state
    loaded = FlowMatchingPolicy.load_from_checkpoint(checkpoint_path, weights_only=True)
    assert not any(p.requires_grad for p in backbone_of(loaded).parameters()) and not backbone_of(loaded).training


@pytest.mark.parametrize("compile", COMPILE)
def test_a_positive_vision_encoder_lr_trains_dinov3_at_that_lr(tmp_path, compile):
    policy, probe, checkpoint_path = train(tmp_path, vision_encoder_lr=VISION_LR, compile=compile)
    trained = set(probe.initial) - {MASK_TOKEN}
    for i, step in enumerate(probe.steps):
        assert step["in optimizer"] == trained
        lr, vision_lr = step["lrs"]
        assert vision_lr == pytest.approx(lr * VISION_LR / LR), i  # one schedule, the two lrs in their ratio
        assert step["training"], i
        assert MASK_TOKEN in step["no grad"]
    assert probe.steps[1]["lrs"] == pytest.approx([LR, VISION_LR])  # after the warmup step: the full lrs
    # The velocity head starts at 0, so no gradient reaches the backbone before it moves (steps 0 and 1)
    assert probe.steps[-1]["finite nonzero grad"] == trained
    assert probe.unchanged_after_batch[-1] == {MASK_TOKEN}  # the live weights: every trained tensor moved
    assert probe.validation_modes and not any(probe.validation_modes)
    # After fit, the EMA's weights are copied in: averaged, so neither the last live weights nor the initial ones
    after_fit = {name: p.detach().cpu() for name, p in backbone_of(policy).named_parameters()}
    assert any(not torch.equal(after_fit[name], probe.last_live[name].cpu()) for name in trained)
    assert any(not torch.equal(after_fit[name], probe.initial[name].cpu()) for name in trained)
    loaded = FlowMatchingPolicy.load_from_checkpoint(checkpoint_path, weights_only=True)
    assert {name for name, p in backbone_of(loaded).named_parameters() if p.requires_grad} == trained
    assert backbone_of(loaded).training
    assert all(torch.equal(p.cpu(), after_fit[name]) for name, p in backbone_of(loaded).named_parameters())
