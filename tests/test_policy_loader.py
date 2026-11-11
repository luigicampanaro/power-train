"""load_policy on a real checkpoint of train.yaml (a short fit on CPU, the stand-in vision encoder): the weights
the checkpoint saved (the EMA's), in eval mode, and the sizes the policy needs at inference."""

import hydra
import pytest
import torch

from diff_style.inference.policy_loader import load_policy
from diff_style.inference.spec import PolicySpec
from test_episode_dataset import write_many_episodes
from test_flow_matching import run_logger, small_train_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The policy and a checkpoint of a 2-step fit: 3 past qpos, 2 frames, 5 actions, 8x6 images. A plain checkpoint,
    saved after the fit."""
    policy, trainer, run_dir = fit(tmp_path_factory.mktemp("run"), callbacks=[])
    path = run_dir / "policy.ckpt"
    trainer.save_checkpoint(path)
    return policy, path


@pytest.fixture(scope="module")
def ema_checkpoint(tmp_path_factory):
    """last.ckpt of train.yaml's EMA and last-checkpoint callbacks, saved by the validation at step 2, during the fit
    (as the checkpoints of a real run): the EMA weights, and the live ones beside them."""
    _, _, run_dir = fit(tmp_path_factory.mktemp("run"), callbacks=["ema", "last_checkpoint"])
    return run_dir / "checkpoints" / "last.ckpt"


def fit(tmp_path, callbacks: list[str]):
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=4", "datamodule.num_workers=0", "image_size=[8,6]", "obs_horizon_1d=3",
        "obs_horizon_2d=2", "action_horizon=5", "trainer.accelerator=cpu", "trainer.precision=32",
        "trainer.max_steps=2", "trainer.max_epochs=1", "trainer.val_check_interval=2", "+trainer.limit_val_batches=1",
        "+trainer.enable_progress_bar=false", "~callbacks.progress_bar", "model.lr_scheduler.warmup_steps=1",
    ])
    policy = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=run_logger(tmp_path / "run"),
                                      callbacks=[hydra.utils.instantiate(cfg.callbacks[name]) for name in callbacks],
                                      default_root_dir=tmp_path / "run")
    trainer.fit(policy, datamodule=hydra.utils.instantiate(cfg.datamodule))
    return policy, trainer, tmp_path / "run"


def test_the_policy_of_an_ema_checkpoint_has_the_ema_weights(ema_checkpoint):
    """The EMA callback saves the average as the state_dict, the live weights as current_model_state."""
    checkpoint = torch.load(ema_checkpoint, map_location="cpu", weights_only=True)
    loaded, _ = load_policy(ema_checkpoint, torch.device("cpu"))
    differs = []
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"][name], rtol=0, atol=0, msg=name)
        differs.append(not torch.equal(value, checkpoint["current_model_state"][name]))
    assert any(differs)  # the live weights moved away from the average: the test can tell them apart


def test_the_normalization_stats_are_those_of_the_train_split(ema_checkpoint):
    """The EMA's copy of the policy is made before the policy gets its stats (the callbacks' setup runs first); the
    stats reach the copy with each update. The saved ones must be the datamodule's (stats.pt), not 0 and 1."""
    stats = torch.load(ema_checkpoint.parent.parent / "stats.pt", weights_only=True)
    loaded, _ = load_policy(ema_checkpoint, torch.device("cpu"))
    for normalizer, key in [(loaded.qpos_normalizer, "qpos"), (loaded.action_normalizer, "actions")]:
        torch.testing.assert_close(normalizer.mean, stats[key]["mean"], rtol=0, atol=0)
        torch.testing.assert_close(normalizer.std, stats[key]["std"], rtol=0, atol=0)


def test_the_spec_has_the_sizes_of_the_training_config(trained):
    _, path = trained
    _, spec = load_policy(path, torch.device("cpu"))
    assert spec == PolicySpec(num_cameras=2, obs_horizon_1d=3, obs_horizon_2d=2, action_horizon=5, qpos_dim=7,
                              action_dim=7, image_size=(8, 6))


def test_the_loaded_policy_predicts_as_the_trained_one_in_eval_mode(trained):
    policy, path = trained
    loaded, _ = load_policy(path, torch.device("cpu"))
    assert not loaded.training
    batch = {"images": torch.rand(2, 2, 2, 3, 8, 6), "qpos": torch.randn(2, 3, 7), "past_actions": torch.randn(2, 3, 7)}
    predictions = []
    for model in [policy.eval(), loaded]:
        torch.manual_seed(0)
        predictions.append(model.predict_actions(batch))
    torch.testing.assert_close(predictions[1], predictions[0], rtol=0, atol=0)
