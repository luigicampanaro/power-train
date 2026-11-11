"""FlowMatchingPolicy: the flow path against torchcfm, the Euler sampler against torchdiffeq, the loss, training in
every precision, torch.compile, and the Hydra config trained and reloaded from its checkpoint. A tiny stand-in
replaces DINOv3, so these tests run without it. The network's own tests are in tests/test_flow_matching_network.py."""

import io
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from diff_style.policies.flow_matching import FlowMatchingPolicy, euler, flow_path
from diff_style.utils.lr_schedulers import warmup_cosine
from references import REFERENCES
from test_episode_dataset import write_many_episodes
from test_flow_matching_network import DIM, EAGER_GLUE, C, H, K, T1, T2, RecordingBackend, make_batch

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
PIXEL_ENCODER = "test_flow_matching_network.PixelEncoder"  # the vision encoder stand-in, as a Hydra target
LR = 1e-3  # make_policy's AdamW lr
TRAINED_VISION_ENCODER = {"vision_encoder": LR}  # make_policy's submodule_lrs: its own group, at the same lr


def test_flow_path_matches_torchcfm():
    """torchcfm's path with sigma 0: x_t = (1 - t) x_0 + t x_1, and the velocity target dx_t = x_1 - x_0."""
    ref = REFERENCES["torchcfm_path"]
    x_t, dx_t = flow_path(ref["x_0"], ref["x_1"], ref["t"])
    assert torch.allclose(x_t, ref["x_t"], atol=1e-6)
    assert torch.allclose(dx_t, ref["dx_t"], atol=1e-6)


def test_euler_matches_torchdiffeq():
    """torchdiffeq's odeint with Euler steps of 0.25, on the same velocity field v(x, t) = sin(x) - t x."""
    ref = REFERENCES["torchdiffeq_euler"]
    x_1 = euler(lambda x, tau: torch.sin(x) - tau[:, None, None] * x, ref["x_init"], ref["num_steps"])
    assert torch.allclose(x_1, ref["x_1"], atol=1e-6)


def make_policy(submodule_lrs=TRAINED_VISION_ENCODER):
    """A small policy, AdamW at lr LR, no lr scheduler. submodule_lrs: as FlowMatchingPolicy's (0 = frozen)."""
    net_config = {
        "_target_": "diff_style.models.flow_matching_network.FlowMatchingNetwork",
        "vision_encoder": {"_target_": PIXEL_ENCODER},
        "perceiver": {"_target_": "diff_style.models.perceiver.Perceiver", "input_dim": 8, "num_latents": K,
                      "dim": DIM, "num_heads": 4},
        "obs_encoder": {"_target_": "diff_style.models.transformer.Transformer", "dim": DIM, "depth": 1, "num_heads": 4},
        "action_decoder": {"_target_": "diff_style.models.transformer.Transformer", "dim": DIM, "depth": 2,
                           "num_heads": 4, "cross_attention": True, "cond_dim": DIM},
        "num_cameras": C, "qpos_dim": 7, "action_dim": 6, "obs_horizon_1d": T1, "obs_horizon_2d": T2, "action_horizon": H,
    }
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(net_config, optimizer={"_target_": "torch.optim.AdamW", "lr": LR}, qpos_dim=7,
                                action_dim=6, action_horizon=H, submodule_lrs=submodule_lrs, num_inference_steps=4)
    policy.qpos_normalizer.set_stats({"mean": torch.full((7,), 2.0), "std": torch.full((7,), 3.0)})
    policy.action_normalizer.set_stats({"mean": torch.full((6,), -1.0), "std": torch.full((6,), 0.5)})
    return policy


def vision_and_rest(policy) -> tuple[list, list]:
    """The parameters of the vision encoder, and every other parameter of the policy."""
    vision = list(policy.net.vision_encoder.parameters())
    vision_ids = {id(p) for p in vision}
    return vision, [p for p in policy.parameters() if id(p) not in vision_ids]


def test_a_submodule_at_lr_0_is_frozen():
    """No gradients, eval mode whatever train() or eval() the policy gets, not in the optimizer; every other parameter
    in it, once, and in training mode with the policy."""
    policy = make_policy({"vision_encoder": 0.0})
    policy.eval().train()
    vision, rest = vision_and_rest(policy)
    assert not any(p.requires_grad for p in vision)
    assert not any(module.training for module in policy.net.vision_encoder.modules())
    assert not policy.net.vision_encoder(torch.rand(1, 1, 3, 4, 4)).requires_grad  # no graph through the encoder
    assert all(module.training for module in policy.net.action_decoder.modules())
    groups = policy.configure_optimizers()["optimizer"].param_groups
    assert len(groups) == 1 and groups[0]["lr"] == LR
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in rest]


def test_a_submodule_at_a_positive_lr_trains_in_its_own_group():
    """Two groups: every other parameter at the optimizer's lr, the vision encoder's at its own lr. Each parameter in
    exactly one group. The vision encoder trains as the rest: gradients, training mode."""
    policy = make_policy({"vision_encoder": 1e-5})
    policy.train()
    vision, rest = vision_and_rest(policy)
    assert all(p.requires_grad for p in vision)
    assert all(module.training for module in policy.net.vision_encoder.modules())
    groups = policy.configure_optimizers()["optimizer"].param_groups
    assert [group["lr"] for group in groups] == [LR, 1e-5]
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in rest]
    assert [id(p) for p in groups[1]["params"]] == [id(p) for p in vision]


def test_any_submodule_of_the_network_can_be_frozen_or_get_its_own_lr():
    """The paths are the network's, not the policy's: the policy knows none of the network's modules."""
    policy = make_policy({"perceiver": 0.0, "action_decoder.layers.0": 1e-5})
    policy.train()
    frozen, own_group = policy.net.perceiver, policy.net.action_decoder.layers[0]
    assert not any(p.requires_grad for p in frozen.parameters())
    assert not any(module.training for module in frozen.modules())
    groups = policy.configure_optimizers()["optimizer"].param_groups
    assert [group["lr"] for group in groups] == [LR, 1e-5]
    assert [id(p) for p in groups[1]["params"]] == [id(p) for p in own_group.parameters()]
    grouped = {id(p) for group in groups for p in group["params"]}
    assert grouped == {id(p) for p in policy.parameters() if p.requires_grad}


def test_without_submodule_lrs_every_parameter_trains_in_one_group():
    policy = make_policy({})
    groups = policy.configure_optimizers()["optimizer"].param_groups
    assert len(groups) == 1 and [id(p) for p in groups[0]["params"]] == [id(p) for p in policy.parameters()]


def test_a_negative_submodule_lr_is_refused():
    with pytest.raises(ValueError, match="the lr of vision_encoder must be >= 0"):
        make_policy({"vision_encoder": -1e-5})


def test_an_unknown_submodule_is_refused():
    with pytest.raises(AttributeError, match="vision_encodr"):
        make_policy({"vision_encodr": 0.0})


def test_nested_submodules_are_refused():
    """A parameter in two groups: one lr only."""
    with pytest.raises(ValueError, match="vision_encoder.proj and vision_encoder share parameters"):
        make_policy({"vision_encoder": 1e-5, "vision_encoder.proj": 0.0})


def test_a_list_in_the_optimizer_config_loads_back_with_weights_only():
    """AdamW's betas, a list in the config, reach the optimizer as a plain list, not an OmegaConf ListConfig: the
    optimizer's state goes into the checkpoint, and load_from_checkpoint(weights_only=True) refuses OmegaConf objects."""
    policy = make_policy()
    policy.hparams["optimizer"]["betas"] = [0.9, 0.95]
    buffer = io.BytesIO()
    torch.save(policy.configure_optimizers()["optimizer"].state_dict(), buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True)["param_groups"][0]["betas"] == [0.9, 0.95]


@pytest.mark.parametrize("vision_encoder_lr", [0.0, 1e-5, 1e-3])
def test_training_steps_are_adamw_with_the_vision_encoder_at_its_own_lr(vision_encoder_lr):
    """5 steps of the policy's own optimizer against torch's AdamW, built here with the two groups, from the same
    weights and the same batches and noise: the weights are bit-identical after every step. lr 0: the vision encoder
    keeps its exact initial weights, while the rest trains."""
    submodule_lrs = {"vision_encoder": vision_encoder_lr}
    policy, reference = make_policy(submodule_lrs), make_policy(submodule_lrs)
    initial_vision = [p.detach().clone() for p in vision_and_rest(policy)[0]]
    torch.nn.init.normal_(policy.net.velocity_head.weight, std=0.02)  # out of the zero init: gradients reach the encoder
    reference.load_state_dict(policy.state_dict())
    optimizer = policy.configure_optimizers()["optimizer"]
    vision, rest = vision_and_rest(reference)
    groups = [{"params": rest, "lr": LR}] + ([{"params": vision, "lr": vision_encoder_lr}] if vision_encoder_lr else [])
    reference_optimizer = torch.optim.AdamW(groups)
    batch = make_batch(b=4)
    for step in range(5):
        for model, opt in [(policy, optimizer), (reference, reference_optimizer)]:
            torch.manual_seed(step)
            opt.zero_grad()
            model.loss(batch).backward()
            opt.step()
        for (name, p), q in zip(policy.named_parameters(), reference.parameters()):
            assert torch.equal(p, q), f"step {step}: {name}"
    moved = [not torch.equal(p, p0) for p, p0 in zip(vision_and_rest(policy)[0], initial_vision)]
    assert all(moved) if vision_encoder_lr else not any(moved)


def test_the_lr_schedule_scales_both_groups_alike():
    """warmup_cosine is one factor per step, applied to each group's own lr: the vision encoder's lr keeps its ratio
    to the rest's at every step, warmup included."""
    policy = make_policy({"vision_encoder": 1e-5})
    policy.hparams["lr_scheduler"] = {"_target_": "diff_style.utils.lr_schedulers.warmup_cosine", "warmup_steps": 2}
    policy.trainer = SimpleNamespace(estimated_stepping_batches=10)  # the only part of the Trainer it reads
    config = policy.configure_optimizers()
    optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
    for step in range(12):
        factor = scheduler.lr_lambdas[0](step)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([LR * factor, 1e-5 * factor])
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("override, frozen", [([], True), (["model.submodule_lrs.vision_encoder=1e-5"], False)])
def test_the_config_freezes_the_vision_encoder_by_default(override, frozen):
    policy = hydra.utils.instantiate(small_train_config(["output_dir=unused", *override]).model)
    policy.trainer = SimpleNamespace(estimated_stepping_batches=10)
    assert all(p.requires_grad != frozen for p in policy.net.vision_encoder.parameters())
    assert len(policy.configure_optimizers()["optimizer"].param_groups) == (1 if frozen else 2)


def test_the_config_rounds_every_swiglu_hidden_width_up_to_a_multiple_of_256():
    """The MLP factory of the Perceiver and of both transformers: model_dim 512 gives 1536, not an odd 1365."""
    cfg = small_train_config(["output_dir=unused"])
    for module in ["perceiver", "obs_encoder", "action_decoder"]:
        assert hydra.utils.instantiate(cfg.model.net[module].mlp)(512, int(8 / 3 * 512)).w1.out_features == 1536


def test_the_policy_gives_its_sizes_to_the_network():
    """qpos_dim, action_dim and action_horizon belong to the policy (its normalizers, the shape of its samples): the
    network config interpolates them, so one override changes the two."""
    cfg = small_train_config(["output_dir=unused", "model.qpos_dim=5", "model.action_dim=6", "action_horizon=8"])
    policy = hydra.utils.instantiate(cfg.model)
    batch = {"images": torch.rand(2, 2, 2, 3, 4, 4), "qpos": torch.randn(2, 2, 5), "past_actions": torch.randn(2, 2, 6)}
    assert policy.predict_actions(batch).shape == (2, 8, 6)


def test_initial_loss_is_the_mse_of_the_normalized_velocity_target():
    """The velocity head starts at 0, so the loss is mean((x1 - x0)^2), x1 = the normalized actions."""
    policy, batch = make_policy(), make_batch()
    torch.manual_seed(1)
    loss = policy.loss(batch)
    torch.manual_seed(1)
    noise = torch.randn(3, H, 6)  # the first draw of loss(), the same seed
    assert loss.item() == pytest.approx(((batch["actions"] + 1) / 0.5 - noise).pow(2).mean().item(), rel=1e-5)


def test_the_exact_velocity_has_zero_loss():
    """x_tau = (1 - tau) x0 + tau x1 and the target x1 - x0 = (x1 - x_tau) / (1 - tau): a network that
    returns this exact velocity has loss 0. The same velocity drives the sampler test below."""
    policy, batch = make_policy(), make_batch()
    target = (batch["actions"] + 1) / 0.5  # the normalized actions, x1
    policy.net.forward = lambda x, tau, obs_tokens: (target - x) / (1 - tau[:, None, None])
    assert policy.loss(batch).item() < 1e-8


def test_sampler_integrates_from_noise_at_tau_0_to_actions_at_tau_1():
    """With the exact velocity toward a target, (target - x) / (1 - tau), Euler steps land on the target."""
    policy = make_policy()
    target = torch.randn(3, H, 6)
    taus = []

    def velocity(x, tau, obs_tokens):
        taus.append(tau[0].item())
        return (target - x) / (1 - tau[:, None, None])

    policy.net.forward = velocity
    actions = policy.predict_actions(make_batch())
    assert taus == [0.0, 0.25, 0.5, 0.75]  # num_inference_steps=4
    assert torch.allclose(actions, target * 0.5 - 1.0, atol=1e-5)  # denormalized


def test_training_updates_every_parameter_and_lowers_the_loss():
    policy, batch = make_policy(), make_batch(b=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    initial = {name: p.detach().clone() for name, p in policy.named_parameters()}
    losses = []
    for step in range(150):
        torch.manual_seed(step % 4)  # a few fixed noise draws, so the batch can be fitted
        loss = policy.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    unchanged = [name for name, p in policy.named_parameters() if torch.equal(p, initial[name])]
    assert unchanged == []  # every module is used, none is cut off by the zero inits
    assert sum(losses[-10:]) / 10 < 0.5 * sum(losses[:10]) / 10


def small_train_config(overrides: list[str]):
    """train.yaml with the overrides, made small: width 32, 4 heads, 4 Perceiver latents, the PixelEncoder stand-in."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose("train", overrides=["model_dim=32", "model.net.perceiver.num_latents=4", *overrides])
    cfg.model.net.vision_encoder = {"_target_": PIXEL_ENCODER}
    cfg.model.net.perceiver.input_dim = 8
    for module in ["perceiver", "obs_encoder", "action_decoder"]:
        OmegaConf.update(cfg.model.net, f"{module}.num_heads", 4)
    return cfg


def run_logger(run_dir: Path) -> TensorBoardLogger:
    """The logger of scripts/train.py, which the lr monitor of train.yaml needs: TensorBoard, in the run folder."""
    return TensorBoardLogger(run_dir, name="", version="")


def train_config_batch(b=2):
    """A batch for train.yaml's shapes: 2 cameras, 2 past steps, 7 joints, 16 actions; 4x4 images."""
    return {"images": torch.rand(b, 2, 2, 3, 4, 4), "qpos": torch.randn(b, 2, 7),
            "past_actions": torch.randn(b, 2, 7), "actions": torch.randn(b, 16, 7)}


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize("precision, accelerator, weights", [
    ("64-true", "cpu", torch.float64), ("bf16-mixed", "cpu", torch.float32),  # 32-true: the checkpoint test below
    pytest.param("16-mixed", "gpu", torch.float32, marks=GPU), pytest.param("bf16-true", "gpu", torch.bfloat16, marks=GPU),
])
def test_trains_validates_and_predicts_in_every_lightning_precision(precision, accelerator, weights, tmp_path):
    """Lightning converts the weights and casts the batch, but before the datamodule makes the images float:
    the policy casts them. Validation samples actions, so prediction runs too. Not 16-true: pure float16 can't train (AdamW's eps 1e-8 is 0 in float16, and the
    norms' backward overflows without loss scaling); 16-mixed trains in float16 with float32 weights."""
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=4", "datamodule.num_workers=0", "image_size=[8,8]",
        f"trainer.accelerator={accelerator}", f"trainer.precision={precision}", "trainer.max_steps=2",
        "trainer.max_epochs=1", "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false",
        "~callbacks.progress_bar", "+trainer.enable_model_summary=false", "model.lr_scheduler.warmup_steps=1",
    ])
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    policy = hydra.utils.instantiate(cfg.model)
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=run_logger(tmp_path / "run"),
                                      default_root_dir=tmp_path / "run")
    trainer.fit(policy, datamodule=datamodule)
    assert {p.dtype for p in policy.parameters()} == {weights}
    assert all(p.isfinite().all() for p in policy.parameters())
    metrics = trainer.validate(policy, datamodule=datamodule, verbose=False)[0]
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


class LrPerOptimizerStep(pl.Callback):
    def __init__(self) -> None:
        self.lrs = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self.lrs.append(optimizer.param_groups[0]["lr"])


def test_lightning_steps_the_schedule_once_per_optimizer_step_across_epochs(tmp_path):
    """The real config in a real fit: 3 batches per epoch and 2 batches per optimizer step, so a schedule stepped
    per epoch or per batch, or a total_steps in batches, gives other lrs. The schedule spans exactly max_steps."""
    max_steps, warmup_steps = 6, 2
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=8", "datamodule.num_workers=0", "image_size=[8,8]", "trainer.accelerator=cpu",
        "trainer.precision=32-true", f"trainer.max_steps={max_steps}", "+trainer.accumulate_grad_batches=2",
        "+trainer.limit_val_batches=0", "+trainer.enable_progress_bar=false", "+trainer.enable_model_summary=false",
        f"model.lr_scheduler.warmup_steps={warmup_steps}",
    ])
    probe = LrPerOptimizerStep()
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[probe], logger=False, default_root_dir=tmp_path / "run")
    trainer.fit(hydra.utils.instantiate(cfg.model), datamodule=hydra.utils.instantiate(cfg.datamodule))
    assert trainer.num_training_batches == 3  # at most 2 optimizer steps per epoch: the 6 steps span 3 epochs
    reference = warmup_cosine(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=cfg.model.optimizer.lr),
                              warmup_steps, max_steps)
    expected = [reference.lr_lambdas[0](step) * cfg.model.optimizer.lr for step in range(max_steps)]
    assert probe.lrs == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("dtype, tolerance", [(torch.bfloat16, 2e-2), (torch.float16, 1e-2)])
def test_policy_runs_in_its_weights_dtype_outside_lightning(dtype, tolerance):
    """policy.to(dtype) without Lightning (e.g. for deployment): the velocity is float32's within the dtype's
    precision, and the loss and the sampled actions are computed in that dtype, the policy making its noise and
    flow times from its own tensors. The flow times reach the network in float32: bf16 would round them."""
    policy, batch = make_policy(), make_batch(b=4)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for _ in range(20):  # out of the zero init, so the network's output counts
        loss = policy.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    noisy_actions, tau = torch.randn(4, H, 6), torch.rand(4)
    with torch.no_grad():
        full = policy.net(noisy_actions, tau, policy.encode_observations(batch))
        policy, batch = policy.to(dtype), {k: v.to(dtype) for k, v in batch.items()}
        velocity = policy.net(noisy_actions.to(dtype), tau, policy.encode_observations(batch))
    assert ((velocity.float() - full).norm() / full.norm()).item() < tolerance
    taus = []
    policy.net.register_forward_pre_hook(lambda module, args: taus.append(args[1].dtype))
    assert policy.predict_actions(batch).dtype == dtype and set(taus) == {torch.float32}
    if dtype == torch.float16:  # torch 2.5 has no bf16 MSE on CPU (bf16 training runs on GPU, tested above)
        assert policy.loss(batch).dtype == dtype


def test_policy_compile_compiles_its_network():
    """The policy has no forward of its own: policy.compile() compiles the network, which knows its entry points."""
    torch._dynamo.reset()  # a new cache, so no earlier test uses up the recompile limit
    policy, backend = make_policy(), RecordingBackend()
    policy.compile(backend=backend)
    policy.loss(make_batch())
    assert backend.eager_parameters(policy.net) == EAGER_GLUE


@GPU
def test_compiled_policy_matches_eager_and_its_copies_run_their_own_weights():
    """policy.compile(): the same state_dict keys, loss, gradients and sampled actions as eager (float32), and a
    deepcopy, as the EMA callback makes, runs its own weights, not the original's."""
    eager, compiled = make_policy(), make_policy()
    compiled.compile()
    with torch.no_grad():
        for p in eager.parameters():
            p.add_(torch.randn_like(p) * 0.02)  # out of the zero inits, so every gradient counts
    compiled.load_state_dict(eager.state_dict())  # strict: the keys match
    eager, compiled, batch = eager.cuda(), compiled.cuda(), {k: v.cuda() for k, v in make_batch().items()}
    for policy in [eager, compiled]:
        torch.manual_seed(1)
        policy.loss(batch).backward()
    for (name, p), q in zip(eager.named_parameters(), compiled.parameters()):
        assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-6), name
    torch.manual_seed(2)
    actions = eager.predict_actions(batch)
    torch.manual_seed(2)
    assert torch.allclose(compiled.predict_actions(batch), actions, atol=1e-5)
    copy = deepcopy(compiled)
    nn.init.zeros_(copy.net.velocity_head.weight)
    nn.init.zeros_(copy.net.velocity_head.bias)
    noisy_actions, tau = torch.randn(3, H, 6, device="cuda"), torch.rand(3, device="cuda")
    assert copy.net(noisy_actions, tau, copy.encode_observations(batch)).abs().max() == 0


def test_hydra_config_trains_and_reloads_from_the_checkpoint(tmp_path):
    """train.yaml as scripts/train.py uses it (vision encoder swapped for the stand-in), a few steps on
    CPU, then FlowMatchingPolicy.load_from_checkpoint gives the same predictions."""
    (tmp_path / "data").mkdir()
    cfg = small_train_config([
        f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}", f"output_dir={tmp_path / 'run'}",
        "datamodule.batch_size=4", "datamodule.num_workers=0", "image_size=[8,8]",
        "trainer.accelerator=cpu", "trainer.precision=32", "trainer.max_steps=3", "trainer.max_epochs=1",
        "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false", "~callbacks.progress_bar",
        "model.lr_scheduler.warmup_steps=1",
        *[f"model.net.{module}.dropout=0.1" for module in ["obs_encoder", "action_decoder"]],
        *[f"model.net.{module}.attention.dropout=0.1" for module in ["perceiver", "obs_encoder", "action_decoder"]],
        "model.net.action_decoder.causal=true",
        "model.net.obs_encoder.attention.scale=0.5",
    ])

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    policy = hydra.utils.instantiate(cfg.model)
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=run_logger(tmp_path / "run"),
                                      default_root_dir=tmp_path / "run")
    trainer.fit(policy, datamodule=datamodule)
    assert trainer.global_step == 3
    assert torch.allclose(policy.action_normalizer.mean, datamodule.stats["actions"]["mean"])

    path = tmp_path / "policy.ckpt"
    trainer.save_checkpoint(path)
    loaded = FlowMatchingPolicy.load_from_checkpoint(path, weights_only=True)  # plain config, no pickled objects
    batch = datamodule.on_after_batch_transfer(next(iter(datamodule.val_dataloader())), 0)
    for model in [policy, loaded]:
        model.eval()  # no dropout: the same predictions for the same noise
        torch.manual_seed(0)
        model.predicted = model.predict_actions(batch)
    assert torch.equal(policy.predicted, loaded.predicted)
    assert policy.predicted.shape == (len(batch["actions"]), cfg.action_horizon, 7)


@pytest.mark.parametrize("group, option", [(path.parent.name, path.stem) for group in ["attention", "norm", "mlp"]
                                           for path in sorted((CONFIG_DIR / "model" / group).glob("*.yaml"))])
def test_every_layer_config_builds_a_working_policy(group, option):
    """Each file of configs/model/{attention,norm,mlp}, swapped into every slot of that group. norm/none only into the
    Perceiver's feature norm: the vision features may come normalized, a transformer's stream needs its norms."""
    slots = {"attention": ["perceiver.attention", "obs_encoder.attention", "action_decoder.attention"],
             "norm": ["perceiver.feature_norm", "perceiver.norm", "obs_encoder.norm", "action_decoder.norm"],
             "mlp": ["perceiver.mlp", "obs_encoder.mlp", "action_decoder.mlp"]}[group]
    if (group, option) == ("norm", "none"):
        slots = ["perceiver.feature_norm"]
    cfg = small_train_config(["output_dir=unused"] + [f"model/{group}@model.net.{slot}={option}" for slot in slots])
    policy = hydra.utils.instantiate(cfg.model)
    batch = train_config_batch()
    assert torch.isfinite(policy.loss(batch))
    assert policy.predict_actions(batch).shape == (2, 16, 7)
