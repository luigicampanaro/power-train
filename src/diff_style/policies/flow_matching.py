"""Flow-matching policy (Lipman et al. 2023; rectified flow, Liu et al. 2023) over normalized action chunks.

Training: actions x1 and noise x0 ~ N(0, I) are mixed at a random flow time tau ~ U(0, 1),
x_tau = (1 - tau) x0 + tau x1, and the network learns the velocity x1 - x0 (MSE). This is the
"conditional optimal transport" path of Lipman et al., tested against torchcfm's ConditionalFlowMatcher.
Inference: start from noise at tau = 0 and take num_inference_steps Euler steps to tau = 1.

The policy builds its network from the `net` config itself, so the checkpoint holds the whole
config: FlowMatchingPolicy.load_from_checkpoint(path) needs nothing else.
"""

from collections.abc import Callable
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn

from diff_style.models.normalizer import Normalizer


def flow_path(noise: torch.Tensor, actions: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The point at flow time tau on the straight line from noise (tau = 0) to actions (tau = 1),
    and the velocity along it. noise, actions: (B, ...); tau: (B,)."""
    tau = tau.view(-1, *[1] * (actions.ndim - 1))  # (B, 1, ...), broadcasts over the other dimensions
    return (1 - tau) * noise + tau * actions, actions - noise


def euler(velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor,
          num_steps: int) -> torch.Tensor:
    """Integrate dx/dtau = velocity(x, tau) from tau = 0 to 1 in num_steps equal Euler steps.
    x: (B, ...) at tau = 0; velocity gets tau as (B,)."""
    dt = 1.0 / num_steps
    tau_dtype = torch.promote_types(x.dtype, torch.float32)  # at least float32: bf16 would round the flow times
    for i in range(num_steps):
        x = x + dt * velocity(x, torch.full((len(x),), i * dt, device=x.device, dtype=tau_dtype))
    return x


class FlowMatchingPolicy(pl.LightningModule):
    def __init__(self, net: dict, optimizer: dict, qpos_dim: int, action_dim: int, action_horizon: int,
                 lr_scheduler: dict | None = None, submodule_lrs: dict[str, float] | None = None,
                 num_inference_steps: int = 10) -> None:
        """net, optimizer, lr_scheduler: Hydra configs (plain dicts with _target_). The optimizer gets the
        parameter groups (param_groups); the scheduler gets the optimizer and the total number of training steps,
        and scales the lr of every group by the same factor.
        qpos_dim, action_dim, action_horizon: the sizes of the task, for the normalizers and the sampled actions
        (B, action_horizon, action_dim). The network config gets the same values (configs/model: interpolations).
        submodule_lrs: {path in the network: lr}, e.g. {"vision_encoder": 1e-5}. Each submodule gets its own group at
        its own lr. 0 freezes it: no gradients, eval mode (which train() keeps), not in the optimizer, not averaged by
        the EMA. The paths come from the config: the policy knows none of the network's modules."""
        super().__init__()
        self.save_hyperparameters()
        self.net = hydra.utils.instantiate(net)
        self.submodule_lrs = dict(submodule_lrs or {})
        check_submodule_lrs(self.net, self.submodule_lrs)
        for module in self.frozen_submodules():
            module.requires_grad_(False).eval()
        # Stats from the train split, copied in by setup(); saved in the checkpoint with the weights
        self.qpos_normalizer = Normalizer(qpos_dim)
        self.action_normalizer = Normalizer(action_dim)  # also normalizes the past actions

    def compile(self, *args, **kwargs) -> None:
        """torch.compile the network, which knows its own entry points (about 2x faster; the first steps take minutes
        to compile). The policy has no forward of its own. A runtime choice, not a hyperparameter: the entry point
        calls it, and a policy loaded from a checkpoint runs eager. Arguments: as torch.compile's."""
        self.net.compile(*args, **kwargs)

    def train(self, mode: bool = True) -> "FlowMatchingPolicy":
        """nn.Module.train, but the frozen submodules stay in eval mode: no dropout, and no other training-mode
        behavior (DINOv3's random RoPE rescale). Lightning 2.6 does not call it in fit (it restores the mode of each
        module around validation), but other code can."""
        super().train(mode)
        for module in self.frozen_submodules():
            module.eval()
        return self

    def frozen_submodules(self) -> list[nn.Module]:
        return [self.net.get_submodule(path) for path, lr in self.submodule_lrs.items() if lr == 0]

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # trainer.fit puts the datamodule on the trainer, but the Trainer class declares no such attribute
            stats = getattr(self.trainer, "datamodule").stats
            self.qpos_normalizer.set_stats(stats["qpos"])
            self.action_normalizer.set_stats(stats["actions"])

    def encode_observations(self, batch: dict) -> torch.Tensor:
        # Lightning casts the batch to the precision before the datamodule turns the uint8 images into floats:
        # give them the dtype of the rest of the batch (float32 with mixed precision: nothing to do)
        images = batch["images"].to(batch["qpos"].dtype)
        return self.net.encode_observations(images, self.qpos_normalizer.normalize(batch["qpos"]),
                                            self.action_normalizer.normalize(batch["past_actions"]))

    def loss(self, batch: dict) -> torch.Tensor:
        actions = self.action_normalizer.normalize(batch["actions"])  # (B, H, Da), x1
        noise = torch.randn_like(actions)  # x0
        tau = torch.rand(len(actions), device=actions.device, dtype=actions.dtype)
        noisy_actions, target = flow_path(noise, actions, tau)
        return F.mse_loss(self.net(noisy_actions, tau, self.encode_observations(batch)), target)

    @torch.no_grad()
    def sample_normalized_actions(self, batch: dict) -> torch.Tensor:
        """Euler integration from noise (tau = 0) to actions (tau = 1): (B, H, Da), normalized."""
        obs_tokens = self.encode_observations(batch)  # once, reused by every step
        noise = torch.randn(len(obs_tokens), self.hparams["action_horizon"], self.hparams["action_dim"],
                            device=obs_tokens.device, dtype=obs_tokens.dtype)
        return euler(lambda x, tau: self.net(x, tau, obs_tokens), noise, self.hparams["num_inference_steps"])

    def predict_actions(self, batch: dict) -> torch.Tensor:
        """The next action_horizon actions in raw units: (B, H, Da). batch: images, qpos, past_actions."""
        return self.action_normalizer.denormalize(self.sample_normalized_actions(batch))

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _evaluate(self, batch: dict, stage: str) -> None:
        self.log(f"{stage}/loss", self.loss(batch), sync_dist=True)
        # Error of the sampled actions, in normalized units so every joint counts the same
        error = F.mse_loss(self.sample_normalized_actions(batch), self.action_normalizer.normalize(batch["actions"]))
        self.log(f"{stage}/action_mse", error, sync_dist=True, prog_bar=True)
        # The x of these points in TensorBoard: the optimizer steps done, as in the checkpoint names. Lightning would
        # use the index of the last training batch, one less. A logged "step" is the x axis, not a curve.
        self.log("step", float(self.global_step), sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._evaluate(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._evaluate(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # _convert_: plain lists (AdamW's betas), not ListConfig: the optimizer's state goes into the checkpoint
        optimizer = hydra.utils.instantiate(self.hparams["optimizer"], _partial_=True, _convert_="all")(
            self.param_groups())
        if self.hparams["lr_scheduler"] is None:
            return {"optimizer": optimizer}
        scheduler = hydra.utils.instantiate(self.hparams["lr_scheduler"], optimizer=optimizer,
                                            total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def param_groups(self) -> list[dict[str, Any]]:
        """Every trainable parameter at the optimizer's lr, except those of submodule_lrs: one group per submodule,
        at its own lr. No group for a frozen submodule."""
        own_lr = [(trainable_parameters(self.net.get_submodule(path)), lr)
                  for path, lr in self.submodule_lrs.items() if lr > 0]
        grouped = {id(p) for params, _ in own_lr for p in params}
        rest = [p for p in trainable_parameters(self) if id(p) not in grouped]
        return [{"params": rest}, *({"params": params, "lr": lr} for params, lr in own_lr)]


def check_submodule_lrs(net: nn.Module, submodule_lrs: dict[str, float]) -> None:
    """Each path is a submodule of the network (get_submodule raises if not), each lr is >= 0, and no two submodules
    share a parameter: one lr per parameter."""
    owners = {}  # id(parameter) -> path
    for path, lr in submodule_lrs.items():
        if lr < 0:
            raise ValueError(f"the lr of {path} must be >= 0 (0 = frozen), got {lr}")
        for p in net.get_submodule(path).parameters():
            if id(p) in owners:
                raise ValueError(f"{path} and {owners[id(p)]} share parameters: give each parameter one lr")
            owners[id(p)] = path


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]
