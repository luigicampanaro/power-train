from pathlib import Path

import torch

from diff_style.inference.spec import PolicySpec
from diff_style.policies.flow_matching import FlowMatchingPolicy


def load_policy(path: str | Path, device: torch.device) -> tuple[FlowMatchingPolicy, PolicySpec]:
    """The policy of a Lightning checkpoint of scripts/train.py, in eval mode on device, and its sizes.

    The checkpoint's state_dict holds the EMA weights (the EMA callback saves them). The image size is the
    datamodule's: Lightning saves its hyperparameters in the checkpoint too. One read of the file: no pickled
    objects (weights_only), the optimizer state left on disk (mmap)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    policy = FlowMatchingPolicy(**checkpoint["hyper_parameters"])
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.to(device).eval(), read_spec(checkpoint)


def read_spec(checkpoint: dict) -> PolicySpec:
    policy_hparams = checkpoint["hyper_parameters"]
    net_hparams = policy_hparams["net"]
    return PolicySpec(
        num_cameras=net_hparams["num_cameras"],
        obs_horizon_1d=net_hparams["obs_horizon_1d"],
        obs_horizon_2d=net_hparams["obs_horizon_2d"],
        action_horizon=policy_hparams["action_horizon"],
        qpos_dim=policy_hparams["qpos_dim"],
        action_dim=policy_hparams["action_dim"],
        image_size=tuple(checkpoint["datamodule_hyper_parameters"]["image_size"]),
    )
