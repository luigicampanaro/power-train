from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    """The sizes a trained policy expects at inference, read from its checkpoint (see policy_loader.py).

    obs_horizon_1d / obs_horizon_2d: the steps of qpos and past actions / of image frames per sample.
    action_horizon: the actions per predicted chunk. qpos_dim / action_dim: the size of one qpos / action.
    image_size: (H, W) of the model input; the raw camera frames are resized to it, as in training.
    """

    num_cameras: int
    obs_horizon_1d: int
    obs_horizon_2d: int
    action_horizon: int
    qpos_dim: int
    action_dim: int
    image_size: tuple[int, int]
