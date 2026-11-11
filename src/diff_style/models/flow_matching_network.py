"""Velocity network of the flow-matching policy.

    images (B, T2, C, 3, H, W) -> vision encoder -> Perceiver on each image -> K tokens per image
    qpos (B, T1, Dq), past_actions (B, T1, Da) -> one token per timestep each
    every observation token += source embedding (camera c, qpos or past action) + time embedding (its step)
    -> observation encoder (self-attention) -> obs tokens: computed once per sample
    noisy actions (B, H, Da) + time embedding -> action decoder (self-attention, cross-attention to
    the obs tokens, adaLN-Zero from the flow time tau) -> velocity (B, H, Da)

Time offsets are relative to the current step t, and shared by all modalities: the image and the
qpos of step t-1 both have offset -1, the action of step t+2 has offset 2. past_actions[:, k] is the
action issued at step t - T1 + k, one step before the qpos of the same slot (see EpisodeDataset).
The flow time tau has its own sinusoidal embedding, unrelated to these offsets.

Dropout of whole inputs, in training only (each a NullTokenDropout, which puts a learned null token in
place of a sample's tokens, before the source and time embeddings are added, so the network still
knows which input is missing):
    camera_dropout: each camera of each sample, with all its frames (modality dropout)
    qpos_dropout, past_action_dropout: all the qpos / past-action tokens of a sample, so the policy can't
        rely on copying the robot's state or its last action
    condition_dropout: all the obs tokens out of the obs encoder: the unconditional model of
        classifier-free guidance (net.condition_dropout(obs_tokens, drop=...) at sampling time)
embedding_dropout: nn.Dropout on the tokens into the obs encoder and into the action decoder, after the
embeddings are added (GPT-2's embd_pdrop, ViT's pos_drop, Diffusion Policy's p_drop_emb).
"""

import warnings

import torch
from torch import nn

from diff_style.models.layers import INIT_STD, NullTokenDropout, TimestepEmbedding, init_linear
from diff_style.models.perceiver import Perceiver
from diff_style.models.transformer import Transformer
from diff_style.models.vision import VisionEncoder


class FlowMatchingNetwork(nn.Module):
    def __init__(self, vision_encoder: VisionEncoder, perceiver: Perceiver, obs_encoder: Transformer,
                 action_decoder: Transformer, num_cameras: int, qpos_dim: int, action_dim: int,
                 obs_horizon_1d: int, obs_horizon_2d: int, action_horizon: int, embedding_dropout: float = 0.0,
                 camera_dropout: float = 0.0, qpos_dropout: float = 0.0, past_action_dropout: float = 0.0,
                 condition_dropout: float = 0.0) -> None:
        super().__init__()
        check_vision_parts(vision_encoder, perceiver)
        dim = action_decoder.dim
        self.vision_encoder = vision_encoder
        self.perceiver = perceiver
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.num_cameras = num_cameras

        self.qpos_proj = nn.Linear(qpos_dim, out_features=dim)
        self.past_action_proj = nn.Linear(action_dim, dim)
        # Tables used as tensors, not looked up as a vocabulary, so plain Parameters (as ViT's pos_embed)
        # Rows 0..C-1: the cameras, row C: qpos, row C+1: past actions
        self.source_embedding = nn.Parameter(torch.empty(num_cameras + 2, dim))
        # Offsets from -max_past (the oldest past action or image) to action_horizon - 1 (the last action)
        self.max_past = max(obs_horizon_1d, obs_horizon_2d - 1)
        self.time_embedding = nn.Parameter(torch.empty(self.max_past + action_horizon, dim))
        for table in [self.source_embedding, self.time_embedding]:
            nn.init.trunc_normal_(table, std=INIT_STD)  # small tags on the content tokens, as ViT's pos_embed

        self.camera_dropout = NullTokenDropout(dim, camera_dropout)
        self.qpos_dropout = NullTokenDropout(dim, qpos_dropout)
        self.past_action_dropout = NullTokenDropout(dim, past_action_dropout)
        self.condition_dropout = NullTokenDropout(dim, condition_dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.action_proj = nn.Linear(action_dim, dim)
        # The tokens start at the scale of the embeddings and of the Perceiver's (ViT's patch embedding init)
        for proj in [self.qpos_proj, self.past_action_proj, self.action_proj]:
            init_linear(proj, INIT_STD)
        self.tau_embedding = TimestepEmbedding(dim)
        self.velocity_head = nn.Linear(dim, action_dim)
        # The network starts by predicting velocity 0 (DiT zero-initialises its output layer too)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def time(self, start: int, stop: int) -> torch.Tensor:
        """Time embeddings of the offsets start, ..., stop - 1: (stop - start, dim)."""
        first_row, stop_row = start + self.max_past, stop + self.max_past
        if first_row < 0 or stop_row > len(self.time_embedding):  # a slice would wrap or cut without an error
            raise ValueError(f"offsets {start} .. {stop - 1} are outside the time table: {-self.max_past} .. "
                             f"{len(self.time_embedding) - self.max_past - 1}")
        return self.time_embedding[first_row:stop_row]

    def encode_observations(self, images: torch.Tensor, qpos: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        """images: (B, T2, C, 3, H, W) float in [0, 1]. qpos: (B, T1, Dq), past_actions: (B, T1, Da), normalized.
        Returns the obs tokens (B, C * T2' * K + 2 * T1, dim), T2' = frames out of the vision encoder.

        CAUTION: the time offsets of the image tokens are correct only for a vision encoder with one output per frame
        (T2' = T2, as DINOv3). Video encoders that merge frames (T2' < T2) are not supported yet: this method warns,
        and their image tokens get incorrect offsets (see VisionEncoder)."""
        b, _, num_cams = images.shape[:3]
        num_1d_steps = qpos.shape[1]  # T1: the steps of qpos and of past_actions
        if num_cams != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {num_cams}")
        if past_actions.shape[1] != num_1d_steps:
            raise ValueError(f"qpos and past_actions must have the same number of steps, got {num_1d_steps} and "
                             f"{past_actions.shape[1]}")

        # Every camera through the same encoder, each with its own frames: (B, T2, C, ...) -> (B*C, T2, ...)
        features = self.vision_encoder(images.transpose(1, 2).flatten(0, 1))  # (B*C, T2', L, vision dim)
        num_frames = features.shape[1]  # T2'
        warn_if_frames_are_merged(num_frames, images.shape[1])
        image_tokens = self.perceiver(features.flatten(0, 1))  # (B*C*T2', K, dim): K tokens per image
        image_tokens = self.camera_dropout(image_tokens.unflatten(0, (b * num_cams, num_frames)))  # each (sample, camera)
        image_tokens = (image_tokens.unflatten(0, (b, num_cams))  # (B, C, T2', K, dim)
                        + self.source_embedding[:num_cams, None, None]  # (C, 1, 1, dim)
                        + self.time(-(num_frames - 1), 1)[:, None])  # (T2', 1, dim): offsets -(T2'-1) .. 0

        qpos_tokens = (self.qpos_dropout(self.qpos_proj(qpos)) + self.source_embedding[num_cams]
                       + self.time(-(num_1d_steps - 1), 1))  # offsets -(T1-1) .. 0
        past_action_tokens = (self.past_action_dropout(self.past_action_proj(past_actions))
                              + self.source_embedding[num_cams + 1]
                              + self.time(-num_1d_steps, 0))  # offsets -T1 .. -1

        tokens = torch.cat([image_tokens.flatten(1, 3), qpos_tokens, past_action_tokens], dim=1)
        return self.condition_dropout(self.obs_encoder(self.embedding_dropout(tokens)))

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        """noisy_actions: (B, H, Da), tau: (B,) in [0, 1], obs_tokens: from encode_observations. -> velocity (B, H, Da)."""
        x = self.embedding_dropout(self.action_proj(noisy_actions) + self.time(0, noisy_actions.shape[1]))  # offsets 0 .. H-1
        x = self.action_decoder(x, context=obs_tokens, cond=self.tau_embedding(tau))
        return self.velocity_head(x)

    def compile(self, *args, **kwargs) -> None:
        """Module.compile of forward (the action decoder is in its graph) and, one by one, of the modules of
        encode_observations: Module.compile covers __call__ only, and encode_observations is a plain method. Its glue
        stays eager. Module.compile, not torch.compile: the state_dict keys stay the same, and a deepcopy (the EMA's)
        runs eager, with its own weights. Arguments: as torch.compile's."""
        super().compile(*args, **kwargs)
        for module in [self.vision_encoder, self.perceiver, self.obs_encoder]:
            module.compile(*args, **kwargs)


def check_vision_parts(vision_encoder: nn.Module, perceiver: Perceiver) -> None:
    """The backbone promises the vision contract (a VisionEncoder) and gives the token width the Perceiver takes."""
    if not isinstance(vision_encoder, VisionEncoder):
        raise TypeError(f"vision_encoder must be a VisionEncoder (diff_style.models.vision), got "
                        f"{type(vision_encoder).__name__}")
    if vision_encoder.dim != perceiver.input_dim:
        raise ValueError(f"the vision encoder gives {vision_encoder.dim}-dim tokens, the Perceiver takes "
                         f"{perceiver.input_dim}")


def warn_if_frames_are_merged(num_output_frames: int, num_input_frames: int) -> None:
    """The image time offsets -(T2' - 1) .. 0 are one step apart: correct only for one output per input frame."""
    if num_output_frames != num_input_frames:
        warnings.warn(f"the vision encoder gave {num_output_frames} outputs for {num_input_frames} frames: video "
                      f"encoders that merge frames are not supported yet, and the image tokens get incorrect time "
                      f"offsets (see VisionEncoder)", UserWarning, stacklevel=3)
