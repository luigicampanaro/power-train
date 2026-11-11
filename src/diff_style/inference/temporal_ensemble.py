"""Temporal ensemble of action chunks (ACT, Zhao et al. 2023), proleptic (PTE, Park et al. 2024, arXiv 2410.16981).

The policy runs at every control step t and predicts a chunk of H actions, a_{t|t} ... a_{t+H-1|t} (a_{u|v}: the
action for time u, predicted at time v). The action sent at t is the weighted average of every prediction of one
same time among the recent chunks. ACT averages the predictions for t itself; PTE, with offset f, those for t + f:

    B_{t+f} = [a_{t+f|t}, a_{t+f|t-1}, ..., a_{t+f|t-(H-1-f)}]    (index i: the chunk predicted i steps ago)
    a_t = sum_i w_i B_{t+f}[i] / sum_i w_i,    w_i = exp(-m * i)

So the robot is always commanded f steps ahead of where the demonstrations were: it moves faster than them (PTE: up
to 2x at the same success rate, 3x at a lower one, with 24 actions per chunk). Only W = H - f chunks predict t + f.

The action f + j of a chunk is sent j steps after its prediction, when the chunk is j steps old: its weight is always
w_j. Thus no chunk is kept: for each of the next W steps, the ensemble keeps the running sums of the weighted actions
and of their weights. Each step adds the new chunk's actions f ... H-1, sends sum / weight sum of the first step, and
moves the sums up by one step. Fixed shapes and no Python state: torch.compile makes one graph for every step
(CUDA graphs too, mode="reduce-overhead").

Differences from ml/act's TemporalProlepticEnsembler, which keeps the chunks in a zero-filled buffer:
- At the start (and after reset), fewer than W chunks exist. ml/act averages the missing ones as zeros with their
  full weight, which pulls the first actions towards 0 (the mean action, in normalized units). Here the average is over
  the chunks that exist, with their weights normalized to sum to 1: the first action is exactly the first chunk's.
- m < 0 weighs the older chunks more: ACT's original weighting, w_j = exp(-|m| j) with j = 0 the OLDEST chunk (equal
  after normalization). m > 0 (PTE, ml/act): the newest chunks more, reacting faster. m = 0: plain mean.

Units: any (raw or normalized), the average is linear. The sums are in the module's dtype (float32; .double() for
float64): a bf16 chunk (autocast) is averaged in float32. Call reset() at the start of every episode.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TemporalEnsemble(nn.Module):
    def __init__(self, action_horizon: int, action_dim: int, offset: int = 0, decay: float = 0.05) -> None:
        """action_horizon: H, the actions per chunk. action_dim: D, of each action. offset: f, how many steps ahead the
        robot is commanded, 0 <= f < H (0: ACT's ensemble). decay: m of w_j = exp(-m * j), j the age of the chunk in
        steps (PTE: 0.05)."""
        super().__init__()
        if not 0 <= offset < action_horizon:
            raise ValueError(f"offset must be in [0, {action_horizon}), got {offset}")
        self.chunk_shape, self.offset = (action_horizon, action_dim), offset
        window = action_horizon - offset
        weights = torch.exp(-decay * torch.arange(window, dtype=torch.float64)).float()
        if not torch.isfinite(weights).all():  # a NaN decay, or w_j above float32's 3e38: NaN actions
            raise ValueError(f"decay {decay} gives weights out of float32 over a window of {window} steps")
        self.register_buffer("weights", weights[:, None], persistent=False)  # (W, 1): w_j of the action f + j
        # (W, D + 1), row j for the step t + j: the sum of the weighted actions, then the sum of their weights. One
        # buffer, so each step is 5 small kernels
        self.register_buffer("sums", torch.zeros(window, action_dim + 1), persistent=False)

    def reset(self) -> None:
        """Forget the past chunks: at the start of every episode."""
        self.sums.zero_()

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: (H, D) actions, predicted at this step. Returns the (D,) action to send now."""
        if chunk.shape != self.chunk_shape:
            raise ValueError(f"expected a chunk of shape {self.chunk_shape}, got {tuple(chunk.shape)}")
        # detach: else each step's graph would hold the previous ones through the sums
        self.sums[:, :-1].addcmul_(self.weights, chunk[self.offset:].detach())
        self.sums[:, -1:].add_(self.weights)
        action = self.sums[0, :-1] / self.sums[0, -1]
        self._advance_one_step()
        return action

    def _advance_one_step(self) -> None:
        """Row j becomes row j - 1; the last row, a step that no chunk predicted yet, starts at 0."""
        self.sums.copy_(F.pad(self.sums[1:], (0, 0, 0, 1)))
