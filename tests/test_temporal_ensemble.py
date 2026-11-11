"""TemporalEnsemble against the reference implementations of the temporal ensemble of action chunks:
- ACT's evaluation loop (tonyzhaozh/act 76cf30b, imitate_episodes.py, MIT): offset 0, older chunks weighed more.
- LeRobot's ACTTemporalEnsembler (huggingface/lerobot cc2fbf0, policies/act/modeling_act.py, Apache-2.0): the same
  average, computed online.
- The equations 3 and 4 of PTE (Park et al. 2024, arXiv 2410.16981), written out from the whole history of chunks.
The two codes are copied at the bottom of this file, only moved from CUDA to the CPU (and ACT's axis= is dim=).
Their coefficient k weighs the OLDEST chunk most (w_0: the oldest): our decay is -k. The sequences run past the
warm-up (fewer chunks than the window) into the steady state. Then: torch.compile, one graph for every step.
"""

import math

import numpy as np
import pytest
import torch
import torch._dynamo
import torch._dynamo.config

from diff_style.inference.temporal_ensemble import TemporalEnsemble

HORIZON, ACTION_DIM = 8, 3
STEPS = 2 * HORIZON + 3  # the warm-up, then the steady state
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def new_ensemble(offset: int = 2, decay: float = 0.05) -> TemporalEnsemble:
    return TemporalEnsemble(HORIZON, ACTION_DIM, offset, decay)


def random_chunks(steps: int = STEPS) -> torch.Tensor:
    """(steps, H, D) chunks, one per step, seeded. No exact zero: ACT's loop takes a 0 for an empty slot."""
    return torch.randn(steps, HORIZON, ACTION_DIM, generator=torch.Generator().manual_seed(0))


def ensembled(ensemble, chunks: torch.Tensor) -> torch.Tensor:
    """(steps, D): the action that ensemble sends at each step, fed one chunk per step. A copy of each: a CUDA graph
    writes the next action into the same memory."""
    return torch.stack([ensemble(chunk).clone() for chunk in chunks])


def test_a_hand_computed_proleptic_sequence():
    """H 3, f 1, w_i = 2^-i: each step averages the action f + i of the chunk of i steps ago (the newest first)."""
    ensemble = TemporalEnsemble(action_horizon=3, action_dim=1, offset=1, decay=math.log(2))
    chunks = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]).unsqueeze(-1)
    expected = [11.0, (21.0 + 12.0 / 2) / 1.5, (31.0 + 22.0 / 2) / 1.5]  # window H - f = 2: c0 is out at step 2
    torch.testing.assert_close(ensembled(ensemble, chunks).squeeze(-1), torch.tensor(expected))


def test_the_first_actions_are_not_pulled_towards_zero():
    """The warm-up averages only the chunks that exist (ml/act averaged the empty slots as zeros)."""
    torch.testing.assert_close(ensembled(new_ensemble(), torch.full((STEPS, HORIZON, ACTION_DIM), 5.0)),
                               torch.full((STEPS, ACTION_DIM), 5.0))


@pytest.mark.parametrize("k", [0.01, -0.05])  # 0.01: ACT's value
def test_matches_the_act_evaluation_loop(k):
    chunks = random_chunks()
    ours = ensembled(new_ensemble(offset=0, decay=-k), chunks)
    torch.testing.assert_close(ours, act_temporal_aggregation(chunks, k).to(ours.dtype))


@pytest.mark.parametrize("k", [0.01, 0.0, -0.05])
def test_matches_lerobot_act_temporal_ensembler(k):
    chunks = random_chunks()
    lerobot = ACTTemporalEnsembler(k, HORIZON)
    expected = torch.stack([lerobot.update(chunk.unsqueeze(0)).squeeze(0) for chunk in chunks])
    torch.testing.assert_close(ensembled(new_ensemble(offset=0, decay=-k), chunks), expected)


@pytest.mark.parametrize("offset, decay", [(0, 0.05), (1, 0.05), (5, 0.05), (HORIZON - 1, 0.05), (3, 0.0),
                                           (3, -0.01)])
def test_matches_the_pte_equations(offset, decay):
    chunks = random_chunks()
    torch.testing.assert_close(ensembled(new_ensemble(offset, decay), chunks), pte_equations(chunks, offset, decay))


def test_reset_starts_a_new_episode():
    """After reset, the sequence is the one of a new ensemble: no chunk of the last episode is in the average."""
    ensemble, chunks = new_ensemble(), random_chunks()
    ensembled(ensemble, -chunks)
    ensemble.reset()
    torch.testing.assert_close(ensembled(ensemble, chunks), ensembled(new_ensemble(), chunks))


def test_a_chunk_buffer_that_the_caller_reuses_does_not_change_the_past_chunks():
    """TensorRT and CUDA graphs write every output into one same buffer: the ensemble keeps no view of a chunk."""
    ensemble, chunks, buffer = new_ensemble(), random_chunks(), torch.empty(HORIZON, ACTION_DIM)
    actions = []
    for chunk in chunks:
        buffer.copy_(chunk)
        actions.append(ensemble(buffer))
    torch.testing.assert_close(torch.stack(actions), ensembled(new_ensemble(), chunks))


def test_chunks_that_require_grad_give_no_autograd_history():
    """Else each step would chain its graph to the previous one through the sums: the memory would grow forever."""
    ensemble = new_ensemble()
    for chunk in random_chunks().requires_grad_():
        assert not ensemble(chunk).requires_grad


def test_bfloat16_chunks_are_averaged_in_float32_and_a_double_ensemble_in_float64():
    chunks = random_chunks().bfloat16()
    ours = ensembled(new_ensemble(), chunks)
    assert ours.dtype == torch.float32
    torch.testing.assert_close(ours, ensembled(new_ensemble(), chunks.float()))
    assert new_ensemble().double()(random_chunks()[0].double()).dtype == torch.float64


@CUDA
def test_the_ensemble_runs_on_the_device_it_is_moved_to():
    chunks = random_chunks()
    ours = ensembled(new_ensemble().cuda(), chunks.cuda())
    assert ours.is_cuda
    torch.testing.assert_close(ours.cpu(), ensembled(new_ensemble(), chunks))


@pytest.mark.parametrize("device, mode", [("cpu", None), pytest.param("cuda", "reduce-overhead", marks=CUDA)])
def test_compiles_into_one_graph_that_every_step_and_reset_reuse(device, mode, monkeypatch):
    """fullgraph: no graph break. error_on_recompile: the warm-up, the steady state and a new episode all run the
    first graph. CUDA: inductor and CUDA graphs (reduce-overhead), the fast path of a robot; CPU: aot_eager, the
    capture without a slow C++ build."""
    torch._dynamo.reset()  # the compiled forward of an other test's ensemble would make this one a recompile
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    ensemble, chunks = new_ensemble().to(device), random_chunks().to(device)
    compiled = torch.compile(ensemble, fullgraph=True, mode=mode, backend="inductor" if mode else "aot_eager")
    first_episode = ensembled(compiled, chunks)
    ensemble.reset()
    second_episode = ensembled(compiled, chunks)
    expected = ensembled(new_ensemble().to(device), chunks)
    torch.testing.assert_close(first_episode, expected)
    torch.testing.assert_close(second_episode, expected)


@pytest.mark.parametrize("offset", [-1, HORIZON])
def test_an_offset_out_of_the_chunk_raises(offset):
    with pytest.raises(ValueError, match="offset"):
        new_ensemble(offset)


@pytest.mark.parametrize("decay", [-100.0, math.nan])  # exp(100 * 7) overflows float32: NaN actions after a step
def test_a_decay_that_gives_weights_out_of_float32_raises(decay):
    with pytest.raises(ValueError, match="decay"):
        new_ensemble(offset=0, decay=decay)


@pytest.mark.parametrize("shape", [(HORIZON + 1, ACTION_DIM), (HORIZON, ACTION_DIM + 1)])
def test_a_chunk_of_another_shape_raises(shape):
    with pytest.raises(ValueError, match="chunk"):
        new_ensemble()(torch.zeros(shape))


def pte_equations(chunks: torch.Tensor, offset: int, decay: float) -> torch.Tensor:
    """PTE's eq. 4, B_{t+f}[i] = a_{t+f|t-i} for i < H - f, and eq. 3, a_t = sum_i w_i B[i] / sum_i w_i with
    w_i = exp(-m i); only the chunks predicted at a step >= 0 exist."""
    actions = []
    for t in range(len(chunks)):
        predictions = [chunks[t - i, offset + i] for i in range(HORIZON - offset) if t - i >= 0]
        weights = torch.tensor([math.exp(-decay * i) for i in range(len(predictions))])
        actions.append((weights[:, None] * torch.stack(predictions)).sum(0) / weights.sum())
    return torch.stack(actions)


def act_temporal_aggregation(chunks: torch.Tensor, k: float) -> torch.Tensor:
    """tonyzhaozh/act imitate_episodes.py eval_bc, temporal_agg: the policy queried at every step t."""
    max_timesteps, num_queries, state_dim = len(chunks), HORIZON, ACTION_DIM
    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim])
    raw_actions = []
    for t in range(max_timesteps):
        all_actions = chunks[t].unsqueeze(0)
        all_time_actions[[t], t:t+num_queries] = all_actions
        actions_for_curr_step = all_time_actions[:, t]
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        raw_actions.append(raw_action.squeeze(0))
    return torch.stack(raw_actions)


class ACTTemporalEnsembler:
    """huggingface/lerobot ACTTemporalEnsembler, as is (its docstring left out, ty ignores added)."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]  # ty: ignore[unsupported-operator]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)  # ty: ignore[unsupported-operator]
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action
