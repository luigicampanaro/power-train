"""GraphedPolicy against the policy's validation path (the datamodule's preprocessing, then predict_actions under bf16
autocast, as in bf16-mixed training), on the GPU: the same actions for the same noise, new noise on every call, the
inputs of each call, eager or compiled. The small policy of test_flow_matching.py, out of its zero inits."""

import pytest
import torch
from torch._inductor import config as inductor_config

from diff_style.inference.graphed_policy import GraphedPolicy, compile_for_inference
from diff_style.models.preprocess import preprocess_images
from test_flow_matching import make_policy
from test_flow_matching_network import C, H, T1, T2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

IMAGE_SIZE = (4, 4)  # the model input; the raw frames are 6x8


def cuda_policy():
    policy = make_policy()
    with torch.no_grad():
        for p in policy.net.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # out of the zero inits: the velocity counts
    return policy.cuda().eval()


def raw_batch(seed: int) -> dict[str, torch.Tensor]:
    """One sample as ObservationHistory gives it: uint8 frames, raw qpos and past actions, on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    return {"images": torch.randint(0, 256, (1, T2, C, 3, 6, 8), dtype=torch.uint8, generator=generator),
            "qpos": torch.randn(1, T1, 7, generator=generator),
            "past_actions": torch.randn(1, T1, 6, generator=generator)}


def validation_actions(policy, batch: dict, seed: int) -> torch.Tensor:
    batch = {key: value.cuda() for key, value in batch.items()}
    batch["images"] = preprocess_images(batch["images"], IMAGE_SIZE)  # the datamodule's, before autocast
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return policy.predict_actions(batch)[0].float()


def graphed_actions(graphed: GraphedPolicy, batch: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return graphed(batch)


def test_each_call_replays_the_validation_prediction_of_its_input():
    policy = cuda_policy()
    graphed = GraphedPolicy(policy, IMAGE_SIZE)
    for seed in [1, 2]:  # the second call replays the graph captured by the first, on its own input
        actions = graphed_actions(graphed, raw_batch(seed), seed)
        assert actions.shape == (H, 6) and actions.dtype == torch.float32
        torch.testing.assert_close(actions, validation_actions(policy, raw_batch(seed), seed), rtol=0, atol=0)


def test_each_call_draws_new_noise_and_returns_its_own_tensor():
    graphed = GraphedPolicy(cuda_policy(), IMAGE_SIZE)
    first = graphed(raw_batch(0))
    kept = first.clone()
    second = graphed(raw_batch(0))
    assert not torch.equal(first, second)
    torch.testing.assert_close(first, kept, rtol=0, atol=0)  # not overwritten by the next replay


@pytest.mark.parametrize("key, shape", [("images", (1, T2, C, 3, 8, 8)), ("qpos", (1, T1 - 1, 7))])
def test_an_input_of_another_shape_than_the_captured_one_raises(key, shape):
    graphed = GraphedPolicy(cuda_policy(), IMAGE_SIZE)
    graphed(raw_batch(0))
    batch = raw_batch(0)
    batch[key] = torch.zeros(shape, dtype=batch[key].dtype)
    with pytest.raises(ValueError, match=key):
        graphed(batch)


def test_the_compiled_policy_with_frozen_weights_matches_the_validation_prediction(monkeypatch):
    """Inductor folds the frozen weights, and their casts to bf16, into constants: the actions are eager's to bf16
    precision (the fused kernels round differently)."""
    monkeypatch.setattr(inductor_config, "freezing", inductor_config.freezing)  # the other tests get it back
    torch._dynamo.reset()  # a new cache, so no earlier test uses up the recompile limit
    policy = cuda_policy()
    expected = validation_actions(policy, raw_batch(3), seed=3)
    compile_for_inference(policy)
    actions = graphed_actions(GraphedPolicy(policy, IMAGE_SIZE), raw_batch(3), seed=3)
    torch.testing.assert_close(actions, expected, rtol=0, atol=3e-2 * expected.abs().max().item())
