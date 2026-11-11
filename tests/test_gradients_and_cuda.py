"""Every layer and model of the policy, from one list of cases:
    gradients: torch.autograd.gradcheck, in float64 and training mode (dropout on, its masks seeded at every
        call), compares the gradient of every input and parameter with finite differences. It needs no
        reference, so it also covers our own models; the reference tests compare the layers with references.
    CUDA (skipped without a GPU): float32 outputs and gradients equal the CPU's, which the reference tests
        check; bf16 autocast stays close to float32; training mode gives every parameter a finite gradient.
"""

from functools import partial

import pytest
import torch
from torch import nn
from transformers import DINOv3ViTModel

from diff_style.models.layers import (MLP, AdaLNModulation, Attention, DropPath, NullTokenDropout, RotaryEmbedding,
                                      SwiGLUMLP, TimestepEmbedding, math_attention)
from diff_style.models.perceiver import Perceiver
from diff_style.models.transformer import Transformer, TransformerBlock
from test_dinov3_encoder import DINOV3_CACHED, make_encoder
from test_flow_matching import make_policy
from test_flow_matching_network import make_batch, make_network

D = 16  # width of the layer cases, 2 heads

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


class Velocity(nn.Module):
    """FlowMatchingNetwork end to end: observations -> obs tokens -> velocity of the noisy actions."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, images, qpos, past_actions, noisy_actions, tau):
        return self.net(noisy_actions, tau, self.net.encode_observations(images, qpos, past_actions))


class Stacked(nn.Module):
    """A layer that returns a tuple of tensors (the adaLN chunks), with its outputs stacked into one tensor."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, **inputs):
        return torch.stack(self.layer(**inputs))


class PolicyLoss(nn.Module):
    """FlowMatchingPolicy.loss as a module: normalization, the flow path and the network."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **batch):
        return self.policy.loss(batch)


def without_zero_init(module: nn.Module) -> nn.Module:
    """All-zero parameters (adaLN-Zero, the velocity head, norm biases) made random, so every parameter counts."""
    with torch.no_grad():
        for p in module.parameters():
            if not p.any():
                p.normal_(0, 0.1)
    return module


def random_init_dinov3() -> nn.Module:
    """The DINOv3 encoder, unfrozen, with transformers' random init. With the pretrained weights, the float32
    gradients of the early layers lose up to ~7e-4 against float64, on the CPU and on the GPU alike (and bf16 ~7%):
    a property of the weights, not of the code, and more than the tolerances of these tests."""
    encoder = make_encoder()
    encoder.model.load_state_dict(DINOv3ViTModel(encoder.model.config).state_dict())
    return encoder


def keep(*shape):
    """A random boolean mask that keeps the first key of every row, so no query is left with nothing."""
    mask = torch.rand(*shape) > 0.4
    mask[..., 0] = True
    return mask


# name -> (module, inputs as forward's keyword arguments). Float inputs are differentiated, masks are not.
CASES = {
    "attention": lambda: (Attention(D, 2, qk_norm=True, bias=True, dropout=0.2, is_causal=True),
                          {"x": torch.randn(2, 5, D), "mask": keep(5, 5)}),
    "cross-attention": lambda: (Attention(D, 2, dropout=0.2),
                                {"x": torch.randn(2, 5, D), "context": torch.randn(2, 7, D), "mask": keep(2, 1, 1, 7)}),
    "attention, math kernel": lambda: (Attention(D, 2, dropout=0.2, attention_fn=math_attention),
                                       {"x": torch.randn(2, 5, D)}),
    "rope": lambda: (RotaryEmbedding(head_dim=8), {"x": torch.randn(2, 5, 2, 8)}),
    "mlp": lambda: (MLP(D, 32, dropout=0.2), {"x": torch.randn(2, 5, D)}),
    "swiglu mlp": lambda: (SwiGLUMLP(D, 24, bias=True, dropout=0.2), {"x": torch.randn(2, 5, D)}),
    "timestep embedding": lambda: (TimestepEmbedding(D), {"tau": torch.rand(4)}),
    "adaln modulation": lambda: (Stacked(AdaLNModulation(12, D, num_chunks=6)), {"cond": torch.randn(2, 12)}),
    "drop path": lambda: (DropPath(0.4), {"x": torch.randn(8, 3, D)}),
    "null token dropout": lambda: (NullTokenDropout(D, 0.5), {"x": torch.randn(8, 3, D)}),
    "llama block": lambda: (TransformerBlock(D, 2, causal=True, dropout=0.1, drop_path=0.2),
                            {"x": torch.randn(2, 5, D)}),
    "dit block": lambda: (TransformerBlock(D, 2, mlp_ratio=4.0, cond_dim=D, attention=partial(Attention, bias=True),
                                           norm=partial(nn.LayerNorm, eps=1e-6), mlp=MLP),
                          {"x": torch.randn(2, 5, D), "cond": torch.randn(2, D)}),
    "decoder block": lambda: (TransformerBlock(D, 2, cross_attention=True, dropout=0.1, drop_path=0.2),
                              {"x": torch.randn(2, 5, D), "context": torch.randn(2, 7, D), "context_mask": keep(5, 7)}),
    "transformer": lambda: (Transformer(D, depth=2, num_heads=2, cross_attention=True, cond_dim=D, dropout=0.1,
                                        drop_path=0.2),
                            {"x": torch.randn(2, 5, D), "context": torch.randn(2, 7, D), "cond": torch.randn(2, D)}),
    "perceiver": lambda: (Perceiver(input_dim=12, num_latents=3, dim=D, num_heads=2,
                                    attention=partial(Attention, qk_norm=True)),  # QK-norm: the config's
                          {"x": torch.randn(2, 7, 12), "mask": keep(2, 7)}),
    "flow matching network": lambda: (
        Velocity(make_network(embedding_dropout=0.1, camera_dropout=0.3, qpos_dropout=0.3, past_action_dropout=0.3,
                              condition_dropout=0.3)),
        {**{k: v for k, v in make_batch(b=3).items() if k != "actions"},
         "noisy_actions": torch.randn(3, 4, 6), "tau": torch.rand(3)}),
    "flow matching policy loss": lambda: (PolicyLoss(make_policy()), make_batch(b=3)),
}
if DINOV3_CACHED:  # the vision encoder, unfrozen (frozen, it has no gradients)
    CASES["dinov3 encoder"] = lambda: (random_init_dinov3(), {"frames": torch.rand(2, 1, 3, 32, 32)})
RANDOM = {"flow matching policy loss"}  # draws its own noise: the CPU and the GPU draw differently


def build(name: str, dtype: torch.dtype = torch.float32, device: str = "cpu") -> tuple[nn.Module, dict]:
    """The case, built from the same seed every time, in dtype on device."""
    torch.manual_seed(0)
    module, inputs = CASES[name]()
    module = without_zero_init(module).to(device, dtype)
    inputs = {k: v.to(device, dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
    return module, inputs


def trainable_parameters(module: nn.Module) -> dict:
    """The parameters by name, without the frozen ones (e.g. DINOv3's mask token): no gradient to check."""
    return {name: p for name, p in module.named_parameters() if p.requires_grad}


def run(module: nn.Module, inputs: dict, grad_out: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """The output and the gradients of every float input and trainable parameter (by name), seeded the same every
    time; 0 for the unused ones (the null tokens in evaluation mode). grad_out: the upstream gradient; None = ones."""
    inputs = {k: v.detach().requires_grad_() if v.is_floating_point() else v for k, v in inputs.items()}
    torch.manual_seed(0)
    out = module(**inputs)
    leaves = {k: v for k, v in inputs.items() if v.is_floating_point()} | trainable_parameters(module)
    grads = torch.autograd.grad(out, list(leaves.values()), torch.ones_like(out) if grad_out is None else grad_out,
                                materialize_grads=True)
    return out.detach(), dict(zip(leaves, grads))


@pytest.mark.parametrize("name", CASES)
def test_gradients_match_finite_differences(name):
    """For each input and parameter in turn, gradcheck compares its gradient along a random direction with the
    finite difference along it (what gradcheck's fast mode checks per input). One number per check: a failure
    names the input or parameter and fails at once, instead of gradcheck rebuilding a whole Jacobian."""
    module, inputs = build(name, torch.float64)
    masks = {k: v for k, v in inputs.items() if not v.is_floating_point()}
    leaves = {k: v for k, v in inputs.items() if v.is_floating_point()} | trainable_parameters(module)
    leaves = {k: v.detach() for k, v in leaves.items()}
    params = trainable_parameters(module).keys()

    def forward(values: dict) -> torch.Tensor:
        torch.manual_seed(0)  # the same dropout masks and noise at every call
        kwargs = masks | {k: v for k, v in values.items() if k not in params}
        return torch.func.functional_call(module, {k: values[k] for k in params}, (), kwargs)

    for key, leaf in leaves.items():
        direction = torch.randn_like(leaf)
        step = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        along = lambda t: forward(leaves | {key: leaf + t * direction})  # this leaf moved by t, the others fixed
        assert torch.autograd.gradcheck(along, (step,), fast_mode=True, raise_exception=False), key


@cuda
@pytest.mark.parametrize("name", sorted(set(CASES) - RANDOM))
def test_cuda_matches_cpu_in_float32(name, monkeypatch):
    """Outputs and the gradients of every input and parameter. Evaluation mode: the GPU draws other dropout masks."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)  # full float32 matmuls, as on the CPU
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    (cpu, cpu_inputs), (gpu, gpu_inputs) = build(name), build(name, device="cuda")
    cpu_out, cpu_grads = run(cpu.eval(), cpu_inputs)
    gpu_out, gpu_grads = run(gpu.eval(), gpu_inputs)
    torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=1e-4, atol=1e-5)
    for key, grad in cpu_grads.items():
        torch.testing.assert_close(gpu_grads[key].cpu(), grad, rtol=1e-4, atol=1e-5, msg=lambda m: f"{key}: {m}")


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
@pytest.mark.parametrize("name", CASES)
def test_cuda_bf16_autocast_is_close_to_float32_and_trains(name):
    """Evaluation mode: the output and the input and parameter gradients within a few % of float32 (bf16 keeps 8
    bits of mantissa). Training mode, dropout on: every parameter gets a finite gradient."""
    module, inputs = build(name, device="cuda")
    out, grads = run(module.eval(), inputs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        half_out, half_grads = run(module, inputs)
    assert relative_error(half_out, out) < 2e-2
    # Gradients that are 0 in exact arithmetic come out as rounding noise (e.g. a key bias: it adds the same score
    # to a whole softmax row), with no relative error to speak of: compare the others
    largest = max(grad.norm() for grad in grads.values())
    for key, grad in grads.items():
        if grad.norm() > 1e-4 * largest:
            assert relative_error(half_grads[key], grad) < 5e-2, key
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, train_grads = run(module.train(), inputs)
    assert [key for key, grad in train_grads.items() if not grad.isfinite().all()] == []


@cuda
def test_drop_path_drops_whole_samples_on_cuda():
    x = torch.randn(256, 3, D, device="cuda")
    out = DropPath(0.5)(x)
    dropped = (out == 0).flatten(1).all(1)
    torch.testing.assert_close(out[~dropped], x[~dropped] / 0.5)
    assert 0 < dropped.sum() < 256


@cuda
def test_null_token_dropout_replaces_whole_samples_on_cuda():
    dropout = NullTokenDropout(D, 0.5).cuda()
    x = torch.randn(256, 3, D, device="cuda", dtype=torch.bfloat16)
    out = dropout(x)
    assert dropout.null is not None and out.dtype == torch.bfloat16
    dropped = (out == dropout.null.to(torch.bfloat16)).flatten(1).all(1)
    assert torch.equal(out[~dropped], x[~dropped])
    assert 0 < dropped.sum() < 256
