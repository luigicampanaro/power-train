"""Perceiver against Hugging Face's Idefics2 connector (outputs and gradients), plus its properties and init."""

from functools import partial

import pytest
import torch
from torch import nn

from diff_style.models.layers import MLP, Attention, SwiGLUMLP
from diff_style.models.perceiver import Perceiver
from references import REFERENCES, assert_gradients_match, with_grad

# Idefics2's names -> ours (tests/references/capture_outputs.py: idefics2_perceiver). Its MLP on the latents (zeroed
# there) and the norm before it have no counterpart: the Perceiver has no MLP on the latents.
IDEFICS2_NAMES = {"projection.gate_proj": "input_mlp.w1", "projection.up_proj": "input_mlp.w3",
                  "projection.down_proj": "input_mlp.w2", "layer.input_latents_norm": "latent_norm",
                  "layer.input_context_norm": "input_norm", "layer.self_attn.q_proj": "attention.wq",
                  "layer.self_attn.k_proj": "attention.wk", "layer.self_attn.v_proj": "attention.wv",
                  "layer.self_attn.o_proj": "attention.wo"}


def from_idefics2(state: dict) -> dict:
    return {"latents": state["latents"]} | {f"{ours}.weight": state[f"{theirs}.weight"]
                                            for theirs, ours in IDEFICS2_NAMES.items()}


def make_perceiver(seed=0, **parts):
    torch.manual_seed(seed)
    return Perceiver(input_dim=12, num_latents=4, dim=32, num_heads=4, **parts)


def test_matches_idefics2_connector():
    """Idefics2's SwiGLU projection, then latents + attention(RMSNorm(latents), RMSNorm(context)), on a batch of 3
    with padded contexts. The feature norm is off: Idefics2 takes the encoder's normalized output as it is."""
    ref = REFERENCES["idefics2_perceiver"]
    perceiver = make_perceiver(feature_norm=nn.Identity)
    perceiver.load_state_dict(from_idefics2(ref["state_dict"]))  # strict: every parameter of ours is set
    x = with_grad(ref["x"])
    out = perceiver(x, mask=ref["keep"])
    torch.testing.assert_close(out, ref["out"])
    assert_gradients_match(out, ref, {"x": x}, perceiver, from_idefics2)


@pytest.mark.parametrize("feature_norm", [None, nn.Identity])  # None: the default
def test_feature_norm_makes_the_output_independent_of_each_input_token_scale(feature_norm):
    """By default the features are normalized before the input MLP, whatever the stage before gives;
    feature_norm=nn.Identity feeds them as they are."""
    perceiver = make_perceiver() if feature_norm is None else make_perceiver(feature_norm=feature_norm)
    x = torch.randn(2, 9, 12)
    scaled = x * torch.empty(2, 9, 1).uniform_(0.1, 10)  # each token its own scale
    assert torch.allclose(perceiver(scaled), perceiver(x), atol=1e-4) == (feature_norm is None)


@pytest.mark.parametrize("mlp", [SwiGLUMLP, MLP])
def test_output_has_num_latents_tokens_whatever_the_input_length(mlp):
    perceiver = make_perceiver(mlp=mlp)
    for length in [1, 50]:
        assert perceiver(torch.randn(3, length, 12)).shape == (3, 4, 32)


def test_input_order_does_not_matter():
    """No positions are added: the inputs are a set (they carry their own position, e.g. DINOv3)."""
    perceiver = make_perceiver()
    x = torch.randn(2, 9, 12)
    assert torch.allclose(perceiver(x), perceiver(x[:, torch.randperm(9)]), atol=1e-5)


@pytest.mark.parametrize("mlp", [SwiGLUMLP, partial(MLP, bias=True)])
def test_init_is_the_llama3_init_of_the_transformer(mlp):
    """Every weight matrix and the latents start at trunc_normal(0.02), biases at 0 and norm weights at 1: the input
    MLP projects into the latents' space (as a ViT's patch embedding), and the attention is the latents' only
    residual branch (Llama 3's depth init with one branch: 0.02 / sqrt(1)). The init_weights of SwiGLUMLP and
    Attention are tested against torchtitan's, MLP's against V-JEPA 2's (test_transformer.py)."""
    torch.manual_seed(0)
    perceiver = Perceiver(input_dim=384, num_latents=256, dim=256, num_heads=4, mlp=mlp,
                          attention=partial(Attention, bias=True), norm=nn.LayerNorm)
    params = dict(perceiver.named_parameters())
    matrices = [name for name, p in params.items() if p.ndim == 2]
    assert len(matrices) == 1 + 4 + (3 if mlp is SwiGLUMLP else 2)  # the latents, wq wk wv wo, the input MLP's
    for name in matrices:
        assert params[name].std().item() == pytest.approx(0.02, rel=0.05), name
    assert [name for name, p in params.items() if name.endswith("bias") and p.any()] == []
    assert all(params[f"{norm}.weight"].eq(1).all() for norm in ["feature_norm", "latent_norm", "input_norm"])
