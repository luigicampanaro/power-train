"""Attention, RoPE, the MLPs, the timestep embedding and the adaLN modulation, against reference implementations,
plus the properties they promise.

References, outputs and gradients: torchtitan v0.1.0 Llama 3 (RoPE, SwiGLU), Hugging Face's LlamaMLP (SwiGLU)
and Qwen3 attention (QK-norm), timm's Mlp and SwiGLU with dropout, diffusers (timestep embedding), as fixed-mask captures
(tests/references/); torch.nn.MultiheadAttention; the attention kernels against softmax(q k^T / sqrt(d))
v written out.
"""

from functools import partial
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from diff_style.models.layers import attention as attention_module
from diff_style.models.layers import (MLP, AdaLNModulation, Attention, Modulation, RotaryEmbedding, SwiGLUMLP,
                                      TimestepEmbedding, causal_mask, flash3, math_attention, sdpa)
from diff_style.models.transformer import TransformerBlock

from references import (DROPOUT_REFERENCES, REFERENCES, assert_gradients_match, reference_gradients, renamed,
                        with_grad)
from references.deterministic_dropout import deterministic_dropout

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_rope_matches_torchtitan():
    ref = REFERENCES["llama_rope"]
    rope = RotaryEmbedding(head_dim=16)
    xq, xk = with_grad(ref["xq"]), with_grad(ref["xk"])
    q, k = rope(xq), rope(xk)
    assert torch.allclose(q, ref["q"], atol=1e-6)
    assert torch.allclose(k, ref["k"], atol=1e-6)
    assert_gradients_match((q, k), ref, {"xq": xq, "xk": xk})


def test_rope_scores_depend_only_on_the_distance():
    """q at position i and k at position j: the score is a function of i - j only."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=16)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)  # (B, L, heads, head_dim)

    def score(i, j):
        return (rope(q, torch.tensor([i])) * rope(k, torch.tensor([j]))).sum().item()

    assert score(3, 1) == pytest.approx(score(10, 8), abs=1e-4) == pytest.approx(score(2, 0), abs=1e-4)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-2)


def test_rope_leaves_position_zero_unchanged_and_keeps_the_dtype():
    x = torch.randn(2, 4, 3, 8, dtype=torch.bfloat16)
    out = RotaryEmbedding(head_dim=8)(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[:, 0], x[:, 0])


def test_swiglu_mlp_matches_torchtitan():
    ref = REFERENCES["llama_feed_forward"]
    mlp = SwiGLUMLP(dim=32, hidden_dim=ref["state_dict"]["w1.weight"].shape[0])
    mlp.load_state_dict(ref["state_dict"])  # strict: the same parameter names
    x = with_grad(ref["x"])
    out = mlp(x)
    assert torch.allclose(out, ref["out"], atol=1e-6)
    assert_gradients_match(out, ref, {"x": x}, mlp)


@pytest.mark.parametrize("bias", [False, True])
def test_swiglu_mlp_matches_hugging_face_llama_mlp(bias):
    """Hugging Face's down_proj(silu(gate_proj x) * up_proj x): w1 = gate_proj, w3 = up_proj, w2 = down_proj."""
    ref = REFERENCES["hf_llama_mlp_bias" if bias else "hf_llama_mlp"]
    mlp = SwiGLUMLP(dim=32, hidden_dim=85, bias=bias)
    names = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    mlp.load_state_dict(renamed(ref["state_dict"], names))  # strict: every weight and bias mapped
    x = with_grad(ref["x"])
    out = mlp(x)
    assert torch.allclose(out, ref["out"], atol=1e-6)
    assert_gradients_match(out, ref, {"x": x}, mlp, lambda grads: renamed(grads, names))


def test_swiglu_hidden_width_matches_torchtitan_llama_feed_forward():
    """The MLP width a TransformerBlock builds, int(8/3 * dim) rounded up to multiple_of, against torchtitan's Llama 3
    FeedForward: every dim from 32 to 1024 in steps of 32, 2048 and 4096 (4096 -> 11008 with 256, Llama 2 7B's).
    Without the rounding, 512 gives 1365, an odd width that the tensor cores run 30% slower."""
    ref = REFERENCES["llama_ffn_hidden_dims"]
    with torch.device("meta"):  # shapes only, no memory
        ours = [TransformerBlock(dim, 1, mlp=partial(SwiGLUMLP, multiple_of=multiple_of)).feed_forward
                for dim, multiple_of in zip(ref["dim"].tolist(), ref["multiple_of"].tolist())]
    assert [mlp.w1.out_features for mlp in ours] == ref["hidden_dim"].tolist()
    assert all(mlp.w1.out_features == mlp.w3.out_features == mlp.w2.in_features for mlp in ours)


def test_mlp_matches_timm_mlp_with_dropout():
    """In training, with timm's dropout on the hidden activations (its drop1), and another output width (its
    out_features). The default, out_dim = dim, is DiT's and torch's MLP in tests/test_transformer.py."""
    ref = DROPOUT_REFERENCES["timm_mlp"]
    mlp = MLP(dim=32, hidden_dim=128, dropout=ref["p"], out_dim=24)  # exact GELU, as timm's default
    mlp.load_state_dict(ref["state_dict"])  # strict: timm's names, fc1 and fc2
    x = with_grad(ref["x"])
    with deterministic_dropout():
        out = mlp(x)
    assert torch.allclose(out, ref["out"], atol=1e-6)
    assert_gradients_match(out, ref, {"x": x}, mlp)


def test_swiglu_mlp_matches_timm_swiglu_with_dropout():
    """timm's SwiGLU: fc2(drop1(silu(fc1_g x) * fc1_x x)), our w2(dropout(silu(w1 x) * w3 x)), with another output
    width (its out_features)."""
    ref = DROPOUT_REFERENCES["timm_swiglu"]
    mlp = SwiGLUMLP(dim=32, hidden_dim=85, dropout=ref["p"], out_dim=24)
    names = {"fc1_g": "w1", "fc1_x": "w3", "fc2": "w2"}
    mlp.load_state_dict(renamed(ref["state_dict"], names))
    x = with_grad(ref["x"])
    with deterministic_dropout():
        out = mlp(x)
    assert torch.allclose(out, ref["out"], atol=1e-6)
    assert_gradients_match(out, ref, {"x": x}, mlp, lambda grads: renamed(grads, names))


@pytest.mark.parametrize("make", [partial(MLP, 32, 128), partial(SwiGLUMLP, 32, 85)])
def test_mlp_dropout_is_on_in_training_only(make):
    torch.manual_seed(0)
    mlp, no_dropout = make(dropout=0.5), make()
    no_dropout.load_state_dict(mlp.state_dict())
    x = torch.randn(2, 5, 32)
    assert not torch.allclose(mlp(x), no_dropout(x), atol=1e-3)
    mlp.eval()
    assert torch.equal(mlp(x), no_dropout(x))


def attention_and_reference(seed=0, dropout=0.0, bias=False):
    """Our Attention and an nn.MultiheadAttention with the same weights."""
    torch.manual_seed(seed)
    attention = Attention(dim=32, num_heads=4, dropout=dropout, bias=bias)
    mha = nn.MultiheadAttention(32, 4, bias=bias, batch_first=True, dropout=dropout)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([attention.wq.weight, attention.wk.weight, attention.wv.weight]))
        mha.out_proj.weight.copy_(attention.wo.weight)
        if bias:
            mha.in_proj_bias.copy_(torch.cat([attention.wq.bias, attention.wk.bias, attention.wv.bias]))
            mha.out_proj.bias.copy_(attention.wo.bias)
    return attention, mha


def mha_to_ours(state: dict) -> dict:
    """torch's nn.MultiheadAttention names -> ours; it keeps q, k and v in one in_proj: split it."""
    ours = {}
    for kind in ["weight", "bias"]:
        if f"in_proj_{kind}" in state:
            ours |= dict(zip([f"wq.{kind}", f"wk.{kind}", f"wv.{kind}"], state[f"in_proj_{kind}"].chunk(3)))
            ours[f"wo.{kind}"] = state[f"out_proj.{kind}"]
    return ours


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_attention_matches_torch_multihead_attention(cross, masked, bias):
    """Outputs, and the gradients of the inputs and of every weight and bias, in float64."""
    attention, mha = (module.double() for module in attention_and_reference(bias=bias))
    x = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
    context = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True) if cross else x
    mask = None
    if masked:
        mask = torch.rand(5, context.shape[1]) > 0.5
        mask[:, 0] = True  # every query keeps at least one key
    ours = attention(x, context=context if cross else None, mask=mask)
    # nn.MultiheadAttention's boolean mask is the opposite: True = may NOT attend
    reference, _ = mha(x, context, context, attn_mask=None if mask is None else ~mask, need_weights=False)
    assert torch.allclose(ours, reference, atol=1e-10)
    inputs = {"x": x, "context": context} if cross else {"x": x}
    ref = reference_gradients(reference, inputs, dict(mha.named_parameters()), torch.randn_like(ours))
    assert_gradients_match(ours, ref, inputs, attention, mha_to_ours)


def test_no_mask_and_an_all_true_mask_give_the_same_output():
    attention, _ = attention_and_reference()
    x, context = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    all_true = torch.ones(5, 7, dtype=torch.bool)
    assert torch.allclose(attention(x, context), attention(x, context, mask=all_true), atol=1e-6)


def test_masked_keys_get_no_gradient():
    attention, _ = attention_and_reference()
    x, context = torch.randn(2, 5, 32), torch.randn(2, 7, 32, requires_grad=True)
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[:, 3] = False  # no query may read key 3
    attention(x, context, mask=mask).sum().backward()
    grad = context.grad
    assert grad is not None
    assert torch.equal(grad[:, 3], torch.zeros(2, 32))
    assert (grad[:, [0, 1, 2, 4, 5, 6]].abs().sum(-1) > 0).all()


def test_qk_norm_makes_the_output_independent_of_the_query_and_key_scale():
    """RMSNorm on each head's queries and keys: scaling wq or wk changes nothing."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    for qk_norm in [True, False]:
        attention = Attention(dim=32, num_heads=4, qk_norm=qk_norm)
        before = attention(x)
        with torch.no_grad():
            attention.wq.weight.mul_(10)
            attention.wk.weight.mul_(2)  # scaling down instead would make eps (1e-5) count against tiny keys
            attention.wq.weight[:8].mul_(3)  # the first head only
        assert torch.allclose(attention(x), before, atol=1e-4) == qk_norm  # eps leaves ~2e-5


QWEN3_NAMES = {"q_proj": "wq", "k_proj": "wk", "v_proj": "wv", "o_proj": "wo", "q_norm": "q_norm", "k_norm": "k_norm"}


def test_qk_norm_attention_matches_qwen3():
    """Qwen3's attention: per-head RMSNorm on q and k, causal (its RoPE is turned off in the reference)."""
    ref = REFERENCES["qwen3_attention"]
    attention = Attention(dim=32, num_heads=4, qk_norm=True)
    attention.load_state_dict(renamed(ref["state_dict"], QWEN3_NAMES))
    x = with_grad(ref["x"])
    out = attention(x, mask=causal_mask(5, 5))
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x}, attention, lambda grads: renamed(grads, QWEN3_NAMES))


@pytest.mark.parametrize("causal_flag", [False, True])  # the combined mask, or is_causal + the padding mask
def test_qk_norm_attention_matches_qwen3_with_padding(causal_flag):
    """A padded batch: Qwen3's additive causal + padding mask, ours as booleans (B, 1, Lq, Lk)."""
    ref = REFERENCES["qwen3_attention_padded"]
    attention = Attention(dim=32, num_heads=4, qk_norm=True, is_causal=causal_flag)
    attention.load_state_dict(renamed(ref["state_dict"], QWEN3_NAMES))
    padding = ref["keep"][:, :, -1:, :]  # (B, 1, 1, Lk): the last query sees every key that isn't padding
    x = with_grad(ref["x"])
    out = attention(x, mask=padding if causal_flag else ref["keep"])
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x}, attention, lambda grads: renamed(grads, QWEN3_NAMES))
    assert not torch.allclose(attention(ref["x"]), ref["out"], atol=1e-3)  # the padding matters


def random_qkv_and_mask(dtype=torch.float32):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, length, 8, dtype=dtype) for length in [5, 7, 7])  # (B, heads, L, head_dim)
    mask = torch.rand(2, 1, 5, 7) > 0.5
    mask[..., 0] = True  # every query keeps at least one key
    return q, k, v, mask


@pytest.mark.parametrize("masked, is_causal", [(False, False), (True, False), (False, True)])  # torch: one or the other
@pytest.mark.parametrize("scale", [None, 0.3])
@pytest.mark.parametrize("kernels", [None, ["MATH"]])  # the GPU kernels: tests/test_attention_cuda.py
def test_sdpa_matches_the_written_out_softmax(masked, is_causal, scale, kernels):
    q, k, v, mask = random_qkv_and_mask(torch.float64)
    mask = mask if masked else None
    expected = math_attention(q, k, v, mask, is_causal=is_causal, scale=scale)
    assert torch.allclose(sdpa(q, k, v, mask, is_causal=is_causal, scale=scale, kernels=kernels), expected, atol=1e-12)


def test_causal_mask_lets_query_i_see_keys_0_to_i():
    assert causal_mask(3, 3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


@pytest.mark.parametrize("kernel", [math_attention, partial(sdpa, kernels=["MATH"])])
def test_attention_dropout_drops_weights_without_changing_the_mean(kernel):
    """Dropout zeroes attention weights and scales the others by 1 / (1 - p): one draw differs from
    the output without dropout, the mean over many draws is the same."""
    q, k, v, mask = random_qkv_and_mask()
    many = [t.expand(4000, -1, -1, -1) for t in (q[:1], k[:1], v[:1])]  # one sample, 4000 dropout draws
    torch.manual_seed(0)
    draws = kernel(*many, mask[:1], dropout_p=0.5)
    exact = math_attention(q[:1], k[:1], v[:1], mask[:1])
    assert not torch.allclose(draws[0], exact[0], atol=1e-2)
    assert torch.allclose(draws.mean(0), exact[0], atol=0.05)
    assert torch.equal(kernel(q, k, v, mask, dropout_p=0.0), kernel(q, k, v, mask))  # p = 0: no dropout


def install_fake_flash3(monkeypatch):
    """FA3 can't run here (Hopper only): a stand-in with its interface, (B, L, heads, head_dim) tensors
    in and (out, log-sum-exp) out, checks that we pass and read the right layout."""
    def flash_attn_func(q, k, v, softmax_scale=None, causal=False):
        assert q.shape == (2, 5, 4, 8) and k.shape == v.shape and k.shape[:2] == (2, 7)
        out = math_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                             is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2), None

    monkeypatch.setattr(attention_module, "flash_attn_func", flash_attn_func)


def test_flash3_passes_and_reads_the_fa3_layout(monkeypatch):
    install_fake_flash3(monkeypatch)
    q, k, v, mask = random_qkv_and_mask()
    assert torch.allclose(flash3(q, k, v), math_attention(q, k, v), atol=1e-6)
    expected = math_attention(q, k, v, is_causal=True, scale=0.3)
    assert torch.allclose(flash3(q, k, v, is_causal=True, scale=0.3), expected, atol=1e-6)
    with pytest.raises(ValueError, match="no mask"):
        flash3(q, k, v, mask)
    with pytest.raises(ValueError, match="no dropout"):
        flash3(q, k, v, dropout_p=0.1)


def test_flash3_without_the_package_says_so(monkeypatch):
    monkeypatch.setattr(attention_module, "flash_attn_func", None)
    q, k, v, _ = random_qkv_and_mask()
    with pytest.raises(ImportError, match="flash_attn_interface"):
        flash3(q, k, v)


@pytest.mark.parametrize("name", [path.stem for path in sorted(CONFIG_DIR.glob("attention_fn/*.yaml"))])
def test_every_attention_fn_config_computes_attention(name, monkeypatch):
    install_fake_flash3(monkeypatch)
    attention_fn = hydra.utils.instantiate(OmegaConf.load(CONFIG_DIR / "attention_fn" / f"{name}.yaml"))
    q, k, v, _ = random_qkv_and_mask()
    assert torch.allclose(attention_fn(q, k, v, None), math_attention(q, k, v), atol=1e-5)


def test_attention_uses_its_kernel():
    sdpa_attention, _ = attention_and_reference()
    written_out = Attention(dim=32, num_heads=4, attention_fn=math_attention)
    written_out.load_state_dict(sdpa_attention.state_dict())
    x, context = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    assert torch.allclose(written_out(x, context), sdpa_attention(x, context), atol=1e-5)
    calls = []

    def spy_fn(q, k, v, mask, dropout_p, is_causal, scale):
        calls.append((q.shape, k.shape, dropout_p, is_causal, scale))
        return sdpa(q, k, v, mask, is_causal=is_causal, scale=scale)

    spy = Attention(dim=32, num_heads=4, dropout=0.3, is_causal=True, scale=0.5, attention_fn=spy_fn)
    spy(x)
    spy.eval()
    spy(x)
    # q and k (B, heads, L, head_dim); no dropout in eval
    assert calls == [((2, 4, 5, 8), (2, 4, 5, 8), 0.3, True, 0.5), ((2, 4, 5, 8), (2, 4, 5, 8), 0.0, True, 0.5)]


def test_is_causal_equals_a_causal_mask_and_combines_with_a_mask():
    torch.manual_seed(0)
    causal = Attention(dim=32, num_heads=4, is_causal=True)
    plain = Attention(dim=32, num_heads=4)
    plain.load_state_dict(causal.state_dict())
    x = torch.randn(2, 5, 32)
    assert torch.allclose(causal(x), plain(x, mask=causal_mask(5, 5)), atol=1e-6)
    keep = torch.rand(5, 5) > 0.3
    keep[:, 0] = True  # every query keeps at least one key
    assert torch.allclose(causal(x, mask=keep), plain(x, mask=keep & causal_mask(5, 5)), atol=1e-6)


def test_is_causal_is_refused_in_cross_attention():
    with pytest.raises(ValueError, match="self-attention only"):
        Attention(dim=32, num_heads=4, is_causal=True)(torch.randn(1, 5, 32), context=torch.randn(1, 7, 32))


def test_timestep_embedding_matches_diffusers():
    """Our tau in [0, 1] is the diffusion timestep tau * 1000 of diffusers' Timesteps."""
    ref = REFERENCES["diffusers_timestep_embedding"]
    embedding = TimestepEmbedding(dim=32)

    def to_ours(state):  # diffusers' TimestepEmbedding (linear_1, linear_2) -> our MLP (mlp.fc1, mlp.fc2)
        return renamed(state, {"linear_1": "mlp.fc1", "linear_2": "mlp.fc2"})

    embedding.load_state_dict(to_ours(ref["state_dict"]))  # strict: every weight and bias mapped
    tau = with_grad(ref["tau"])
    out = embedding(tau)
    assert torch.allclose(out, ref["out"], atol=1e-5)
    # The saved gradients are from another machine: the phases reach tau * 1000, where cos and sin differ in their
    # last bits across libm builds, and that shift changes every gradient below by more than float32 precision.
    # Check the gradients against the exact (float64) derivative of the same forward instead: no machine enters,
    # and a wrong scale, frequency or backward still fails.
    exact = TimestepEmbedding(dim=32).double()
    exact.load_state_dict({name: value.double() for name, value in embedding.state_dict().items()})
    exact_tau = ref["tau"].clone().double().requires_grad_()
    exact_grads = torch.autograd.grad(exact(exact_tau), [exact_tau, *exact.parameters()],
                                      [grad.double() for grad in ref["grad_out"]])
    grads = torch.autograd.grad(out, [tau, *embedding.parameters()], ref["grad_out"])
    names = ["tau", *dict(embedding.named_parameters())]
    for name, grad, expected in zip(names, grads, exact_grads):
        torch.testing.assert_close(grad.double(), expected, rtol=1e-4, atol=1e-4, msg=name)


def test_timestep_frequencies_are_computed_once_and_left_alone():
    """Not a buffer: model conversion, the EMA and checkpoints don't touch them; they follow the input's device."""
    embedding = TimestepEmbedding(dim=32)
    freqs = embedding.freqs.clone()
    for convert in [nn.Module.bfloat16, nn.Module.half, nn.Module.double]:
        assert torch.equal(convert(embedding).freqs, freqs)  # bf16 or fp16 would round them
    assert list(embedding.buffers()) == [] and "freqs" not in embedding.state_dict()


def test_timestep_embedding_starts_at_dits_init():
    """DiT's TimestepEmbedder: the two Linears from N(0, 0.02), biases 0 (not nn.Linear's Kaiming uniform)."""
    torch.manual_seed(0)
    embedding = TimestepEmbedding(dim=384)
    for linear in (embedding.mlp.fc1, embedding.mlp.fc2):
        assert linear.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert not linear.bias.any()


def test_adaln_modulation_starts_at_zero():
    """DiT's zero init: every shift, scale and gate is 0, whatever the condition."""
    torch.manual_seed(0)
    chunks = AdaLNModulation(cond_dim=16, dim=32, num_chunks=6)(torch.randn(3, 16))
    assert len(chunks) == 6
    for chunk in chunks:
        assert chunk.shape == (3, 1, 32) and not chunk.any()


def test_adaln_modulation_chunks_are_consecutive_slices_of_the_linear_output():
    """cond -> SiLU -> Linear (B, num_chunks * dim), cut in num_chunks pieces of dim, one value per channel shared
    by every token. The parts are named as in diffusers' adaLN norms (silu, linear)."""
    torch.manual_seed(0)
    modulation = AdaLNModulation(cond_dim=16, dim=32, num_chunks=2)
    nn.init.normal_(modulation.linear.weight)
    cond = torch.randn(3, 16)
    out = nn.functional.linear(nn.functional.silu(cond), modulation.linear.weight, modulation.linear.bias)
    shift, scale = modulation(cond)
    assert torch.equal(shift[:, 0], out[:, :32]) and torch.equal(scale[:, 0], out[:, 32:])
    assert list(modulation.state_dict()) == ["linear.weight", "linear.bias"]


def test_modulation_computes_dit_formulas_and_a_missing_setting_leaves_the_tokens_unchanged():
    x, shift, scale, gate = torch.randn(4, 5, 32), torch.randn(4, 1, 32), torch.randn(4, 1, 32), torch.randn(4, 1, 32)
    modulation = Modulation(shift, scale, gate)
    assert torch.equal(modulation.modulate(x), x * (1 + scale) + shift)
    assert torch.equal(modulation.gated(x), gate * x)
    assert Modulation(shift, scale).gated(x) is x  # the final norm: no gate
    assert Modulation().modulate(x) is x and Modulation().gated(x) is x  # no adaLN


@pytest.mark.parametrize("settings", [{"shift": torch.zeros(1, 1, 4)}, {"scale": torch.zeros(1, 1, 4)}])
def test_modulation_rejects_a_shift_without_a_scale_and_the_reverse(settings):
    with pytest.raises(ValueError, match="together"):
        Modulation(**settings)
