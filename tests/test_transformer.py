"""TransformerBlock and Transformer against reference blocks, built with the reference's parts:
    Llama defaults (RMSNorm, SwiGLU)             -> torchtitan's Llama 3 block, RoPE off, and its depth init
    LayerNorm, tanh-GELU MLP, attention biases   -> diffusers' adaLN-Zero block; two of them with cross-attention
                                                    and its adaLN final norm (a decoder, in float64)
    RMSNorm (the default) in the final norm      -> diffusers' adaLN final norm with RMSNorm
    LayerNorm, GELU MLP, attention biases        -> V-JEPA 2's ViT depth init (the init of the other parts)
    LayerNorm, GELU MLP, cross-attention         -> torch's nn.TransformerDecoderLayer
plus the properties of the conditioning and of the cross-attention. Outputs and gradients (tests/references/)."""

from functools import partial

import pytest
import torch
from torch import nn

from diff_style.models.layers import MLP, Attention, causal_mask
from diff_style.models.transformer import Transformer, TransformerBlock
from references import REFERENCES, assert_gradients_match, with_grad


@pytest.mark.parametrize("causal_flag", [False, True])  # a causal mask tensor, or causal=True
def test_block_matches_torchtitan_llama_block(causal_flag):
    """RMSNorm -> causal self-attention -> RMSNorm -> SwiGLU, with the reference's weights (RoPE off)."""
    ref = REFERENCES["llama_block"]
    block = TransformerBlock(dim=32, num_heads=4, causal=causal_flag)
    block.load_state_dict(ref["state_dict"])  # strict: the same parameter names and SwiGLU width
    x = with_grad(ref["x"])
    out = block(x) if causal_flag else block(x, mask=causal_mask(10, 10))
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x}, block)
    if not causal_flag:
        assert not torch.allclose(block(ref["x"]), ref["out"], atol=1e-3)  # causality matters


# diffusers' BasicTransformerBlock layer names -> ours
ADALN_ZERO_BLOCK_LAYERS = {"norm1.linear": "adaLN_modulation.linear", "attn1.to_q": "attention.wq",
                           "attn1.to_k": "attention.wk", "attn1.to_v": "attention.wv", "attn1.to_out.0": "attention.wo",
                           "attn2.to_q": "cross_attention.wq", "attn2.to_k": "cross_attention.wk",
                           "attn2.to_v": "cross_attention.wv", "attn2.to_out.0": "cross_attention.wo",
                           "ff.net.0.proj": "feed_forward.fc1", "ff.net.2": "feed_forward.fc2"}


def adaln_zero_block_state_dict(reference: dict) -> dict:
    """diffusers' parameter names -> ours: "<layer>.weight" -> "<our layer>.weight", the same for biases.
    Its condition embedder (norm1.emb) makes c, the input of our block: it has no counterpart."""
    state = {}
    for key, value in reference.items():
        layer, parameter = key.rsplit(".", 1)
        if not layer.startswith("norm1.emb."):
            state[f"{ADALN_ZERO_BLOCK_LAYERS[layer]}.{parameter}"] = value
    return state


def adaln_final_norm_state_dict(reference: dict) -> dict:
    """diffusers' AdaLayerNormContinuous -> our final norm. Its projection gives the scale, then the shift; ours the
    shift, then the scale (as DiT): the two halves change places."""
    return {f"norm_modulation.linear.{parameter}": torch.cat(reference[f"linear.{parameter}"].chunk(2)[::-1])
            for parameter in ["weight", "bias"]}


def adaln_zero_decoder_state_dict(reference: dict) -> dict:
    """diffusers' blocks.<i> -> our layers.<i>, each as one block; its norm_out -> our final norm."""
    def part(prefix: str) -> dict:
        return {key.removeprefix(prefix): value for key, value in reference.items() if key.startswith(prefix)}

    state = adaln_final_norm_state_dict(part("norm_out."))
    depth = len({key.split(".")[1] for key in reference if key.startswith("blocks.")})
    for i in range(depth):
        block_state = adaln_zero_block_state_dict(part(f"blocks.{i}."))
        state |= {f"layers.{i}.{key}": value for key, value in block_state.items()}
    return state


# timm's and V-JEPA 2's ViT block names -> ours (they keep q, k and v in one `qkv` Linear: split)
VIT_BLOCK_LAYERS = {"norm1": "attention_norm", "attn.proj": "attention.wo", "norm2": "ffn_norm",
                    "mlp.fc1": "feed_forward.fc1", "mlp.fc2": "feed_forward.fc2"}


def vit_block_state_dict(reference: dict) -> dict:
    """A ViT block's "<layer>.weight" -> "<our layer>.weight", the same for biases; attn.qkv -> wq, wk, wv."""
    state = {}
    for key, value in reference.items():
        layer, parameter = key.rsplit(".", 1)
        if layer == "attn.qkv":
            for name, chunk in zip(["wq", "wk", "wv"], value.chunk(3)):
                state[f"attention.{name}.{parameter}"] = chunk
        else:
            state[f"{VIT_BLOCK_LAYERS[layer]}.{parameter}"] = value
    return state


# The parts of the DiT block: attention with biases, LayerNorm (eps 1e-6), tanh-GELU MLP
DIT_ATTENTION = partial(Attention, bias=True)
DIT_NORM = partial(nn.LayerNorm, eps=1e-6)
DIT_MLP = partial(MLP, activation=partial(nn.GELU, approximate="tanh"))


def test_adaln_block_matches_diffusers_adaln_zero_block():
    ref = REFERENCES["diffusers_adaln_zero_block"]
    block = TransformerBlock(dim=32, num_heads=4, mlp_ratio=4.0, cond_dim=32,
                             attention=DIT_ATTENTION, norm=DIT_NORM, mlp=DIT_MLP)
    block.load_state_dict(adaln_zero_block_state_dict(ref["state_dict"]))  # strict: every parameter is mapped
    x, c = with_grad(ref["x"]), with_grad(ref["c"])
    out = block(x, cond=c)
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x, "c": c}, block, adaln_zero_block_state_dict)


def test_conditioned_final_norm_matches_diffusers_adaln_final_norm_with_rmsnorm():
    """Our default norm, RMSNorm (eps 1e-5), against diffusers' AdaLayerNormContinuous with norm_type "rms_norm"."""
    ref = REFERENCES["diffusers_adaln_rms_final_norm"]
    transformer = Transformer(dim=32, depth=0, num_heads=4, cond_dim=32)
    transformer.load_state_dict(adaln_final_norm_state_dict(ref["state_dict"]))
    x, c = with_grad(ref["x"]), with_grad(ref["c"])
    out = transformer(x, cond=c)
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x, "c": c}, transformer, adaln_final_norm_state_dict)


def test_conditioned_cross_attention_decoder_matches_diffusers_in_float64():
    """Two diffusers adaLN-Zero blocks with cross-attention, then its adaLN final norm, against our Transformer built
    from config only (no affine weights in any norm, as diffusers'). The cross-attention is not modulated in either.
    Float64, at float64 tolerances: the same formulas, not only close ones."""
    ref = REFERENCES["diffusers_adaln_zero_decoder"]
    decoder = Transformer(dim=32, depth=2, num_heads=4, mlp_ratio=4.0, cross_attention=True, cond_dim=32,
                          attention=DIT_ATTENTION, norm=partial(DIT_NORM, elementwise_affine=False), mlp=DIT_MLP)
    decoder.double().load_state_dict(adaln_zero_decoder_state_dict(ref["state_dict"]))  # strict
    x, context, c = with_grad(ref["x"]), with_grad(ref["context"]), with_grad(ref["c"])
    out = decoder(x, context=context, cond=c)
    torch.testing.assert_close(out, ref["out"])  # float64: rtol and atol 1e-7
    assert_gradients_match(out, ref, {"x": x, "context": context, "c": c}, decoder, adaln_zero_decoder_state_dict)


def decoder_layer_state_dict(reference: dict) -> dict:
    """torch's nn.TransformerDecoderLayer names -> ours; its attention keeps q, k and v in one in_proj: split it."""
    state = {"attention_norm.weight": reference["norm1.weight"], "cross_attention_norm.weight": reference["norm2.weight"],
             "ffn_norm.weight": reference["norm3.weight"], "feed_forward.fc1.weight": reference["linear1.weight"],
             "feed_forward.fc2.weight": reference["linear2.weight"]}
    for theirs, ours in [("self_attn", "attention"), ("multihead_attn", "cross_attention")]:
        for key, chunk in zip(["wq", "wk", "wv"], reference[f"{theirs}.in_proj_weight"].chunk(3)):
            state[f"{ours}.{key}.weight"] = chunk
        state[f"{ours}.wo.weight"] = reference[f"{theirs}.out_proj.weight"]
    return state


@pytest.mark.parametrize("causal_flag", [False, True])  # a causal mask tensor, or causal=True
def test_cross_attention_block_matches_torch_transformer_decoder_layer(causal_flag):
    """Pre-norm self-attention -> cross-attention -> MLP, all without biases, with a causal self-attention
    and a cross-attention mask. causal=True makes only the self-attention causal, as torch's tgt_is_causal."""
    ref = REFERENCES["torch_decoder_layer"]
    block = TransformerBlock(dim=32, num_heads=4, mlp_ratio=4.0, cross_attention=True, causal=causal_flag,
                             norm=partial(nn.LayerNorm, bias=False), mlp=partial(MLP, bias=False))
    block.load_state_dict(decoder_layer_state_dict(ref["state_dict"]))  # strict: every parameter is mapped
    x, context = with_grad(ref["x"]), with_grad(ref["context"])
    out = block(x, context, mask=None if causal_flag else ref["causal"], context_mask=ref["keep"])
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x, "context": context}, block, decoder_layer_state_dict)


def test_transformer_gives_its_dropout_and_causal_to_every_block():
    transformer = Transformer(dim=32, depth=2, num_heads=4, cross_attention=True, causal=True, dropout=0.3)
    assert [layer.dropout.p for layer in transformer.layers] == [0.3, 0.3]
    assert [layer.attention.is_causal for layer in transformer.layers] == [True, True]
    assert [layer.cross_attention.is_causal for layer in transformer.layers] == [False, False]  # sees all the context


def test_adaln_zero_block_starts_as_the_identity():
    """DiT's zero init: every gate is 0, so the block returns its input whatever the condition."""
    torch.manual_seed(0)
    block = TransformerBlock(dim=32, num_heads=4, cond_dim=16)
    x = torch.randn(3, 6, 32)
    assert torch.equal(block(x, cond=torch.randn(3, 16)), x)


def test_conditioned_transformer_starts_as_a_plain_rms_norm():
    torch.manual_seed(0)
    transformer = Transformer(dim=32, depth=2, num_heads=4, cond_dim=16)
    x = torch.randn(3, 6, 32)
    assert torch.allclose(transformer(x, cond=torch.randn(3, 16)), transformer.norm(x))


def test_init_matches_torchtitan_llama3_depth_init():
    """torchtitan's Llama 3 init: wq, wk, wv, w1 from trunc_normal(0.02); wo, w2, w3 from 0.02 / sqrt(2 (l + 1)) in
    block l, so deeper blocks start smaller; norm weights 1. The same seed gives the same weights: the same
    distributions, drawn in the same order."""
    ref = REFERENCES["llama_depth_init"]
    transformer = Transformer(dim=32, depth=ref["depth"], num_heads=4)
    torch.manual_seed(ref["seed"])
    transformer.init_weights()
    blocks = blocks_state_dict(transformer)
    assert blocks.keys() == ref["state_dict"].keys()
    for key, value in ref["state_dict"].items():
        # The same seed and draw order, up to the last bit: trunc_normal_'s erfinv differs by a few ULPs across
        # CPUs and BLAS libraries. A wrong std, layer or draw order is orders of magnitude larger.
        torch.testing.assert_close(blocks[key], value, rtol=1e-5, atol=1e-8,
                                   msg=lambda message: f"{key}: {message}")


def test_init_with_vit_parts_matches_vjepa2_depth_init():
    """V-JEPA 2's ViT (Meta): the same depth rule, for the parts torchtitan's block has not: attention biases,
    LayerNorm, GELU MLP. Linears from trunc_normal(0.02), biases 0, LayerNorms 1 and 0, then block l's attn.proj and
    mlp.fc2 divided by sqrt(2 (l + 1)). It divides after the draw, we draw with the smaller std: the same values, up
    to the last bit of float32."""
    ref = REFERENCES["vjepa2_depth_init"]
    transformer = Transformer(dim=32, depth=ref["depth"], num_heads=4, mlp_ratio=4.0,
                              attention=DIT_ATTENTION, norm=nn.LayerNorm, mlp=MLP)
    torch.manual_seed(ref["seed"])
    transformer.init_weights()
    expected = {}
    for i in range(ref["depth"]):
        block = {key.removeprefix(f"{i}."): value for key, value in ref["state_dict"].items()
                 if key.startswith(f"{i}.")}
        expected |= {f"{i}.{key}": value for key, value in vit_block_state_dict(block).items()}
    blocks = blocks_state_dict(transformer)
    assert blocks.keys() == expected.keys()
    for key, value in expected.items():
        # As above: the same draws, up to the last bits of erfinv, which are library- and CPU-specific.
        torch.testing.assert_close(blocks[key], value, rtol=1e-5, atol=1e-8, msg=key)


def test_init_gives_the_cross_attention_the_attention_init_and_keeps_adaln_at_zero():
    """The parts no reference has: the cross-attention writes back into the stream as the self-attention does (q, k, v
    at 0.02, wo at 0.02 / sqrt(2 (l + 1)), biases 0), and the adaLN Linears keep their zero init (DiT)."""
    torch.manual_seed(0)
    decoder = Transformer(dim=256, depth=2, num_heads=4, cross_attention=True, cond_dim=16, attention=DIT_ATTENTION)
    for layer_id, layer in enumerate(decoder.layers):
        cross_attention = layer.cross_attention
        assert cross_attention.wq.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert cross_attention.wo.weight.std().item() == pytest.approx(0.02 / (2 * (layer_id + 1)) ** 0.5, rel=0.05)
        assert not cross_attention.wq.bias.any() and not cross_attention.wo.bias.any()
    assert not any(p.any() for name, p in decoder.named_parameters() if "modulation" in name)


def blocks_state_dict(transformer: Transformer) -> dict:
    """The blocks' parameters, keyed "<block>.<parameter>" as a ModuleList of blocks."""
    return {key.removeprefix("layers."): value for key, value in transformer.state_dict().items()
            if key.startswith("layers.")}


# adaLN output order, as DiT: shift, scale, gate of the self-attention, then of the MLP
@pytest.mark.parametrize("chunk, expected", [
    ("gate_attn", lambda b, x: x + b.attention(b.attention_norm(x))),
    ("scale_attn+gate_attn", lambda b, x: x + b.attention(2 * b.attention_norm(x))),
    ("shift_attn+gate_attn", lambda b, x: x + b.attention(b.attention_norm(x) + 1)),
    ("gate_ffn", lambda b, x: x + b.feed_forward(b.ffn_norm(x))),
    ("scale_ffn+gate_ffn", lambda b, x: x + b.feed_forward(2 * b.ffn_norm(x))),
    ("shift_ffn+gate_ffn", lambda b, x: x + b.feed_forward(b.ffn_norm(x) + 1)),
])
def test_adaln_chunks_are_shift_scale_gate_for_attention_then_mlp(chunk, expected):
    torch.manual_seed(0)
    order = ["shift_attn", "scale_attn", "gate_attn", "shift_ffn", "scale_ffn", "gate_ffn"]
    block = TransformerBlock(dim=32, num_heads=4, cond_dim=16)
    with torch.no_grad():  # weight 0: the modulation is the bias, 1 for the named chunks, 0 for the others
        assert block.adaLN_modulation is not None
        bias = block.adaLN_modulation.linear.bias.view(6, 32)
        for name in chunk.split("+"):
            bias[order.index(name)] = 1.0
    x = torch.randn(2, 5, 32)
    assert torch.allclose(block(x, cond=torch.randn(2, 16)), expected(block, x), atol=1e-6)


def test_each_sample_is_modulated_by_its_own_condition():
    torch.manual_seed(0)
    block = TransformerBlock(dim=32, num_heads=4, cond_dim=16)
    assert block.adaLN_modulation is not None
    torch.nn.init.normal_(block.adaLN_modulation.linear.weight)  # out of the zero init
    x, cond = torch.randn(3, 5, 32), torch.randn(3, 16)
    together = block(x, cond=cond)
    for i in range(3):
        assert torch.allclose(together[i], block(x[i:i + 1], cond=cond[i:i + 1])[0], atol=1e-5)
    assert not torch.allclose(together[0], block(x[:1], cond=cond[1:2])[0], atol=1e-3)


def test_cross_attention_reads_the_context_except_masked_tokens():
    torch.manual_seed(0)
    block = TransformerBlock(dim=32, num_heads=4, cross_attention=True)
    x, context = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    changed = context.clone()
    changed[:, 6] = 100.0
    assert not torch.allclose(block(x, context), block(x, changed), atol=1e-3)
    keep = torch.ones(1, 7, dtype=torch.bool)  # (queries, keys), broadcast to every query
    keep[:, 6] = False
    assert torch.allclose(block(x, context, context_mask=keep), block(x, changed, context_mask=keep), atol=1e-6)
