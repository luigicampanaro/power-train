"""Every dropout of the policy against a reference implementation, plus the properties each one promises.

References, as seeded training-mode outputs and gradients (tests/references/; no reference code is imported):
    DropPath, and a block with every dropout on       -> timm's drop_path and ViT Block
    the same, in a block with cross-attention         -> torchscale's DecoderLayer
    Transformer drop path rates                       -> timm's VisionTransformer
    embedding dropout                                 -> GPT-2's embd_pdrop
    NullTokenDropout (condition and modality dropout) -> diffusers' label dropout (LabelEmbedding)
Dropout draws random masks: the same seed gives the same masks only with the torch version of the capture.
The MLPs' hidden dropout is tested with the MLPs, in tests/test_layers.py.
"""

from functools import partial

import hydra
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from diff_style.models.flow_matching_network import FlowMatchingNetwork
from diff_style.models.layers import INIT_STD, MLP, Attention, DropPath, NullTokenDropout
from diff_style.models.transformer import Transformer, TransformerBlock
from test_flow_matching import small_train_config, train_config_batch
from test_flow_matching_network import DIM, C, H, K, T1, T2, ZeroPerceiver, make_batch, make_network
from references import DROPOUT_REFERENCES as REFERENCES
from references import assert_gradients_match, with_grad
from test_transformer import vit_block_state_dict


# Drop path

def test_drop_path_matches_timm():
    ref = REFERENCES["timm_drop_path"]
    x = with_grad(ref["x"])
    torch.manual_seed(ref["seed"])
    out = DropPath(ref["p"])(x)
    assert torch.allclose(out, ref["out"], atol=1e-6)
    assert_gradients_match(out, ref, {"x": x})


def test_drop_path_is_off_in_eval_and_rejects_p_1():
    x = torch.randn(8, 3)
    assert DropPath(0.5).eval()(x) is x
    assert DropPath(0.0)(x) is x
    with pytest.raises(ValueError, match="not in"):
        DropPath(1.0)


def vit_block(ref: dict) -> TransformerBlock:
    """Our block with timm's ViT parts (LayerNorm eps 1e-6, GELU MLP, attention with biases), every dropout on."""
    block = TransformerBlock(dim=32, num_heads=4, mlp_ratio=4.0,
                             attention=partial(Attention, bias=True, dropout=ref["attn_drop"]),
                             norm=partial(nn.LayerNorm, eps=1e-6), mlp=partial(MLP, dropout=ref["proj_drop"]),
                             dropout=ref["proj_drop"], drop_path=ref["drop_path"])
    block.load_state_dict(vit_block_state_dict(ref["state_dict"]))  # strict: every parameter is mapped
    return block


def test_block_dropouts_match_timm_vit_block():
    """In training, with the attention, branch, MLP and drop path dropouts all drawing in timm's order."""
    ref = REFERENCES["timm_vit_block"]
    block = vit_block(ref)
    x = with_grad(ref["x"])
    torch.manual_seed(ref["seed"])
    out = block(x)
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x}, block, vit_block_state_dict)  # back through the same dropout masks
    block.eval()
    assert not torch.allclose(block(ref["x"]), ref["out"], atol=1e-3)  # the dropouts matter


def test_transformer_drop_path_rates_match_timm_vit():
    ref = REFERENCES["timm_vit_drop_path_rates"]
    transformer = Transformer(dim=32, depth=ref["depth"], num_heads=4, drop_path=ref["drop_path_rate"])
    assert [layer.drop_path.p for layer in transformer.layers] == pytest.approx(ref["rates"].tolist())


# torchscale's DecoderLayer names -> ours
TORCHSCALE_DECODER_LAYERS = {
    "self_attn_layer_norm": "attention_norm", "self_attn.q_proj": "attention.wq", "self_attn.k_proj": "attention.wk",
    "self_attn.v_proj": "attention.wv", "self_attn.out_proj": "attention.wo",
    "encoder_attn_layer_norm": "cross_attention_norm", "encoder_attn.q_proj": "cross_attention.wq",
    "encoder_attn.k_proj": "cross_attention.wk", "encoder_attn.v_proj": "cross_attention.wv",
    "encoder_attn.out_proj": "cross_attention.wo",
    "final_layer_norm": "ffn_norm", "ffn.fc1": "feed_forward.fc1", "ffn.fc2": "feed_forward.fc2"}


def torchscale_decoder_state_dict(reference: dict) -> dict:
    """torchscale's "<layer>.weight" -> "<our layer>.weight", the same for biases."""
    state = {}
    for key, value in reference.items():
        layer, parameter = key.rsplit(".", 1)
        state[f"{TORCHSCALE_DECODER_LAYERS[layer]}.{parameter}"] = value
    return state


def test_cross_attention_block_dropouts_match_torchscale_decoder_layer():
    """The branch timm's ViT block has no counterpart for. In training: each branch's dropout, the MLP's hidden
    dropout and drop path on all three branches, the cross-attention included (some samples skip it), drawing in
    torchscale's order. In eval: none of them."""
    ref = REFERENCES["torchscale_decoder_layer"]
    block = TransformerBlock(dim=32, num_heads=4, mlp_ratio=4.0, cross_attention=True,
                             attention=partial(Attention, bias=True), norm=nn.LayerNorm,
                             mlp=partial(MLP, dropout=ref["activation_dropout"]),
                             dropout=ref["dropout"], drop_path=ref["drop_path"])
    block.load_state_dict(torchscale_decoder_state_dict(ref["state_dict"]))  # strict: every parameter is mapped
    x, context = with_grad(ref["x"]), with_grad(ref["context"])
    torch.manual_seed(ref["seed"])
    out = block(x, context)
    assert torch.allclose(out, ref["out"], atol=1e-5)
    assert_gradients_match(out, ref, {"x": x, "context": context}, block, torchscale_decoder_state_dict)
    block.eval()
    assert torch.allclose(block(ref["x"], ref["context"]), ref["eval_out"], atol=1e-5)


# Embedding dropout

def capture_input(module: nn.Module) -> dict:
    """The first argument of each call of module, in captured["x"]."""
    captured = {}
    module.register_forward_pre_hook(lambda _, args: captured.update(x=args[0]))
    return captured


def test_embedding_dropout_matches_gpt2():
    """GPT-2: dropout(token embedding + position embedding). Ours, on the actions: dropout(action_proj(actions)
    + time embedding). One-hot actions through action_proj with GPT-2's token table are its token embedding."""
    ref = REFERENCES["gpt2_embedding_dropout"]  # vocabulary 6 = our action_dim, 4 positions = our H
    net = make_network(embedding_dropout=ref["p"])
    with torch.no_grad():
        net.action_proj.weight.copy_(ref["wte"].T)
        net.action_proj.bias.zero_()
        net.time_embedding[net.max_past:net.max_past + H] = ref["wpe"]  # offsets 0 .. H-1
    decoder_input = capture_input(net.action_decoder)
    actions, tau, obs_tokens = F.one_hot(ref["input_ids"], 6).float(), torch.rand(5), torch.randn(5, 3, DIM)
    torch.manual_seed(ref["seed"])
    net(actions, tau, obs_tokens)
    assert torch.allclose(decoder_input["x"], ref["out"], atol=1e-6)
    # Gradients of GPT-2's token table (our action_proj weight, transposed) and position table (our time rows)
    grad_proj, grad_time = torch.autograd.grad(decoder_input["x"], [net.action_proj.weight, net.time_embedding],
                                               ref["grad_out"])
    torch.testing.assert_close(grad_proj.T, ref["input_grads"]["wte"])
    torch.testing.assert_close(grad_time[net.max_past:], ref["input_grads"]["wpe"])
    assert not grad_time[:net.max_past].any()  # the past offsets: no action uses them


def test_embedding_dropout_reaches_the_obs_encoder_input_in_training_only():
    net, batch = make_network(embedding_dropout=0.3), make_batch()
    encoder_input = capture_input(net.obs_encoder)
    net.eval()
    net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    clean = encoder_input["x"]
    net.train()
    torch.manual_seed(7)
    net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    torch.manual_seed(7)
    assert torch.allclose(encoder_input["x"], F.dropout(clean, 0.3), atol=1e-6)  # the only random draw


# NullTokenDropout: condition and modality dropout

def null_token(dropout: nn.Module) -> torch.Tensor:
    """The null token of a NullTokenDropout built with p > 0."""
    assert isinstance(dropout, NullTokenDropout) and dropout.null is not None
    return dropout.null


def test_null_token_dropout_matches_diffusers_label_dropout():
    """diffusers drops a label to its extra null class; we drop the label's embedding to the null token."""
    ref = REFERENCES["diffusers_label_dropout"]
    table = ref["embedding_table"]  # rows 0..4: the classes, row 5: the null class
    dropout = NullTokenDropout(dim=8, p=ref["p"])
    with torch.no_grad():
        null_token(dropout).copy_(table[5])
    table = with_grad(table)
    x = table[ref["labels"]]
    torch.manual_seed(ref["seed"])
    out = dropout(x)
    assert torch.equal(out, ref["out"])
    assert 0 < (out != x).any(1).sum() < len(x)  # some dropped, some kept
    # Gradients: the class rows are ours, the null-class row is our null token
    grad_table, grad_null = torch.autograd.grad(out, [table, null_token(dropout)], ref["grad_out"])
    expected = ref["input_grads"]["embedding_table"]
    torch.testing.assert_close(grad_table[:5], expected[:5])
    torch.testing.assert_close(grad_null, expected[5])
    assert not grad_table[5].any()  # a dropped sample's gradient goes to the null token only
    forced = dropout.eval()(x, drop=ref["force_drop_ids"] == 1)  # diffusers' force_drop_ids: also outside training
    assert torch.equal(forced, ref["forced_out"])


def test_null_token_starts_at_trunc_normal_init_std(trunc_normal_stds):
    """A learned embedding, as the Perceiver's latents: trunc_normal(INIT_STD)."""
    dropout = NullTokenDropout(dim=8, p=0.5)
    assert trunc_normal_stds.get(id(null_token(dropout))) == INIT_STD


def test_null_token_dropout_replaces_whole_samples_and_keeps_the_dtype():
    torch.manual_seed(0)
    dropout = NullTokenDropout(dim=8, p=0.5)
    x = torch.randn(64, 3, 4, 8, dtype=torch.bfloat16)
    out = dropout(x)
    assert out.dtype == torch.bfloat16
    dropped = (out == null_token(dropout).to(torch.bfloat16)).flatten(1).all(1)
    assert torch.equal(out[~dropped], x[~dropped])  # kept samples unchanged, not rescaled
    assert 0 < dropped.sum() < 64


def test_null_token_dropout_is_off_in_eval_and_has_no_parameter_at_p_0():
    x = torch.randn(4, 8)
    assert NullTokenDropout(8, 0.5).eval()(x) is x
    assert NullTokenDropout(8)(x) is x
    assert list(NullTokenDropout(8).parameters()) == []  # nothing DDP would find unused
    with pytest.raises(ValueError, match="no null token"):
        NullTokenDropout(8)(x, drop=torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="not in"):
        NullTokenDropout(8, 1.5)


def test_null_token_dropout_drops_exactly_the_given_samples_in_training():
    """A given drop replaces the random draw: no other sample dropped, no given sample kept, no random number used."""
    dropout = NullTokenDropout(8, 0.5).train()
    x = torch.randn(6, 3, 8)
    drop = torch.tensor([True, False, False, True, False, True])
    rng_state = torch.get_rng_state()
    out = dropout(x, drop=drop)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.equal(out[drop], null_token(dropout).expand(3, 3, 8))
    assert torch.equal(out[~drop], x[~drop])
    assert torch.equal(dropout(x, drop=torch.zeros(6, dtype=torch.bool)), x)
    torch.manual_seed(0)
    random = dropout(x, drop=None)
    torch.manual_seed(0)
    assert torch.equal(random, dropout(x))  # drop=None: the random draw of the default


@pytest.mark.parametrize("drop", [torch.tensor([True]), torch.tensor(True), torch.ones(2, 3, dtype=torch.bool)],
                         ids=["one element", "0-dim", "one per token"])
def test_null_token_dropout_rejects_a_drop_that_is_not_one_value_per_sample(drop):
    """Classifier-free guidance in one batch of 2 (conditional, unconditional) needs drop = [False, True]: a
    one-element drop would broadcast to the two samples and cancel the guidance without an error."""
    dropout = NullTokenDropout(8, 0.5).eval()
    with pytest.raises(ValueError, match="one value per sample"):
        dropout(torch.randn(2, 3, 8), drop=drop)


def zero_embedding_network(**dropouts) -> FlowMatchingNetwork:
    """Obs tokens with all content and embeddings 0: the only nonzero tokens are null tokens."""
    net = make_network(**dropouts)
    net.obs_encoder = nn.Identity()
    net.perceiver = ZeroPerceiver()
    with torch.no_grad():
        for proj in [net.qpos_proj, net.past_action_proj]:
            proj.weight.zero_()
            proj.bias.zero_()
        net.time_embedding.zero_()
        net.source_embedding.zero_()
    return net


SOURCES = {f"camera {c}": slice(c * T2 * K, (c + 1) * T2 * K) for c in range(C)}  # all frames of a camera
SOURCES |= {"qpos": slice(C * T2 * K, C * T2 * K + T1), "past actions": slice(C * T2 * K + T1, C * T2 * K + 2 * T1)}


def dropped_sources(net: FlowMatchingNetwork, null: torch.Tensor, b: int = 32) -> dict:
    """For each source, which samples hold only null tokens there. Fails if a source is partly dropped."""
    batch = make_batch(b)
    tokens = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"]).detach()
    dropped = {}
    for name, positions in SOURCES.items():
        is_null = (tokens[:, positions] == null).all(-1)  # (B, tokens of the source)
        is_zero = (tokens[:, positions] == 0).all(-1)
        assert torch.equal(is_null.all(1) | is_zero.all(1), torch.ones(b, dtype=torch.bool)), name  # whole or not at all
        dropped[name] = is_null.all(1)
    return dropped


def test_camera_dropout_drops_each_camera_of_each_sample_whole():
    net = zero_embedding_network(camera_dropout=0.5)
    dropped = dropped_sources(net, null_token(net.camera_dropout))
    for c in range(C):
        assert 0 < dropped[f"camera {c}"].sum() < 32
    assert not torch.equal(dropped["camera 0"], dropped["camera 1"])  # drawn per camera
    assert not dropped["qpos"].any() and not dropped["past actions"].any()


@pytest.mark.parametrize("source", ["qpos", "past actions"])
def test_proprio_dropout_drops_only_its_own_tokens(source):
    name = {"qpos": "qpos_dropout", "past actions": "past_action_dropout"}[source]
    net = zero_embedding_network(**{name: 0.5})
    dropped = dropped_sources(net, null_token(getattr(net, name)))
    assert 0 < dropped[source].sum() < 32
    assert [other for other in SOURCES if other != source and dropped[other].any()] == []


def test_dropped_inputs_keep_their_source_and_time_embeddings():
    """The null token replaces the content only: the network still knows which input is missing."""
    net = make_network(camera_dropout=1.0, qpos_dropout=1.0, past_action_dropout=1.0)
    net.obs_encoder = nn.Identity()
    batch = make_batch()
    tokens = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    source, time = net.source_embedding, net.time
    expected = torch.cat([(null_token(net.camera_dropout) + source[:C, None, None] + time(-(T2 - 1), 1)[:, None])
                          .expand(C, T2, K, DIM).flatten(0, 2),
                          null_token(net.qpos_dropout) + source[C] + time(-(T1 - 1), 1),
                          null_token(net.past_action_dropout) + source[C + 1] + time(-T1, 0)])
    assert torch.allclose(tokens, expected.expand(3, -1, -1), atol=1e-6)


def test_condition_dropout_replaces_every_obs_token_and_can_be_forced():
    net, batch = make_network(condition_dropout=0.5), make_batch(32)
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    clean = net.eval().encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    dropped = (obs == null_token(net.condition_dropout)).all(-1).all(-1)
    assert torch.equal(obs[~dropped], clean[~dropped])
    assert 0 < dropped.sum() < 32
    # The unconditional pass of classifier-free guidance, at sampling time
    unconditional = net.condition_dropout(clean, drop=torch.ones(32, dtype=torch.bool))
    assert torch.equal(unconditional, null_token(net.condition_dropout).expand_as(clean))


ALL_DROPOUTS = {"embedding_dropout": 0.3, "camera_dropout": 0.5, "qpos_dropout": 0.5, "past_action_dropout": 0.5,
                "condition_dropout": 0.5}


def test_network_dropouts_are_off_in_eval():
    net, plain = make_network(**ALL_DROPOUTS).eval(), make_network()
    plain.load_state_dict(net.state_dict(), strict=False)  # the same weights, without the null tokens
    batch, tau = make_batch(), torch.rand(3)
    for model in [net, plain]:
        obs = model.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
        model.velocity = model(batch["actions"], tau, obs)
    assert torch.equal(net.velocity, plain.velocity)


def test_every_parameter_gets_a_gradient_with_every_dropout_on():
    """Null tokens included, whether or not a sample was dropped: DDP needs a gradient for every parameter."""
    net = make_network(**ALL_DROPOUTS)
    net.obs_encoder = Transformer(DIM, depth=2, num_heads=4, drop_path=0.5)
    nn.init.normal_(net.velocity_head.weight)  # out of the zero init, so the gradients reach the inputs
    batch = make_batch(2)
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    net(batch["actions"], torch.rand(2), obs).square().sum().backward()
    assert [name for name, p in net.named_parameters() if p.grad is None] == []


# Hydra

@pytest.mark.parametrize("mlp", ["swiglu", "gelu"])
def test_every_dropout_is_set_from_the_hydra_config(mlp):
    modules = ["obs_encoder", "action_decoder"]
    cfg = small_train_config([
        "output_dir=unused",
        *[f"model.net.{name}={p}" for name, p in ALL_DROPOUTS.items()],
        *[f"model/mlp@model.net.{module}.mlp={mlp}" for module in modules],
        *[f"model.net.{module}.mlp.dropout=0.2" for module in modules],
        *[f"model.net.{module}.drop_path=0.3" for module in modules],
        *[f"model.net.{module}.dropout=0.1" for module in modules],
    ])
    policy = hydra.utils.instantiate(cfg.model)
    net = policy.net
    for name, p in ALL_DROPOUTS.items():
        assert getattr(net, name).p == p
    for module in [net.obs_encoder, net.action_decoder]:
        depth = len(module.layers)
        assert [layer.drop_path.p for layer in module.layers] == pytest.approx([0.3 * i / (depth - 1) for i in range(depth)])
        assert {layer.feed_forward.dropout.p for layer in module.layers} == {0.2}
        assert {layer.dropout.p for layer in module.layers} == {0.1}
    assert torch.isfinite(policy.loss(train_config_batch()))
