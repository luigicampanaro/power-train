"""Save seeded inputs, weights and outputs of the reference implementations that the policy is
tested against (tests/test_layers.py, test_transformer.py, test_perceiver.py, test_flow_matching.py) to
outputs.pt, next to this file. For every layer, also the gradients of its inputs and parameters for a fixed
upstream gradient (capture_gradients.py), which the tests send back through our layers.

The references are not dependencies of diff_style: torchtitan v0.1.0 needs a newer torch than ours.
Every reference has a permissive license: torchtitan (BSD-3-Clause), Hugging Face transformers and
diffusers (Apache-2.0), torchcfm, torchdiffeq and V-JEPA 2 (MIT). Clone torchtitan and V-JEPA 2 (no release tag:
its last commit) outside, then run from the diff_style folder:

    git clone --depth 1 --branch v0.1.0 https://github.com/pytorch/torchtitan /tmp/references/torchtitan_v0.1.0
    git clone https://github.com/facebookresearch/vjepa2 /tmp/references/vjepa2_204698b
    git -C /tmp/references/vjepa2_204698b checkout 204698b45b3712590f06245fbfba32d3be539812
    PYTHONPATH=/tmp/references/torchtitan_v0.1.0:/tmp/references/vjepa2_204698b \
        uv run --no-project --python 3.12 \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        --with torch==2.8.0 --with torchvision==0.23.0 --with transformers==4.57.1 --with diffusers==0.35.2 \
        --with torchcfm==1.0.7 --with pot --with torchdiffeq==0.2.5 \
        --with tyro --with timm --with datasets --with torchdata --with tiktoken --with blobfile --with tensorboard \
        --with einops \
        python tests/references/capture_outputs.py

(torchtitan imports its whole training stack, hence the long --with list; torchcfm imports pot, V-JEPA 2 einops.)
Every capture is seeded.
"""

from pathlib import Path

import torch
from torch import nn
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormContinuous
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdiffeq import odeint
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import (FeedForward, TransformerBlock, apply_rotary_emb,
                                                  precompute_freqs_cis)
from transformers import LlamaConfig, Qwen3Config, get_cosine_schedule_with_warmup
from transformers.models.idefics2.configuration_idefics2 import Idefics2PerceiverConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2MLP, Idefics2PerceiverLayer
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from src.models.vision_transformer import VisionTransformer as VJEPA2VisionTransformer  # V-JEPA 2's package is `src`

from capture_gradients import gradients

OUTPUT = Path(__file__).resolve().parent / "outputs.pt"


def llama_rope() -> dict:
    torch.manual_seed(0)
    xq, xk = torch.randn(2, 10, 4, 16).requires_grad_(), torch.randn(2, 10, 4, 16).requires_grad_()  # (B, L, heads, head_dim)
    q, k = apply_rotary_emb(xq, xk, precompute_freqs_cis(16, 10))  # theta 10000
    return {"xq": xq.detach(), "xk": xk.detach(), "q": q.detach(), "k": k.detach(),
            **gradients((q, k), {"xq": xq, "xk": xk})}


def llama_feed_forward() -> dict:
    torch.manual_seed(1)
    ff = FeedForward(dim=32, hidden_dim=4 * 32, multiple_of=1, ffn_dim_multiplier=None)  # hidden int(2/3 * 128) = 85
    x = torch.randn(2, 5, 32).requires_grad_()
    out = ff(x)
    return {"state_dict": ff.state_dict(), "x": x.detach(), "out": out.detach(), **gradients(out, {"x": x}, ff)}


def llama_ffn_hidden_dims() -> dict:
    """The SwiGLU hidden width of torchtitan's FeedForward for hidden_dim = 4 * dim (int(2/3 * 4 dim), rounded up to
    multiple_of), for every dim from 32 to 1024 in steps of 32, 2048 and 4096; multiple_of 256 (Meta's default)
    and 1024 (torchtitan's 8B). Built on the meta device: shapes only."""
    dims = [*range(32, 1025, 32), 2048, 4096]
    cases = [(dim, multiple_of) for multiple_of in (256, 1024) for dim in dims]
    with torch.device("meta"):
        widths = [FeedForward(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=None).w1.out_features
                  for dim, multiple_of in cases]
    return {"dim": torch.tensor([dim for dim, _ in cases]), "multiple_of": torch.tensor([m for _, m in cases]),
            "hidden_dim": torch.tensor(widths)}


def hf_llama_mlp(bias: bool) -> dict:
    """Hugging Face's LlamaMLP, down_proj(silu(gate_proj x) * up_proj x), with or without biases (mlp_bias)."""
    torch.manual_seed(10 + bias)
    mlp = LlamaMLP(LlamaConfig(hidden_size=32, intermediate_size=85, hidden_act="silu", mlp_bias=bias))
    x = torch.randn(2, 5, 32).requires_grad_()
    out = mlp(x)
    return {"state_dict": mlp.state_dict(), "x": x.detach(), "out": out.detach(), **gradients(out, {"x": x}, mlp)}


def llama_block() -> dict:
    """Pre-norm block with causal self-attention, RMSNorm (eps 1e-5) and SwiGLU. RoPE is turned off (every
    rotation 1), as in our Transformer; RoPE itself is captured in llama_rope."""
    torch.manual_seed(2)
    args = TransformerModelArgs(dim=32, n_heads=4, multiple_of=1)  # SDPA with is_causal=True
    block = TransformerBlock(layer_id=0, model_args=args)
    block.init_weights()
    with torch.no_grad():  # norm weights start at 1: make them random, so a missing norm weight shows
        block.attention_norm.weight.uniform_(0.5, 1.5)
        block.ffn_norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 10, 32).requires_grad_()
    no_rotation = torch.ones(10, 32 // 4 // 2, dtype=torch.complex64)  # precompute_freqs_cis's shape, angle 0
    out = block(x, no_rotation)
    return {"state_dict": block.state_dict(), "x": x.detach(), "out": out.detach(), **gradients(out, {"x": x}, block)}


def llama_depth_init() -> dict:
    """torchtitan's Llama 3 init (depth_init, its default) of 3 blocks, in the order Transformer.init_weights runs
    them, from one seed: wq, wk, wv, w1 from trunc_normal(0.02); wo, w2, w3 from trunc_normal(0.02 / sqrt(2 (layer_id
    + 1))); norm weights 1. Its token embedding and output layer, which our Transformer has not, are left out."""
    args = TransformerModelArgs(dim=32, n_heads=4, n_layers=3, multiple_of=1)
    layers = nn.ModuleList(TransformerBlock(layer_id, args) for layer_id in range(args.n_layers))
    torch.manual_seed(14)
    for layer in layers:
        layer.init_weights()
    return {"seed": 14, "depth": args.n_layers, "state_dict": layers.state_dict()}


def vjepa2_depth_init() -> dict:
    """V-JEPA 2's ViT init (Meta's own code), the same depth rule for the parts torchtitan's block has not (attention
    biases, LayerNorm, GELU MLP): every Linear from trunc_normal(0.02), biases 0, LayerNorms 1 and 0, then block l's
    attn.proj and mlp.fc2 divided by sqrt(2 (l + 1)). Its own init functions, run on its blocks only (its patch
    embedding, which our Transformer has not, is left out)."""
    vit = VJEPA2VisionTransformer(img_size=(32, 32), patch_size=16, embed_dim=32, depth=3, num_heads=4, mlp_ratio=4.0,
                                  qkv_bias=True)
    torch.manual_seed(15)
    vit.blocks.apply(vit._init_weights)
    vit._rescale_blocks()
    return {"seed": 15, "depth": 3, "state_dict": vit.blocks.state_dict()}


def diffusers_timestep_embedding() -> dict:
    """diffusers' Timesteps (cos first, no frequency shift, 256 features), then its TimestepEmbedding MLP."""
    torch.manual_seed(3)
    features, embedder = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0), TimestepEmbedding(256, 32)
    tau = torch.cat([torch.tensor([0.0, 1.0]), torch.rand(6)]).requires_grad_()  # flow times, both ends included
    out = embedder(features(tau * 1000))
    return {"state_dict": embedder.state_dict(), "tau": tau.detach(), "out": out.detach(),
            **gradients(out, {"tau": tau}, embedder)}


def diffusers_adaln_zero_block() -> dict:
    """diffusers' BasicTransformerBlock with adaLN-Zero, unchanged: LayerNorm without affine (eps 1e-6), attention
    with biases, tanh-GELU MLP. It makes its condition c from a timestep and a class label (norm1.emb); c, read
    with a hook, is the input of our block. Evaluation mode: no class-label dropout in c. Not zero-initialised
    (the block alone keeps its random init), so every branch counts."""
    torch.manual_seed(5)
    block = BasicTransformerBlock(32, num_attention_heads=4, attention_head_dim=8, activation_fn="gelu-approximate",
                                  num_embeds_ada_norm=10, attention_bias=True, norm_type="ada_norm_zero",
                                  norm_elementwise_affine=False, norm_eps=1e-6).eval()
    captured = {}
    block.norm1.emb.register_forward_hook(lambda module, args, c: captured.update(c=c))
    x = torch.randn(2, 5, 32).requires_grad_()
    out = block(x, timestep=torch.rand(2) * 1000, class_labels=torch.randint(0, 10, (2,)))
    c = captured["c"]
    return {"state_dict": block.state_dict(), "x": x.detach(), "c": c.detach(), "out": out.detach(),
            **gradients(out, {"x": x, "c": c}, block)}


def diffusers_adaln_zero_decoder() -> dict:
    """Two diffusers BasicTransformerBlocks with adaLN-Zero and cross-attention, then AdaLayerNormContinuous, unchanged,
    in float64: LayerNorm without affine (eps 1e-6) everywhere, attention with biases, tanh-GELU MLP. Each block
    makes its condition c from a timestep and a class label (norm1.emb). Block 1's embedder gets block 0's weights,
    so the two make the same c, which hooks read; the final norm gets it too. Evaluation mode: no label dropout."""
    torch.manual_seed(13)
    blocks = nn.ModuleList(
        BasicTransformerBlock(32, num_attention_heads=4, attention_head_dim=8, cross_attention_dim=32,
                              activation_fn="gelu-approximate", num_embeds_ada_norm=10, attention_bias=True,
                              norm_type="ada_norm_zero", norm_elementwise_affine=False, norm_eps=1e-6)
        for _ in range(2))
    blocks[1].norm1.emb.load_state_dict(blocks[0].norm1.emb.state_dict())
    decoder = nn.ModuleDict({"blocks": blocks,
                             "norm_out": AdaLayerNormContinuous(32, 32, elementwise_affine=False, eps=1e-6)})
    decoder.double().eval()
    conditions = []
    for block in blocks:
        block.norm1.emb.register_forward_hook(lambda module, args, c: conditions.append(c))
    x = torch.randn(2, 5, 32, dtype=torch.float64).requires_grad_()
    context = torch.randn(2, 7, 32, dtype=torch.float64).requires_grad_()
    timestep, class_labels = torch.rand(2) * 1000, torch.randint(0, 10, (2,))
    hidden = x
    for block in blocks:
        hidden = block(hidden, encoder_hidden_states=context, timestep=timestep, class_labels=class_labels)
    assert torch.equal(conditions[0], conditions[1]), "the two blocks must get the same condition"
    out = decoder["norm_out"](hidden, conditions[0])
    reference = gradients(out, {"x": x, "context": context, "c0": conditions[0], "c1": conditions[1]}, decoder)
    input_grads = reference["input_grads"]
    # Our decoder gives one c to every block: its gradient is the sum of the gradients of the blocks' c
    reference["input_grads"] = {"x": input_grads["x"], "context": input_grads["context"],
                                "c": input_grads["c0"] + input_grads["c1"]}
    return {"state_dict": decoder.state_dict(), "x": x.detach(), "context": context.detach(),
            "c": conditions[0].detach(), "out": out.detach(), **reference}


def diffusers_adaln_rms_final_norm() -> dict:
    """diffusers' AdaLayerNormContinuous with RMSNorm without affine (eps 1e-5, our default norm), then a scale and a
    shift from c. float32: diffusers' RMSNorm computes the variance in float32 anyway."""
    torch.manual_seed(6)
    norm = AdaLayerNormContinuous(32, 32, elementwise_affine=False, eps=1e-5, norm_type="rms_norm")
    x, c = torch.randn(2, 5, 32).requires_grad_(), torch.randn(2, 32).requires_grad_()
    out = norm(x, c)
    return {"state_dict": norm.state_dict(), "x": x.detach(), "c": c.detach(), "out": out.detach(),
            **gradients(out, {"x": x, "c": c}, norm)}


def qwen3_attention() -> dict:
    """Attention with per-head QK RMSNorm (eps 1e-5), causal. RoPE is turned off (cos 1, sin 0): Qwen3
    rotates the two halves of each head, torchtitan (and we) rotate pairs, tested separately."""
    torch.manual_seed(7)
    config = Qwen3Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8,
                         rms_norm_eps=1e-5, attention_bias=False)
    config._attn_implementation = "sdpa"  # causal when no mask is given
    attention = Qwen3Attention(config, layer_idx=0)
    with torch.no_grad():  # norm weights start at 1: make them random, so a missing norm weight shows
        attention.q_norm.weight.uniform_(0.5, 1.5)
        attention.k_norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 5, 32).requires_grad_()
    out, _ = attention(x, position_embeddings=(torch.ones(1, 5, 8), torch.zeros(1, 5, 8)), attention_mask=None)
    return {"state_dict": attention.state_dict(), "x": x.detach(), "out": out.detach(),
            **gradients(out, {"x": x}, attention)}


def qwen3_attention_padded() -> dict:
    """qwen3_attention with a padded batch: sample 1's last two keys are padding. Qwen3 takes the causal
    and padding masks as one additive mask (0 = attend, very negative = not); `keep` is the same as booleans."""
    torch.manual_seed(9)
    config = Qwen3Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8,
                         rms_norm_eps=1e-5, attention_bias=False)
    config._attn_implementation = "sdpa"
    attention = Qwen3Attention(config, layer_idx=0)
    with torch.no_grad():
        attention.q_norm.weight.uniform_(0.5, 1.5)
        attention.k_norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 7, 32).requires_grad_()
    padding = torch.ones(2, 7, dtype=torch.bool)
    padding[1, 5:] = False
    keep = padding[:, None, None, :] & torch.ones(7, 7, dtype=torch.bool).tril()  # (B, 1, Lq, Lk)
    additive = torch.zeros(2, 1, 7, 7).masked_fill(~keep, torch.finfo(torch.float32).min)
    out, _ = attention(x, position_embeddings=(torch.ones(1, 7, 8), torch.zeros(1, 7, 8)), attention_mask=additive)
    return {"state_dict": attention.state_dict(), "x": x.detach(), "keep": keep, "out": out.detach(),
            **gradients(out, {"x": x}, attention)}


def idefics2_perceiver() -> dict:
    """Idefics2's connector, the structure of our Perceiver: its SwiGLU modality projection (Idefics2MLP, width
    12 -> 32), then one Idefics2PerceiverLayer: latents + attention(RMSNorm(latents), RMSNorm(context)). Its two
    extras are off by its own means: the additive mask hides the latents' own keys (Idefics2 appends them to the
    context's) and the context padding, and its MLP on the latents gets down_proj = 0, so that branch adds 0."""
    torch.manual_seed(16)
    config = Idefics2PerceiverConfig(hidden_size=32, rms_norm_eps=1e-5, resampler_n_latents=4, resampler_depth=1,
                                     resampler_n_heads=4, resampler_head_dim=8, num_key_value_heads=4)
    config._attn_implementation = "eager"
    model = nn.Module()
    model.projection = Idefics2MLP(hidden_size=12, intermediate_size=85, output_size=32, hidden_act="silu")
    model.layer = Idefics2PerceiverLayer(config, layer_idx=0)
    model.latents = nn.Parameter(torch.randn(4, 32))
    with torch.no_grad():  # norm weights start at 1: make them random, so a missing norm weight shows
        model.layer.input_latents_norm.weight.uniform_(0.5, 1.5)
        model.layer.input_context_norm.weight.uniform_(0.5, 1.5)
        model.layer.mlp.down_proj.weight.zero_()
    x = torch.randn(3, 9, 12).requires_grad_()
    keep = torch.ones(3, 9, dtype=torch.bool)  # the context tokens, True = real, False = padding
    keep[0, 6:], keep[2, 8:] = False, False
    keys = torch.cat([keep, torch.zeros(3, 4, dtype=torch.bool)], dim=1)  # Idefics2's keys: context, then latents
    additive = torch.zeros(3, 1, 4, 13).masked_fill(~keys[:, None, None, :], torch.finfo(torch.float32).min)
    out = model.layer(model.latents.expand(3, -1, -1), model.projection(x), attention_mask=additive)
    return {"state_dict": model.state_dict(), "x": x.detach(), "keep": keep, "out": out.detach(),
            **gradients(out, {"x": x}, model)}


def torch_decoder_layer() -> dict:
    """torch's pre-norm nn.TransformerDecoderLayer without biases, GELU MLP: causal self-attention, then
    cross-attention with a mask. Evaluation mode: no dropout."""
    torch.manual_seed(12)
    layer = nn.TransformerDecoderLayer(32, 4, dim_feedforward=128, activation="gelu", batch_first=True,
                                       norm_first=True, bias=False).eval()
    with torch.no_grad():  # LayerNorm weights start at 1: make them random, so a missing norm weight shows
        for norm in [layer.norm1, layer.norm2, layer.norm3]:
            norm.weight.uniform_(0.5, 1.5)
    x, context = torch.randn(2, 5, 32).requires_grad_(), torch.randn(2, 7, 32).requires_grad_()
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    keep = torch.rand(5, 7) > 0.3  # the cross-attention mask, True = may attend
    keep[:, 0] = True  # every query keeps at least one context token
    out = layer(x, context, tgt_mask=~causal, memory_mask=~keep)  # torch's boolean masks: True = may NOT attend
    return {"state_dict": layer.state_dict(), "x": x.detach(), "context": context.detach(), "causal": causal,
            "keep": keep, "out": out.detach(), **gradients(out, {"x": x, "context": context}, layer)}


def torchcfm_path() -> dict:
    """torchcfm's ConditionalFlowMatcher with sigma 0: x_t = (1 - t) x_0 + t x_1, u_t = x_1 - x_0."""
    torch.manual_seed(8)
    x_0, x_1, t = torch.randn(4, 3, 2), torch.randn(4, 3, 2), torch.rand(4)
    _, x_t, u_t = ConditionalFlowMatcher(sigma=0.0).sample_location_and_conditional_flow(x_0, x_1, t)
    return {"x_0": x_0, "x_1": x_1, "t": t, "x_t": x_t, "dx_t": u_t}


def torchdiffeq_euler() -> dict:
    """torchdiffeq's odeint, Euler steps of 0.25 from t = 0 to 1, on v(x, t) = sin(x) - t x."""
    torch.manual_seed(9)
    x_init = torch.randn(4, 3, 2)
    x_1 = odeint(lambda t, x: torch.sin(x) - t * x, x_init, torch.tensor([0.0, 1.0]), method="euler",
                 options={"step_size": 0.25})[-1]
    return {"x_init": x_init, "num_steps": 4, "x_1": x_1}


def hf_warmup_cosine() -> dict:
    """Hugging Face's get_cosine_schedule_with_warmup: the lr factor at each step (base lr 1)."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=20)
    factors = []
    for _ in range(20):
        factors.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return {"warmup_steps": 5, "total_steps": 20, "factors": torch.tensor(factors)}


if __name__ == "__main__":
    references = {
        "llama_rope": llama_rope(),
        "llama_feed_forward": llama_feed_forward(),
        "llama_ffn_hidden_dims": llama_ffn_hidden_dims(),
        "llama_block": llama_block(),
        "llama_depth_init": llama_depth_init(),
        "vjepa2_depth_init": vjepa2_depth_init(),
        "hf_llama_mlp": hf_llama_mlp(bias=False),
        "hf_llama_mlp_bias": hf_llama_mlp(bias=True),
        "diffusers_timestep_embedding": diffusers_timestep_embedding(),
        "diffusers_adaln_zero_block": diffusers_adaln_zero_block(),
        "diffusers_adaln_zero_decoder": diffusers_adaln_zero_decoder(),
        "diffusers_adaln_rms_final_norm": diffusers_adaln_rms_final_norm(),
        "qwen3_attention": qwen3_attention(),
        "qwen3_attention_padded": qwen3_attention_padded(),
        "idefics2_perceiver": idefics2_perceiver(),
        "torch_decoder_layer": torch_decoder_layer(),
        "torchcfm_path": torchcfm_path(),
        "torchdiffeq_euler": torchdiffeq_euler(),
        "hf_warmup_cosine": hf_warmup_cosine(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(references, OUTPUT)
    print(f"saved {list(references)} to {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} kB)")
