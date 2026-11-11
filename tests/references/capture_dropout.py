"""Save seeded inputs, weights and outputs of the reference dropout implementations that the policy is
tested against (tests/test_dropout.py, test_layers.py) to dropout.pt, next to this file.

Dropout is random, so these outputs are reproducible only with the same random draws: the same seed
AND the same torch version (torch 2.8 draws other Bernoulli masks than 2.5.1). So, unlike
capture_outputs.py, this runs with the torch version of the project, and the fixture saves it.
Every reference has a permissive license: timm, Hugging Face transformers and diffusers (Apache-2.0),
torchscale (MIT). torchscale has no release tag, and its package pins an old timm: clone its last commit outside,
then run from the diff_style folder:

    git clone https://github.com/microsoft/torchscale /tmp/references/torchscale_4d1e0e8
    git -C /tmp/references/torchscale_4d1e0e8 checkout 4d1e0e82e5adf86dd424f1463192635b73fc8efc
    PYTHONPATH=/tmp/references/torchscale_4d1e0e8 uv run --no-project --python 3.12 \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        --with torch==2.5.1 --with timm==1.0.30 --with transformers==4.57.1 --with diffusers==0.35.2 \
        --with fairscale==0.4.13 --with einops \
        python tests/references/capture_dropout.py

For every reference, also the gradients of its inputs and parameters for a fixed upstream gradient
(capture_gradients.py), back through the same dropout masks.
Every reference runs in training mode on CPU, with the saved `seed` set right before its forward pass.
"""

from pathlib import Path

import torch
from diffusers.models.embeddings import LabelEmbedding
from timm.layers import Mlp, SwiGLU, drop_path
from timm.models.vision_transformer import Block, VisionTransformer
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import DecoderLayer
from transformers import GPT2Config, GPT2Model

from capture_gradients import gradients

OUTPUT = Path(__file__).resolve().parent / "dropout.pt"


def timm_mlp() -> dict:
    """timm's Mlp with another output width (out_features) and dropout on the hidden activations only (drop2,
    on the output, is the block's)."""
    torch.manual_seed(0)
    mlp = Mlp(in_features=32, hidden_features=128, out_features=24, drop=(0.3, 0.0)).train()
    x = torch.randn(4, 5, 32).requires_grad_()
    torch.manual_seed(100)
    out = mlp(x)
    return {"seed": 100, "state_dict": mlp.state_dict(), "x": x.detach(), "p": 0.3, "out": out.detach(),
            **gradients(out, {"x": x}, mlp)}


def timm_swiglu() -> dict:
    """timm's SwiGLU, act(fc1_g x) * fc1_x x -> drop1 -> fc2, without biases, as Llama's, with another output
    width (out_features)."""
    torch.manual_seed(1)
    swiglu = SwiGLU(in_features=32, hidden_features=85, out_features=24, bias=False, drop=(0.3, 0.0)).train()
    x = torch.randn(4, 5, 32).requires_grad_()
    torch.manual_seed(101)
    out = swiglu(x)
    return {"seed": 101, "state_dict": swiglu.state_dict(), "x": x.detach(), "p": 0.3, "out": out.detach(),
            **gradients(out, {"x": x}, swiglu)}


def timm_drop_path() -> dict:
    torch.manual_seed(2)
    x = torch.randn(16, 3, 4).requires_grad_()
    torch.manual_seed(102)
    out = drop_path(x, 0.4, training=True)
    return {"seed": 102, "x": x.detach(), "p": 0.4, "out": out.detach(), **gradients(out, {"x": x})}


def timm_vit_block() -> dict:
    """timm's ViT block in training, with every dropout on: attention weights (attn_drop), each branch's
    output and the MLP's hidden activations (proj_drop), stochastic depth (drop_path)."""
    torch.manual_seed(3)
    block = Block(dim=32, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.2, attn_drop=0.1,
                  drop_path=0.3).train()
    with torch.no_grad():  # LayerNorm weights start at 1: make them random, so a missing norm weight shows
        block.norm1.weight.uniform_(0.5, 1.5)
        block.norm2.weight.uniform_(0.5, 1.5)
    x = torch.randn(8, 6, 32).requires_grad_()
    torch.manual_seed(103)
    out = block(x)
    return {"seed": 103, "state_dict": block.state_dict(), "x": x.detach(), "attn_drop": 0.1, "proj_drop": 0.2,
            "drop_path": 0.3, "out": out.detach(), **gradients(out, {"x": x}, block)}


def torchscale_decoder_layer() -> dict:
    """torchscale's DecoderLayer (the decoder of DeepNet and Magneto; OFA's has the same branches) in training:
    pre-norm self-attention -> cross-attention -> FFN, each branch's output with dropout, then drop path, and the
    FFN's hidden activations with dropout. Drop path 0.5 on every branch, the cross-attention included: some samples
    skip it, the others keep it. Attention dropout off (timm_vit_block checks it). Also the output in eval mode."""
    torch.manual_seed(7)
    args = DecoderConfig(decoder_embed_dim=32, decoder_attention_heads=4, decoder_ffn_embed_dim=128, decoder_layers=2,
                         dropout=0.2, activation_dropout=0.3, drop_path_rate=0.5, subln=False)
    layer = DecoderLayer(args, depth=1, is_encoder_decoder=True).train()  # the last of 2 layers: drop path 0.5
    with torch.no_grad():  # LayerNorm weights start at 1 and biases at 0: make them random, so a missing one shows
        for norm in [layer.self_attn_layer_norm, layer.encoder_attn_layer_norm, layer.final_layer_norm]:
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    kept = []  # for each drop path call (self-attention, cross-attention, FFN): the samples that keep the branch
    layer.drop_path.register_forward_hook(lambda module, args, out: kept.append(out.flatten(1).any(1)))
    x = torch.randn(8, 5, 32).requires_grad_()
    context = torch.randn(8, 7, 32).requires_grad_()
    torch.manual_seed(107)
    out = layer(x, encoder_out=context)[0]
    assert 0 < kept[1].sum() < 8, "some samples must skip the cross-attention, and the others keep it"
    reference = gradients(out, {"x": x, "context": context}, layer)
    with torch.no_grad():
        eval_out = layer.eval()(x, encoder_out=context)[0]
    return {"seed": 107, "state_dict": layer.state_dict(), "x": x.detach(), "context": context.detach(),
            "dropout": 0.2, "activation_dropout": 0.3, "drop_path": 0.5, "out": out.detach(), "eval_out": eval_out,
            **reference}


def timm_vit_drop_path_rates() -> dict:
    """The drop path rate of each block of timm's ViT: linear from 0 to drop_path_rate."""
    vit = VisionTransformer(img_size=16, patch_size=8, embed_dim=32, depth=5, num_heads=4, drop_path_rate=0.3)
    return {"depth": 5, "drop_path_rate": 0.3,
            "rates": torch.tensor([float(getattr(block.drop_path1, "drop_prob", 0.0)) for block in vit.blocks])}


def gpt2_embedding_dropout() -> dict:
    """GPT-2's embd_pdrop: dropout(token embedding + position embedding), before the first block."""
    torch.manual_seed(4)
    config = GPT2Config(vocab_size=6, n_positions=4, n_embd=32, n_layer=0, n_head=4, embd_pdrop=0.3)
    model = GPT2Model(config).train()
    input_ids = torch.randint(0, 6, (5, 4))
    captured = {}
    model.drop.register_forward_hook(lambda module, args, out: captured.update(out=out))
    torch.manual_seed(104)
    model(input_ids)
    return {"seed": 104, "wte": model.wte.weight.detach(), "wpe": model.wpe.weight.detach(), "input_ids": input_ids,
            "p": 0.3, "out": captured["out"].detach(),
            **gradients(captured["out"], {"wte": model.wte.weight, "wpe": model.wpe.weight})}


def diffusers_label_dropout() -> dict:
    """diffusers' LabelEmbedding: a dropped sample's label becomes the extra "null" class, as random draws
    in training and as force_drop_ids (the unconditional pass of classifier-free guidance)."""
    torch.manual_seed(5)
    embedder = LabelEmbedding(num_classes=5, hidden_size=8, dropout_prob=0.4)
    labels = torch.randint(0, 5, (32,))
    force_drop_ids = torch.randint(0, 2, (32,))
    torch.manual_seed(105)
    out = embedder.train()(labels)
    return {"seed": 105, "embedding_table": embedder.embedding_table.weight.detach(), "labels": labels, "p": 0.4,
            "out": out.detach(), **gradients(out, {"embedding_table": embedder.embedding_table.weight}),
            "force_drop_ids": force_drop_ids,
            "forced_out": embedder.eval()(labels, force_drop_ids=force_drop_ids).detach()}


if __name__ == "__main__":
    references = {
        "torch_version": torch.__version__.split("+")[0],
        "timm_mlp": timm_mlp(),
        "timm_swiglu": timm_swiglu(),
        "timm_drop_path": timm_drop_path(),
        "timm_vit_block": timm_vit_block(),
        "torchscale_decoder_layer": torchscale_decoder_layer(),
        "timm_vit_drop_path_rates": timm_vit_drop_path_rates(),
        "gpt2_embedding_dropout": gpt2_embedding_dropout(),
        "diffusers_label_dropout": diffusers_label_dropout(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(references, OUTPUT)
    print(f"saved {list(references)} to {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} kB)")
