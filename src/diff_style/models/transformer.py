"""Pre-norm transformer: the block of the observation encoder and of the action decoder.

Each block is up to three residual branches, each after a norm:
    self-attention -> cross-attention to `context` (optional) -> MLP
`dropout` drops elements of each branch's output before it is added back, in training only (as
nn.TransformerDecoderLayer's dropout1/2/3; the attention weights have their own, in the attention factory,
and so do the MLP's hidden activations). `drop_path` then drops each branch whole, per sample (stochastic depth).
With `cond_dim` set, a conditioning vector (e.g. the flow time) modulates the self-attention and MLP
branches with adaLN-Zero (DiT, Peebles & Xie 2023): it gives each branch a shift and a scale after
the norm, and a gate on the output. The gates start at zero, so these two branches start at 0. The
cross-attention branch is not modulated (PixArt-α) and has no gate: it reads the context from the first step.

The attention, norm and MLP are factories (functools.partial, or Hydra configs with `_partial_: true`),
called once per sub-layer. The defaults are Llama's (RMSNorm, SwiGLU MLP, attention without biases), and
with them a plain block is exactly torchtitan's Llama 3 block with RoPE off. With DiT's parts
(LayerNorm, GELU MLP, attention with biases) a conditioned block is exactly DiT's block, and with
torch's parts a cross-attention block is exactly nn.TransformerDecoderLayer (tests/test_transformer.py).
Positions come from the inputs (e.g. the time embedding of FlowMatchingNetwork): the blocks add none.

Init: Llama 3's (torchtitan, depth init). Each branch's projections in start at trunc_normal(0.02), its projection
back into the stream at 0.02 / sqrt(2 (l + 1)) in block l, biases at 0. Transformer runs it when it is built.
The attention and MLP factories must thus make modules with init_weights(output_std), as Attention, MLP and SwiGLUMLP.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from diff_style.models.layers import AdaLNModulation, Attention, DropPath, Modulation, SwiGLUMLP, depth_init_std

RMSNorm = partial(nn.RMSNorm, eps=1e-5)  # Llama 3's eps


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 8 / 3, cross_attention: bool = False,
                 cond_dim: int | None = None, causal: bool = False,
                 attention: Callable[..., nn.Module] = Attention,
                 norm: Callable[..., nn.Module] = RMSNorm,
                 mlp: Callable[..., nn.Module] = SwiGLUMLP,
                 dropout: float = 0.0, drop_path: float = 0.0) -> None:
        """mlp_ratio: MLP hidden width / dim. The default 8/3 gives SwiGLU the parameters of a 4x GELU MLP (Llama).
        causal: each token's self-attention sees itself and the tokens before it only (the cross-attention
        sees the whole context). No mask tensor, so the flash kernels still run.
        attention(dim, num_heads, is_causal=...), norm(dim, elementwise_affine=...), mlp(dim, hidden_dim): factories.
        drop_path: probability of skipping each branch for a sample, training only (timm's ViT block).
        """
        super().__init__()
        affine = cond_dim is None  # with adaLN, the condition gives the shift and scale instead
        self.attention_norm = norm(dim, elementwise_affine=affine)
        self.attention = attention(dim, num_heads, is_causal=causal)
        if cross_attention:
            self.cross_attention_norm = norm(dim)
            self.cross_attention = attention(dim, num_heads)
        else:
            self.cross_attention = None
        self.ffn_norm = norm(dim, elementwise_affine=affine)
        self.feed_forward = mlp(dim, int(mlp_ratio * dim))
        self.dropout = nn.Dropout(dropout)  # one module: it has no state, and eval() turns it off
        self.drop_path = DropPath(drop_path)  # the same, and each call draws its own samples
        # Shift, scale and gate for the self-attention and for the MLP branch
        self.adaLN_modulation = AdaLNModulation(cond_dim, dim, num_chunks=6) if cond_dim is not None else None

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, dim). context: (B, Lc, dim), attended by the cross-attention. cond: (B, cond_dim), for adaLN.
        mask / context_mask: masks of the self- / cross-attention (see layers/attention.py), None = attend to all.
        """
        attention_mod, ffn_mod = self.modulations(cond)
        h = attention_mod.modulate(self.attention_norm(x))
        x = x + self.drop_path(attention_mod.gated(self.dropout(self.attention(h, mask=mask))))
        if self.cross_attention is not None:
            h = self.cross_attention_norm(x)
            x = x + self.drop_path(self.dropout(self.cross_attention(h, context=context, mask=context_mask)))
        h = ffn_mod.modulate(self.ffn_norm(x))
        x = x + self.drop_path(ffn_mod.gated(self.dropout(self.feed_forward(h))))
        return x

    def modulations(self, cond: torch.Tensor | None) -> tuple[Modulation, Modulation]:
        """The adaLN settings of the self-attention and of the MLP branch; without adaLN, settings that change nothing."""
        if self.adaLN_modulation is None:
            return Modulation(), Modulation()
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(cond)
        return Modulation(shift_attn, scale_attn, gate_attn), Modulation(shift_ffn, scale_ffn, gate_ffn)

    def init_weights(self, output_std: float) -> None:
        """torchtitan's Llama 3 TransformerBlock.init_weights: the norms reset, then each branch's own init, with
        output_std for its projection back into the stream. The adaLN Linear keeps its zero init."""
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(output_std)
        if self.cross_attention is not None:
            self.cross_attention_norm.reset_parameters()
            self.cross_attention.init_weights(output_std)
        self.feed_forward.init_weights(output_std)


class Transformer(nn.Module):
    """`depth` TransformerBlocks and a final norm. With cond_dim, the final norm is modulated too
    (shift and scale, as DiT's final layer).
    drop_path: the last block's; it grows linearly from 0 at the first block (timm's ViT, DINOv2)."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 8 / 3, cross_attention: bool = False,
                 cond_dim: int | None = None, causal: bool = False,
                 attention: Callable[..., nn.Module] = Attention,
                 norm: Callable[..., nn.Module] = RMSNorm,
                 mlp: Callable[..., nn.Module] = SwiGLUMLP,
                 dropout: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            TransformerBlock(dim, num_heads, mlp_ratio, cross_attention, cond_dim, causal, attention, norm, mlp, dropout,
                             rate)
            for rate in torch.linspace(0, drop_path, depth).tolist()
        )
        self.norm = norm(dim, elementwise_affine=cond_dim is None)
        self.norm_modulation = AdaLNModulation(cond_dim, dim, num_chunks=2) if cond_dim is not None else None
        self.init_weights()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Arguments as TransformerBlock.forward; the same context, cond and masks go to every block."""
        for layer in self.layers:
            x = layer(x, context=context, cond=cond, mask=mask, context_mask=context_mask)
        return self.final_modulation(cond).modulate(self.norm(x))

    def final_modulation(self, cond: torch.Tensor | None) -> Modulation:
        """The final norm's shift and scale (no gate, as DiT's final layer); without adaLN, settings that change nothing."""
        if self.norm_modulation is None:
            return Modulation()
        shift, scale = self.norm_modulation(cond)
        return Modulation(shift, scale)

    def init_weights(self) -> None:
        """Llama 3's depth init (torchtitan's default): block l's projections back into the stream start at
        0.02 / sqrt(2 (l + 1)), so that the sum of the blocks' outputs does not grow with depth at the start."""
        for layer_id, layer in enumerate(self.layers):
            layer.init_weights(depth_init_std(layer_id))
        self.norm.reset_parameters()
