from collections.abc import Callable

import torch
from torch import nn

from diff_style.models.layers import INIT_STD, Attention, SwiGLUMLP
from diff_style.models.transformer import RMSNorm


class Perceiver(nn.Module):
    """Any number of input tokens -> `num_latents` tokens: HPT's stem (Wang et al. 2024) with the Transformer's
    parts. A per-token MLP adapts the inputs (e.g. frozen DINOv3 patch tokens), then learned latent queries read them
    with one cross-attention, pre-norm as in TransformerBlock:
        latents + cross_attention(norm(latents), norm(mlp(feature_norm(x))))
    feature_norm(input_dim) normalizes the inputs whatever the stage before gives (RMSNorm by default);
    nn.Identity feeds them as they are, e.g. features already normalized by the encoder.
    The output length is num_latents, whatever the input length. The latents carry on into the output (HPT returns the
    read alone): they tell the obs encoder which slot a token is, as the camera and time embeddings are the same for
    every token of an image. There is no MLP on the latents: the transformer after the Perceiver does the computation,
    on the tokens of every camera at once. The inputs get no position here: they carry their own (DINOv3's).
    attention(dim, num_heads), norm(dim), mlp(dim, hidden_dim, out_dim=...): factories, as in TransformerBlock;
    the input MLP's hidden width is mlp_ratio * dim.
    """

    def __init__(self, input_dim: int, num_latents: int, dim: int, num_heads: int, mlp_ratio: float = 8 / 3,
                 attention: Callable[..., nn.Module] = Attention, norm: Callable[..., nn.Module] = RMSNorm,
                 mlp: Callable[..., nn.Module] = SwiGLUMLP,
                 feature_norm: Callable[..., nn.Module] = RMSNorm) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.feature_norm = feature_norm(input_dim)
        self.input_mlp = mlp(input_dim, int(mlp_ratio * dim), out_dim=dim)
        self.input_norm = norm(dim)
        self.latents = nn.Parameter(torch.empty(num_latents, dim))
        self.latent_norm = norm(dim)
        self.attention = attention(dim, num_heads)
        self.init_weights()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, L, input_dim) -> (N, num_latents, dim). mask: (N, L) bool, False = padding the latents ignore."""
        inputs = self.input_norm(self.input_mlp(self.feature_norm(x)))
        latents = self.latents.expand(len(x), -1, -1)  # the same queries for every sample
        context_mask = None if mask is None else mask[:, None, None, :]  # (N, 1, 1, L): every head and latent
        return latents + self.attention(self.latent_norm(latents), context=inputs, mask=context_mask)

    def init_weights(self) -> None:
        """The Transformer's init (Llama 3's): every weight matrix and the latents from trunc_normal(0.02). The input
        MLP projects into the latents' space, as a ViT's patch embedding, and the attention is the latents' only
        residual branch, so Llama's depth init gives its wo 0.02 / sqrt(1) too."""
        self.input_mlp.init_weights(INIT_STD)
        nn.init.trunc_normal_(self.latents, std=INIT_STD)
        for norm in (self.feature_norm, self.input_norm, self.latent_norm):
            if hasattr(norm, "reset_parameters"):  # nn.Identity has nothing to reset
                norm.reset_parameters()
        self.attention.init_weights(INIT_STD)
