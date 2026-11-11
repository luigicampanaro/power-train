from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from diff_style.models.layers.init import INIT_STD, init_linear


class MLP(nn.Module):
    """Two-layer MLP, fc2(activation(fc1 x)): the MLP of ViT and DiT. Same parameter names as timm's Mlp.

    activation: a factory, e.g. nn.GELU or functools.partial(nn.GELU, approximate="tanh") (DiT).
    dropout: on the hidden activations, training only (timm's drop1, nn.TransformerEncoderLayer's dropout).
    The output's dropout is the block's, on every branch.
    out_dim: the output width, default dim (timm's out_features).
    """

    def __init__(self, dim: int, hidden_dim: int, activation: Callable[[], nn.Module] = nn.GELU, bias: bool = True,
                 dropout: float = 0.0, out_dim: int | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim if out_dim is None else out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))

    def init_weights(self, output_std: float) -> None:
        """Llama 3's init, as SwiGLUMLP's: fc1 (into the branch) at 0.02, fc2 (back into the stream) at output_std."""
        init_linear(self.fc1, INIT_STD)
        init_linear(self.fc2, output_std)


class SwiGLUMLP(nn.Module):
    """Gated MLP of Llama (Shazeer 2020, "GLU Variants Improve Transformer"): w2(silu(w1 x) * w3 x),
    w1 the gate, w3 the value. Same parameter names as torchtitan's Llama 3 FeedForward.

    bias: Llama has none.
    dropout: on the gated hidden activations, training only (as timm's SwiGLU drop1; Llama has none).
    The output's dropout is the block's, on every branch.
    out_dim: the output width, default dim (timm's out_features).
    multiple_of: hidden_dim is rounded up to a multiple of it, as Llama's FeedForward does, so the GEMMs fit the
    tensor-core tiles (512 * 8/3 = 1365 -> 1536 with 256). 1 = hidden_dim as given.
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, dropout: float = 0.0,
                 out_dim: int | None = None, multiple_of: int = 1) -> None:
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # gate
        self.w2 = nn.Linear(hidden_dim, dim if out_dim is None else out_dim, bias=bias)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # value
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

    def init_weights(self, output_std: float) -> None:
        """torchtitan's Llama 3 FeedForward.init_weights: w1 at 0.02, then w2 and w3 at output_std."""
        init_linear(self.w1, INIT_STD)
        for linear in (self.w2, self.w3):
            init_linear(linear, output_std)
