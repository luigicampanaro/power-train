"""Llama 3's init (torchtitan): every projection into a branch from trunc_normal(0.02), the projections back into
the residual stream from a smaller std that the stack gives each layer (depth init). Biases start at 0."""

from torch import nn

INIT_STD = 0.02  # Llama 3, GPT-2, ViT, DINOv2/v3, V-JEPA 2: the std of the projections into a branch


def depth_init_std(layer_id: int) -> float:
    """Llama 3's depth init (torchtitan's default): the std of layer layer_id's projections back into the residual
    stream, 0.02 / sqrt(2 (layer_id + 1)), so that the sum of the layers' outputs does not grow with depth at the
    start. The 2: a Llama block's residual branches (attention, MLP)."""
    return INIT_STD / (2 * (layer_id + 1)) ** 0.5


def init_linear(linear: nn.Linear, std: float) -> None:
    """Weights from trunc_normal(0, std), cut at +-2 as torchtitan's (in effect a normal), and biases 0."""
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
