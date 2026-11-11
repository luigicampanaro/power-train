"""Gradients of a reference implementation, saved next to its outputs by the capture scripts, so the tests can
send the same upstream gradient back through our layers and compare every gradient."""

import torch
from torch import nn


def gradients(out: torch.Tensor | tuple, inputs: dict, module: nn.Module | None = None) -> dict:
    """A fixed random upstream gradient sent back from out (a tensor, or a tuple of them): the gradients of the
    inputs (name -> tensor, requiring grad) and of every parameter of module, with that upstream gradient."""
    outs = out if isinstance(out, tuple) else (out,)
    grad_out = [torch.randn(o.shape, generator=torch.Generator().manual_seed(1000 + i), dtype=o.dtype)
                for i, o in enumerate(outs)]
    params = dict(module.named_parameters()) if module is not None else {}
    grads = torch.autograd.grad(outs, [*inputs.values(), *params.values()], grad_out)
    return {"grad_out": grad_out, "input_grads": dict(zip(inputs, grads[:len(inputs)])),
            "param_grads": dict(zip(params, grads[len(inputs):]))}
