"""The reference implementations' saved outputs and gradients, and the checks against them.

outputs.pt: capture_outputs.py (torchtitan, Hugging Face transformers and diffusers, torchcfm, torchdiffeq).
dropout.pt: capture_dropout.py (timm, Hugging Face transformers and diffusers, torchscale, in training mode, with fixed masks).
The capture scripts run outside the project's environment (see each one); pytest does not collect them. Each layer
reference holds its upstream gradient `grad_out` and the gradients it gives, `input_grads` and `param_grads`.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
REFERENCES = torch.load(HERE / "outputs.pt", weights_only=True)
DROPOUT_REFERENCES = torch.load(HERE / "dropout.pt", weights_only=True)


def with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of a saved input that gradients flow to (the saved one stays untouched for the other tests)."""
    return tensor.clone().requires_grad_()


def renamed(state: dict, names: dict) -> dict:
    """state with the layer of each "<layer>.<parameter>" key renamed: names {"gate_proj": "w1"} gives w1.bias."""
    out = {}
    for key, value in state.items():
        layer, parameter = key.split(".", 1)
        out[f"{names[layer]}.{parameter}"] = value
    return out


def reference_gradients(out: torch.Tensor, inputs: dict, params: dict, grad_out: torch.Tensor) -> dict:
    """For a reference run in the test (e.g. torch's own modules): grad_out and the gradients it gives the inputs
    and params (name -> tensor), in the form of the saved references."""
    grads = torch.autograd.grad(out, [*inputs.values(), *params.values()], grad_out)
    return {"grad_out": [grad_out], "input_grads": dict(zip(inputs, grads[:len(inputs)])),
            "param_grads": dict(zip(params, grads[len(inputs):]))}


def assert_gradients_match(out: torch.Tensor | tuple, ref: dict, inputs: dict, module: nn.Module | None = None,
                           to_ours: Callable[[dict], dict] = dict) -> None:
    """Send the reference's upstream gradient back through our out (a tensor, or a tuple of them): the gradients
    of the inputs (name -> tensor) and of every trainable parameter of module must be the reference's, within float32
    precision. to_ours: the reference's parameter names -> ours, as for its weights."""
    outs = out if isinstance(out, tuple) else (out,)
    params = {name: p for name, p in module.named_parameters() if p.requires_grad} if module is not None else {}
    grads = torch.autograd.grad(outs, [*inputs.values(), *params.values()], ref["grad_out"])
    expected = to_ours(ref["param_grads"]) if module is not None else {}
    assert expected.keys() == params.keys()  # every parameter is checked
    expected |= {name: ref["input_grads"][name] for name in inputs}
    for name, grad in zip([*inputs, *params], grads):
        torch.testing.assert_close(grad, expected[name], msg=lambda message: f"gradient of {name}: {message}")
