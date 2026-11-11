"""TrainableAveragedModel: the EMA of the weights, of the trainable parameters only. A frozen parameter (the frozen
vision encoder) is never averaged, so it stays bit-identical. TrainableEMA in a real fit of the training config,
through validation, the end of training and the checkpoint: tests/test_ema_training.py."""

import pytest
import torch
from torch import nn

from diff_style.callbacks.ema import TrainableAveragedModel

DECAY = 0.9  # far from 1, so a few steps move the average


class Frozen(nn.Module):
    """A trainable layer, a frozen layer and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.trainable, self.frozen = nn.Linear(3, 3), nn.Linear(3, 3)
        self.frozen.requires_grad_(False)
        self.register_buffer("count", torch.zeros(()))


def test_trainable_parameters_are_the_exponential_moving_average():
    """The first update copies; then average = lerp(average, current, 1 - decay), i.e. decay * average +
    (1 - decay) * current, computed exactly as torch.lerp."""
    model = Frozen()
    average = TrainableAveragedModel(model, device=None, decay=DECAY)
    expected = None
    for step in range(4):
        with torch.no_grad():
            model.trainable.weight.add_(torch.randn(3, 3))
        current = model.trainable.weight.detach().clone()
        expected = current if expected is None else torch.lerp(expected, current, 1 - DECAY)
        average.update_parameters(model)
        assert torch.equal(average.module.trainable.weight, expected), step


def test_frozen_parameters_are_never_averaged_and_buffers_follow_the_model():
    """A sentinel in the average's frozen layer survives every update: the EMA does not track it. The buffers
    are copied, not averaged (a running count would be averaged into a fraction)."""
    model = Frozen()
    average = TrainableAveragedModel(model, device=None, decay=DECAY)
    with torch.no_grad():
        average.module.frozen.weight.fill_(7.0)
    for step in range(3):
        model.count.fill_(step + 1)
        average.update_parameters(model)
    assert torch.equal(average.module.frozen.weight, torch.full((3, 3), 7.0))
    assert average.module.count.item() == 3


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_the_average_keeps_its_dtype_when_the_model_changes_precision(dtype):
    """Lightning converts the model (64-true, bf16-true) after the callbacks' setup, which builds the average in
    float32. The average stays float32 (as Lightning's EMA): a bf16 model's EMA keeps its small steps."""
    model = Frozen()
    average = TrainableAveragedModel(model, device=None, decay=DECAY)
    model.to(dtype)
    for _ in range(2):
        average.update_parameters(model)
    assert average.module.trainable.weight.dtype == torch.float32
    assert torch.equal(average.module.trainable.weight, model.trainable.weight.float())


def test_a_parameter_unfrozen_later_is_averaged_from_then_on():
    """requires_grad is read at every update, not when the average is built."""
    model = Frozen()
    average = TrainableAveragedModel(model, device=None, decay=DECAY)
    average.update_parameters(model)
    model.frozen.requires_grad_(True)
    with torch.no_grad():
        model.frozen.weight.add_(1.0)
    average.update_parameters(model)
    assert not torch.equal(average.module.frozen.weight, model.frozen.weight)  # averaged, not copied
    assert torch.allclose(average.module.frozen.weight, model.frozen.weight - DECAY)
