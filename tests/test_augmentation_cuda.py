"""CUDA-only checks: determinism, compiled colour jitter, half precision, and no synchronisation.

These tests are in their own module, so the CPU test suite stays fast and the GPU properties
are checked together.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import diff_style.augmentations.ops as ops
from diff_style.augmentations import Augmentor
from diff_style.augmentations.ops import color_jitter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def test_same_generator_seed_gives_the_same_output():
    images = torch.rand(16, 2, 2, 3, 32, 32, device="cuda")
    first = Augmentor()(images, generator=generator())
    second = Augmentor()(images, generator=generator())
    assert torch.equal(first, second)


def test_compiled_color_jitter_matches_the_eager_result(monkeypatch):
    monkeypatch.setattr(ops, "_COMPILE_COLOR", True)
    ops._compiled_color_jitter.cache_clear()  # prove that this call compiles
    images = torch.rand(8, 2, 3, 16, 16, device="cuda")
    brightness = torch.full((8,), 1.2, device="cuda")
    contrast = torch.full((8,), 0.8, device="cuda")
    saturation = torch.full((8,), 1.1, device="cuda")
    hue = torch.linspace(-0.05, 0.05, 8, device="cuda")
    compiled = color_jitter(images, brightness, contrast, saturation, hue)
    assert ops._compiled_color_jitter.cache_info().currsize == 1, "the CUDA path was not compiled"
    assert ops._COMPILE_COLOR, "compilation failed and fell back to the eager function"
    monkeypatch.setattr(ops, "_COMPILE_COLOR", False)
    eager = color_jitter(images, brightness, contrast, saturation, hue)
    assert torch.allclose(compiled, eager, atol=1e-5)


def test_failed_compilation_warns_and_runs_eagerly(monkeypatch):
    def failing(*args):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(ops, "_COMPILE_COLOR", True)
    monkeypatch.setattr(ops, "_compiled_color_jitter", lambda: failing)
    images = torch.rand(2, 1, 3, 8, 8, device="cuda")
    with pytest.warns(UserWarning, match="compilation failed"):
        result = color_jitter(images, 1.2, 0.8, 1.1, 0.0)
    assert not ops._COMPILE_COLOR
    assert torch.allclose(result, ops._color_jitter(images, 1.2, 0.8, 1.1, 0.0, ops._luma(images.device, images.dtype)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_images_keep_the_contract(monkeypatch, dtype):
    monkeypatch.setattr(ops, "_COMPILE_COLOR", False)  # no compile time in this test
    images = torch.rand(8, 2, 2, 3, 32, 32, device="cuda", dtype=dtype)
    original = images.clone()
    output = Augmentor()(images, generator=generator())
    assert output.dtype == dtype
    assert torch.equal(images, original)
    assert torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
def test_the_default_pipeline_does_not_synchronise():
    """No host-device sync per batch: the GPU work overlaps the CPU work of the training loop."""
    augmentor = Augmentor()
    images = torch.rand(8, 2, 2, 3, 16, 16, device="cuda")
    augmentor(images, generator=generator())  # compile and build the motion bank first
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            augmentor(images, generator=generator())
    finally:
        torch.cuda.set_sync_debug_mode("default")
    synchronisations = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
    assert not synchronisations


def test_select_rounds_at_random_with_a_cuda_generator():
    """The rounding draw comes from the generator's seed and offset: different per call, same per seed."""
    def sizes(seed: int) -> list[int]:
        cuda, result = generator(seed), []
        for _ in range(400):
            indices = ops.select(22, 0.1, torch.device("cuda"), cuda)
            result.append(0 if indices is None else indices.numel())
        return result

    first = sizes(0)
    assert first == sizes(0)
    assert first != sizes(1)  # else every batch repeats the same rounding, and the rate is biased again
    assert set(first) == {2, 3}
    assert abs(first.count(3) / 400 - 0.2) <= 4.5 * (0.2 * 0.8 / 400) ** 0.5


def test_a_select_of_no_sample_does_not_repeat_its_rounding_in_the_next_select():
    """5 * 0.1 = 0.5 samples: none half the time, and then nothing is drawn on the device. The next
    select must still round afresh, else every later transform of the batch is skipped too."""
    second = 0
    for seed in range(1000):
        cuda = generator(seed)
        ops.select(5, 0.1, torch.device("cuda"), cuda)
        second += ops.select(5, 0.1, torch.device("cuda"), cuda) is not None
    assert abs(second / 1000 - 0.5) <= 4.5 * (0.25 / 1000) ** 0.5
