"""Statistical checks of what the Augmentor draws: which samples each transform gets, whether its
results land back on those samples, and the parameters it draws for them.

Every transform function the Augmentor calls is replaced by a spy that records its arguments and
result, so a test sees exactly what the Augmentor asked for.

Every run is seeded, so a test always gives the same result. The limits are 4.5 standard deviations
for fractions, and p ~ 1e-4 for the uniformity tests. Correct code is far from them, while a wrong
range, probability, or sample fails clearly.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import diff_style.augmentations.augmentor as augmentor_module
from diff_style.augmentations import DEFAULT_CONFIG, Augmentor
from diff_style.augmentations.ops import MOTION_BINS, motion_bank, random_round, select, split

FRAMES, HEIGHT, WIDTH = 2, 16, 24  # 2 identical frames per sample; not square on purpose

# Config name: (function the Augmentor calls, settings). The ranges differ from the defaults, so a
# range read from the wrong place fails.
TRANSFORMS = {
    "crop": ("crop_rotate", {"ratio": (0.5, 0.8)}),
    "rotation": ("crop_rotate", {"degrees": 12.0, "padding_mode": "zeros"}),
    "color_jitter": ("color_jitter", {"brightness": (0.6, 1.1), "contrast": (0.7, 1.3),
                                      "saturation": (0.5, 1.5), "hue": (-0.1, 0.02)}),
    "sharpness": ("sharpness", {"factor": (0.2, 1.9)}),
    "planckian_jitter": ("planckian_jitter", {"temperature": (4000, 4020)}),
    "gamma_correction": ("gamma_correction", {"gamma": (0.5, 1.5)}),
    "gaussian_noise": ("gaussian_noise", {"sigma": (0.02, 0.07)}),
    "motion_blur": ("motion_blur", {"kernel_size": (2, 9), "angle": (-150.0, -30.0)}),
    "jpeg_compression": ("jpeg_compression", {"quality": (40, 60)}),
    "coarse_dropout": ("coarse_dropout", {"patches": (1, 3), "area": (0.05, 0.2), "fill": 0.3}),
    "patch_brightness": ("patch_brightness", {"factor": (0.3, 1.6), "size": (0.2, 0.6), "softness": 0.4}),
}


@pytest.fixture
def calls(monkeypatch):
    """{function name: (arguments, result)} of the last call of each transform function."""
    recorded = {}
    for name in {function for function, _ in TRANSFORMS.values()}:
        def spy(*args, _function=getattr(augmentor_module, name), _name=name):
            result = _function(*args)
            recorded[_name] = (args, result)
            return result

        monkeypatch.setattr(augmentor_module, name, spy)
    return recorded


def run(name: str, probability: float, clean_probability: float, count: int, **settings):
    """Augment `count` random samples with only transform `name` on. Returns (images, output)."""
    config = {**{other: {"probability": 0.0} for other in DEFAULT_CONFIG},  # every transform off
              name: {"probability": probability, **TRANSFORMS[name][1], **settings}}  # except `name`
    images = torch.rand(count, FRAMES, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0))
    images[:, 1] = images[:, 0]  # both frames equal: one shared draw keeps them equal
    original = images.clone()
    output = Augmentor(config, clean_probability)(images, generator=torch.Generator().manual_seed(1))
    assert torch.equal(images, original), "the caller's images were changed"
    return images, output


def setting(name: str, key: str) -> Any:
    """A setting of transform `name` in TRANSFORMS."""
    return TRANSFORMS[name][1][key]


def draws(calls, name: str, count: int = 4000, probability: float = 1.0, **settings) -> tuple:
    """The arguments the Augmentor passed to transform `name`, every sample augmented."""
    run(name, probability, 0.0, count, **settings)
    return calls[TRANSFORMS[name][0]][0]


def rows_of(subset: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Index in `images` of each sample of `subset`. Each sample is different random noise, so its
    first 4 pixel values identify it."""
    index = {tuple(sample[:4].tolist()): i for i, sample in enumerate(images.flatten(1))}
    return torch.tensor([index[tuple(sample[:4].tolist())] for sample in subset.flatten(1)])


# --- Statistics ---


def assert_fraction(count: int, total: int, probability: float) -> None:
    """`count` of `total` samples, each with `probability`: within 4.5 binomial standard deviations."""
    sd = math.sqrt(probability * (1 - probability) / total)
    assert abs(count / total - probability) <= 4.5 * sd + 1e-9, f"{count}/{total}, expected {probability}"


def assert_uniform(values: torch.Tensor, low: float, high: float) -> None:
    """Every value inside [low, high], spread uniformly: Kolmogorov-Smirnov distance < 2.2 / sqrt(n)."""
    values = values.double().flatten()
    assert low - 1e-6 <= values.min() and values.max() <= high + 1e-6, f"[{values.min()}, {values.max()}]"
    cdf = ((values - low) / (high - low)).sort().values  # uniform values: sorted, they follow i / n
    n = len(cdf)
    steps = torch.arange(1, n + 1, dtype=torch.float64) / n
    distance = torch.maximum(steps - cdf, cdf - (steps - 1 / n)).max().item()
    assert distance < 2.2 / math.sqrt(n), f"KS distance {distance:.4f}, limit {2.2 / math.sqrt(n):.4f}"


def assert_distribution(values: torch.Tensor, probabilities: dict[int, float]) -> None:
    """Integer values: only the expected ones, each as often as expected (chi-square < mean + 5 sd)."""
    counts = torch.bincount(values.long().flatten() - min(probabilities)).tolist()
    observed = {min(probabilities) + k: count for k, count in enumerate(counts) if count}
    assert set(observed) <= set(probabilities), f"unexpected values {sorted(set(observed) - set(probabilities))}"
    total = len(values.flatten())
    expected = {k: p * total for k, p in probabilities.items() if p * total >= 5}
    chi2 = sum((observed.get(k, 0) - e) ** 2 / e for k, e in expected.items())
    dof = max(len(expected) - 1, 1)
    assert chi2 < dof + 5 * math.sqrt(2 * dof), f"chi2 {chi2:.1f} for {dof} dof, observed {observed}"


def integers(low: int, high: int) -> dict[int, float]:
    """Each of low..high equally likely."""
    return {k: 1 / (high - low + 1) for k in range(low, high + 1)}


def rounded(scale: int, low: float, high: float) -> dict[int, float]:
    """Probability of each value of round(scale * r), r uniform in [low, high]."""
    probabilities = {k: (min((k + 0.5) / scale, high) - max((k - 0.5) / scale, low)) / (high - low)
                     for k in range(round(scale * low), round(scale * high) + 1)}
    return {k: p for k, p in probabilities.items() if p > 0}


def assert_positions_uniform(start: torch.Tensor, size: torch.Tensor, length: int) -> None:
    """Boxes of each size: every start that fits in `length` drawn equally often."""
    for value in size.unique().tolist():
        assert_distribution(start[size == value], integers(0, length - value))


def assert_independent(first: torch.Tensor, second: torch.Tensor) -> None:
    """Separate draws: correlation within 4.5 standard deviations of 0."""
    r = torch.corrcoef(torch.stack([first.double().flatten(), second.double().flatten()]))[0, 1].item()
    assert abs(r) < 4.5 / math.sqrt(len(first)), f"correlation {r:.3f}"


# --- Which samples each transform gets, and where its results go ---


def test_select_takes_a_whole_fraction_exactly():
    indices = select(100, 0.3, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert indices is not None
    assert indices.numel() == 30  # 100 * 0.3 is whole: no rounding
    assert len(set(indices.tolist())) == 30  # no sample twice
    assert indices.min() >= 0 and indices.max() < 100


def test_select_of_no_sample_or_every_sample():
    generator = torch.Generator().manual_seed(0)
    assert select(16, 0.0, torch.device("cpu"), generator) is None
    indices = select(16, 1.0, torch.device("cpu"), generator)
    assert indices is not None and torch.equal(indices, torch.arange(16))


def test_select_draws_different_samples_for_different_seeds():
    first = select(100, 0.5, torch.device("cpu"), torch.Generator().manual_seed(0))
    second = select(100, 0.5, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert first is not None and second is not None
    assert not torch.equal(first, second)


def test_random_round_is_exact_on_average():
    """22 * 0.1 = 2.2 samples: 2 or 3 per batch, 2.2 on average, so the rate does not depend on the batch size."""
    generator = torch.Generator().manual_seed(0)
    values = [random_round(2.2, generator) for _ in range(4000)]
    assert set(values) == {2, 3}
    assert_fraction(values.count(3), len(values), 0.2)
    assert random_round(3.0, generator) == 3  # whole values are not rounded


def test_select_rounds_a_fraction_at_random():
    sizes = []
    for seed in range(2000):
        indices = select(22, 0.1, torch.device("cpu"), torch.Generator().manual_seed(seed))
        sizes.append(0 if indices is None else indices.numel())
    assert set(sizes) == {2, 3}
    assert_fraction(sizes.count(3), len(sizes), 0.2)


def test_split_gives_every_index_exactly_once():
    selected, rest = split(100, 0.3, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert selected is not None and rest is not None
    assert selected.numel() == 30 and rest.numel() == 70
    assert sorted(selected.tolist() + rest.tolist()) == list(range(100))


@pytest.mark.parametrize("probability", [0.3, 1.0])
@pytest.mark.parametrize("clean_probability", [0.0, 0.3])
@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_each_transform_gets_its_fraction_of_samples_and_its_results_land_on_them(
    calls, name, clean_probability, probability
):
    function = TRANSFORMS[name][0]
    count = 300 if name == "jpeg_compression" else 1000  # jpeg is a slow CPU round trip
    images, output = run(name, probability, clean_probability, count)
    (subset, *_), result = calls[function]
    rows = rows_of(subset, images)
    # crop_rotate gets every augmented sample, and draws per sample whether to crop or rotate it
    applied = 1 - clean_probability if function == "crop_rotate" else (1 - clean_probability) * probability
    assert_fraction(len(rows), count, applied)
    assert len(set(rows.tolist())) == len(rows), "a sample went through twice"
    assert torch.equal(output[rows], result), "results not written back to their own samples"
    others = torch.ones(count, dtype=torch.bool)
    others[rows] = False
    assert torch.equal(output[others], images[others]), "a sample the transform did not get was changed"
    if name != "gaussian_noise":  # noise is the only draw that differs per frame
        assert torch.allclose(output[:, 0], output[:, 1], atol=1e-6), "the frames got different draws"


def test_with_every_default_transform_on_the_clean_fraction_stays_exactly_untouched():
    images = torch.rand(4000, FRAMES, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0))
    output = Augmentor(clean_probability=0.3)(images, generator=torch.Generator().manual_seed(1))
    untouched = (output == images).flatten(1).all(dim=1)
    assert untouched.sum() == 1200  # 4000 * 0.3 is whole: exactly; crop and color change every other sample
    assert output.isfinite().all() and output.min() >= 0 and output.max() <= 1


# --- The parameters each transform draws ---


def test_crop_draws_size_and_position_uniformly_and_keeps_the_aspect_ratio(calls):
    _, top, left, height, width, angle, size, _ = draws(calls, "crop", probability=0.5)
    cropped = height < HEIGHT  # a ratio <= 0.8 always crops
    assert_fraction(int(cropped.sum()), len(height), 0.5)
    assert (top[~cropped] == 0).all() and (left[~cropped] == 0).all() and (width[~cropped] == WIDTH).all()
    assert_distribution(height[cropped], rounded(HEIGHT, 0.5, 0.8))
    assert_distribution(width[cropped], rounded(WIDTH, 0.5, 0.8))
    ratio_gap = (height[cropped] / HEIGHT - width[cropped] / WIDTH).abs()
    assert (ratio_gap <= 0.5 / HEIGHT + 0.5 / WIDTH).all(), "same ratio for both sides, up to rounding"
    assert_positions_uniform(top[cropped], height[cropped], HEIGHT)
    assert_positions_uniform(left[cropped], width[cropped], WIDTH)
    assert (angle == 0).all() and tuple(size) == (HEIGHT, WIDTH)


def test_rotation_draws_angles_uniformly_in_both_directions(calls):
    _, top, left, height, width, angle, _, padding_mode = draws(calls, "rotation", probability=0.5)
    rotated = angle != 0
    assert_fraction(int(rotated.sum()), len(angle), 0.5)
    assert_uniform(angle[rotated], -12.0, 12.0)
    assert padding_mode == "zeros"
    assert (top == 0).all() and (left == 0).all() and (height == HEIGHT).all() and (width == WIDTH).all()


def test_color_jitter_draws_each_factor_uniformly_and_separately(calls):
    _, brightness, contrast, saturation, hue = draws(calls, "color_jitter")
    for values, key in ((brightness, "brightness"), (contrast, "contrast"), (saturation, "saturation"), (hue, "hue")):
        assert_uniform(values, *setting("color_jitter", key))
    assert_independent(brightness, contrast)
    assert_independent(saturation, hue)


@pytest.mark.parametrize("name, key", [("sharpness", "factor"), ("gamma_correction", "gamma"),
                                       ("gaussian_noise", "sigma")])
def test_one_value_transforms_draw_uniformly_in_their_range(calls, name, key):
    _, value, *_ = draws(calls, name)
    assert_uniform(value, *setting(name, key))


@pytest.mark.parametrize("name, key", [("planckian_jitter", "temperature"), ("jpeg_compression", "quality")])
def test_integer_transforms_draw_every_value_of_their_range_equally(calls, name, key):
    _, value = draws(calls, name, count=4000 if name == "planckian_jitter" else 600)
    assert_distribution(value, integers(*setting(name, key)))


def test_motion_blur_draws_odd_sizes_and_angles_uniformly(calls):
    _, kernels = draws(calls, "motion_blur")
    sizes = (3, 5, 7, 9)  # the odd sizes in kernel_size = (2, 9)
    bank = motion_bank(sizes, torch.device("cpu"), torch.float32).flatten(2)  # (size, angle bin, 81)
    distance = (kernels.flatten(1)[:, None, None] - bank[None]).abs().amax(dim=-1)
    found = distance.flatten(1).argmin(dim=1)  # the first kernel of the bank equal to each drawn one
    assert (distance.flatten(1).amin(dim=1) < 1e-6).all(), "a drawn kernel is not in the bank"
    assert_distribution(found // MOTION_BINS, integers(0, len(sizes) - 1))
    # Bins are 5 degrees from -180: angles in [-150, -30) are bins 6..29. A line at a and a + 180 is the
    # same kernel, so the search finds the first: within [-180, 0) that is the drawn bin itself.
    assert_distribution(found % MOTION_BINS, integers(6, 29))


def test_coarse_dropout_draws_patch_count_box_size_and_position_uniformly(calls):
    _, (top, left, height, width), counts, fill = draws(calls, "coarse_dropout")
    assert_distribution(counts, integers(1, 3))
    used = torch.arange(3).reshape(1, 3) < counts.reshape(-1, 1)  # boxes past the count are not drawn
    low, high = math.sqrt(0.05), math.sqrt(0.2)  # the side of a square of that area
    assert_distribution(height[used], rounded(HEIGHT, low, high))
    assert_distribution(width[used], rounded(WIDTH, low, high))
    assert_positions_uniform(top[used], height[used], HEIGHT)
    assert_positions_uniform(left[used], width[used], WIDTH)
    assert fill == 0.3


def test_patch_brightness_draws_factor_box_size_and_position_uniformly(calls):
    _, (top, left, height, width), factor, softness = draws(calls, "patch_brightness")
    assert_uniform(factor, 0.3, 1.6)
    assert_distribution(height, rounded(HEIGHT, 0.2, 0.6))
    assert_distribution(width, rounded(WIDTH, 0.2, 0.6))
    assert_independent(height, width)
    assert_positions_uniform(top, height, HEIGHT)
    assert_positions_uniform(left, width, WIDTH)
    assert softness == 0.4
