"""Checks for the augmentation operations, the Augmentor, the Hydra config, and the callback.
What the Augmentor draws is checked statistically in test_augmentation_sampling.py.

Every test is seeded, so a CI failure always reproduces locally.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from torchvision.transforms import v2

from diff_style.augmentations import DEFAULT_CONFIG, Augmentor
from diff_style.augmentations.ops import (
    LIMITS,
    coarse_dropout,
    color_jitter,
    crop_rotate,
    gamma_correction,
    gaussian_noise,
    jpeg_compression,
    motion_blur,
    motion_kernel,
    patch_brightness,
    planckian_jitter,
    planckian_rgb,
    sharpness,
)
from diff_style.callbacks import AugmentationCallback

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def image(height: int = 32, width: int = 32) -> torch.Tensor:
    return torch.rand(3, height, width)


def batch(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0).unsqueeze(0)


def boxes(top, left, height, width):
    def tensor(value):
        return torch.tensor(value, dtype=torch.int64).reshape(-1, 1)

    return tensor(top), tensor(left), tensor(height), tensor(width)


def off_config() -> dict:
    return {name: {"probability": 0.0} for name in DEFAULT_CONFIG}


def generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def augment_all(config: dict | None = None) -> Augmentor:
    """An Augmentor without clean samples, so every sample goes through the transforms."""
    return Augmentor(config, clean_probability=0.0)


# --- Operations: exact results for known inputs ---


def test_crop_returns_the_requested_region():
    x = image(64, 64)
    result = crop_rotate(batch(x), 5, 7, 50, 51, 0.0, (50, 51), "border")[0, 0]
    assert torch.allclose(result, x[:, 5:55, 7:58], atol=1e-5)


def test_zero_rotation_is_identity():
    x = image()
    result = crop_rotate(batch(x), 0, 0, 32, 32, 0.0, (32, 32))[0, 0]
    assert torch.allclose(result, x, atol=1e-6)


def test_rotation_keeps_a_square_square_on_a_wide_image():
    x = torch.zeros(3, 60, 120)
    x[:, 25:35, 55:65] = 1.0  # 10 x 10 square in the center
    result = crop_rotate(batch(x), 0, 0, 60, 120, 90.0, (60, 120), "zeros")[0, 0, 0]
    rows, columns = (result > 0.5).nonzero(as_tuple=True)
    assert rows.max() - rows.min() + 1 == 10
    assert columns.max() - columns.min() + 1 == 10


@pytest.mark.parametrize(
    "name, value, reference",
    [
        ("brightness", 1.3, v2.functional.adjust_brightness),
        ("contrast", 1.5, v2.functional.adjust_contrast),
        ("saturation", 1.5, v2.functional.adjust_saturation),
        ("hue", 0.1, v2.functional.adjust_hue),
    ],
)
def test_color_jitter_matches_torchvision(name, value, reference):
    x = image()
    result = color_jitter(batch(x), **{name: value})[0, 0]
    assert torch.allclose(result, reference(x, value), atol=1e-5)


def test_color_jitter_with_every_factor_matches_torchvision_in_order():
    """Every factor above 1 pushes pixels past [0, 1]: torchvision clamps after each step, so must color_jitter."""
    x = image()
    reference = v2.functional.adjust_hue(v2.functional.adjust_saturation(
        v2.functional.adjust_contrast(v2.functional.adjust_brightness(x, 1.3), 1.5), 1.4), 0.1)
    assert torch.allclose(color_jitter(batch(x), 1.3, 1.5, 1.4, 0.1)[0, 0], reference, atol=1e-5)


def test_color_jitter_defaults_are_identity():
    x = image()
    assert torch.allclose(color_jitter(batch(x))[0, 0], x, atol=1e-6)


def test_sharpness_one_is_identity():
    x = image()
    assert torch.allclose(sharpness(batch(x), 1.0)[0, 0], x, atol=1e-6)


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.5])
def test_sharpness_matches_torchvision_including_the_border(factor):
    x = image()
    result = sharpness(batch(x), factor)[0, 0]
    assert torch.allclose(result, v2.functional.adjust_sharpness(x, factor), atol=1e-6)


# Linear sRGB to CIE XYZ (IEC 61966-2-1), to read the chromaticity of planckian_rgb
SRGB_TO_XYZ = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


# Published Planckian locus chromaticities, CIE 1931 2-degree observer
@pytest.mark.parametrize("kelvin, x, y", [(2000, 0.5267, 0.4133), (3000, 0.4369, 0.4041), (4000, 0.3805, 0.3768),
                                          (6500, 0.3135, 0.3237), (10000, 0.2807, 0.2884)])
def test_black_body_color_is_on_the_cie_planckian_locus(kelvin, x, y):
    xyz = SRGB_TO_XYZ @ planckian_rgb(torch.tensor(float(kelvin)))
    assert abs(xyz[0] / xyz.sum() - x) < 1e-3
    assert abs(xyz[1] / xyz.sum() - y) < 1e-3


def test_planckian_lowest_allowed_temperature_gives_a_valid_image():
    result = planckian_jitter(batch(image()), LIMITS["planckian_jitter"]["temperature"][0])
    assert result.isfinite().all()


def test_planckian_6500k_is_identity():
    x = image()
    assert torch.allclose(planckian_jitter(batch(x), 6500)[0, 0], x, atol=1e-5)


def test_planckian_low_temperature_is_warm_and_high_is_cool():
    gray = torch.full((3, 4, 4), 0.5)
    warm = planckian_jitter(batch(gray), 3000)[0, 0, :, 0, 0]
    cool = planckian_jitter(batch(gray), 15000)[0, 0, :, 0, 0]
    red, green, blue = 0, 1, 2
    assert warm[red] > 0.5 > warm[blue]
    assert cool[blue] > 0.5 > cool[red]
    assert warm[green] == cool[green] == 0.5


def test_planckian_scales_linear_light_by_the_black_body_color():
    """Decoded to linear light (x ** 2.2), each channel is scaled by the 3000 K color over the 6500 K
    color, green fixed at 1."""
    gray = torch.full((3, 4, 4), 0.5)
    result = planckian_jitter(batch(gray), 3000)[0, 0, :, 0, 0]
    light = planckian_rgb(torch.tensor(3000.0)) / planckian_rgb(torch.tensor(6500.0))
    assert torch.allclose((result / 0.5) ** 2.2, light / light[1], atol=1e-4)


def test_gamma_two_squares_the_image():
    x = image()
    assert torch.allclose(gamma_correction(batch(x), 2.0)[0, 0], x**2, atol=1e-6)


def test_zero_noise_is_identity():
    x = image()
    assert torch.equal(gaussian_noise(batch(x), 0.0)[0, 0], x)


def test_noise_has_the_requested_standard_deviation():
    gray = torch.full((1, 1, 3, 64, 64), 0.5)
    noise = gaussian_noise(gray, 0.05, torch.Generator().manual_seed(0)) - 0.5
    assert abs(noise.mean()) < 0.002
    assert abs(noise.std() - 0.05) < 0.002


@pytest.mark.parametrize("angle, line", [(0.0, "row"), (90.0, "column")])
def test_motion_kernel_is_a_line_at_its_angle(angle, line):
    kernel = motion_kernel(5, angle, torch.device("cpu"), torch.float32)
    expected = torch.zeros(5, 5)
    if line == "row":
        expected[2, :] = 0.2  # 0 degrees: the middle row, 5 equal weights
    else:
        expected[:, 2] = 0.2  # 90 degrees: the middle column
    assert torch.allclose(kernel, expected, atol=1e-6)


def test_motion_blur_keeps_a_constant_image_constant():
    x = torch.full((3, 32, 32), 0.4)
    kernel = motion_kernel(7, 30.0, torch.device("cpu"), torch.float32)
    assert torch.allclose(motion_blur(batch(x), kernel[None])[0, 0], x, atol=1e-6)


def test_coarse_dropout_fills_only_the_box():
    x = image()
    result = coarse_dropout(batch(x), boxes(4, 6, 8, 10), torch.tensor([1]), 0.25)[0, 0]
    assert torch.all(result[:, 4:12, 6:16] == 0.25)
    outside = torch.ones_like(x, dtype=torch.bool)
    outside[:, 4:12, 6:16] = False
    assert torch.equal(result[outside], x[outside])


def test_coarse_dropout_fills_only_the_first_count_boxes():
    x = image()
    two_boxes = tuple(torch.tensor([pair]) for pair in ([2, 20], [2, 20], [4, 4], [4, 4]))  # top, left, height, width
    result = coarse_dropout(batch(x), two_boxes, torch.tensor([1]), 0.25)[0, 0]
    assert torch.all(result[:, 2:6, 2:6] == 0.25)
    assert torch.equal(result[:, 20:24, 20:24], x[:, 20:24, 20:24])  # the second box is past the count


def test_patch_brightness_with_a_hard_edge_scales_only_the_box():
    x = image()
    result = patch_brightness(batch(x), boxes(4, 6, 8, 10), torch.tensor([0.5]), 0.0)[0, 0]
    assert torch.allclose(result[:, 4:12, 6:16], x[:, 4:12, 6:16] * 0.5, atol=1e-6)
    outside = torch.ones_like(x, dtype=torch.bool)
    outside[:, 4:12, 6:16] = False
    assert torch.equal(result[outside], x[outside])


@pytest.mark.parametrize("side", [8, 40])
def test_soft_patch_reaches_the_full_factor_at_its_center_whatever_its_size(side):
    white = torch.ones(3, 64, 64)
    start = 32 - side // 2
    result = patch_brightness(batch(white), boxes(start, start, side, side), torch.tensor([0.5]), 0.5)[0, 0, 0]
    assert result[32, 32] == 0.5  # center: full factor
    assert 0.5 < result[32, start] < 1.0  # edge pixel: fading
    assert result[32, start - 1] == 1.0  # outside: unchanged


def test_lower_jpeg_quality_changes_more():
    x = image(64, 64)
    low = (jpeg_compression(batch(x), 10)[0, 0] - x).abs().mean()
    high = (jpeg_compression(batch(x), 95)[0, 0] - x).abs().mean()
    assert low > high


# --- Operations: a batch with one parameter per sample equals one sample at a time ---


def test_batched_crop_matches_one_sample_at_a_time():
    images = torch.rand(3, 2, 3, 32, 32)
    top, left, height, width = boxes([4, 6, 8], [5, 7, 9], [20, 18, 16], [21, 19, 17])
    angles = torch.tensor([5.0, -5.0, 10.0])
    batched = crop_rotate(images, top, left, height, width, angles, (32, 32))
    for i in range(3):
        single = crop_rotate(
            images[i : i + 1], int(top[i]), int(left[i]), int(height[i]), int(width[i]), float(angles[i]), (32, 32)
        )
        assert torch.allclose(batched[i], single[0], atol=1e-6)


def test_batched_color_matches_one_sample_at_a_time():
    images = torch.rand(3, 2, 3, 32, 32)
    brightness = torch.tensor([0.8, 1.0, 1.2])
    contrast = torch.tensor([1.2, 0.9, 1.0])
    saturation = torch.tensor([0.7, 1.0, 1.3])
    hue = torch.tensor([-0.05, 0.0, 0.05])
    batched = color_jitter(images, brightness, contrast, saturation, hue)
    for i in range(3):
        single = color_jitter(
            images[i : i + 1], float(brightness[i]), float(contrast[i]), float(saturation[i]), float(hue[i])
        )
        assert torch.allclose(batched[i], single[0], atol=1e-6)


@pytest.mark.parametrize(
    "operation, values",
    [
        (sharpness, [0.5, 1.0, 1.5]),
        (planckian_jitter, [3000, 9000, 15000]),
        (gamma_correction, [0.8, 1.0, 1.2]),
        (jpeg_compression, [30, 75, 90]),
    ],
)
def test_batched_scalar_operation_matches_one_sample_at_a_time(operation, values):
    images = torch.rand(3, 2, 3, 32, 32)
    batched = operation(images, torch.tensor(values))
    for i, value in enumerate(values):
        assert torch.allclose(batched[i], operation(images[i : i + 1], value)[0], atol=1e-6)


def test_batched_motion_blur_matches_one_sample_at_a_time():
    images = torch.rand(3, 2, 3, 32, 32)
    kernels = torch.stack([motion_kernel(7, angle, torch.device("cpu"), torch.float32) for angle in (0.0, 45.0, -30.0)])
    batched = motion_blur(images, kernels)
    for i in range(3):
        assert torch.allclose(batched[i], motion_blur(images[i : i + 1], kernels[i : i + 1])[0], atol=1e-6)


def test_batched_boxes_match_one_sample_at_a_time():
    images = torch.rand(2, 2, 3, 32, 32)
    box = boxes([4, 8], [6, 10], [8, 6], [10, 8])
    dropout = coarse_dropout(images, box, torch.tensor([1, 1]), 0.25)
    patch = patch_brightness(images, box, torch.tensor([0.5, 0.7]), 0.5)
    for i in range(2):
        one_box = tuple(item[i : i + 1] for item in box)
        assert torch.equal(dropout[i], coarse_dropout(images[i : i + 1], one_box, torch.tensor([1]), 0.25)[0])
        single_patch = patch_brightness(images[i : i + 1], one_box, torch.tensor([0.5, 0.7])[i : i + 1], 0.5)[0]
        assert torch.allclose(patch[i], single_patch, atol=1e-6)


# --- Augmentor: config checks ---


@pytest.mark.parametrize(
    "config, message",
    [
        ({"color_jiter": {"probability": 0.5}}, "unknown augmentation"),  # typo in the name
        ({"crop": {"ratios": (0.9, 1.0)}}, "unknown settings"),  # typo in a setting
        ({"rotation": {"probability": 1.5}}, "probability"),
        ({"color_jitter": {"brightness": (1.2, 0.8)}}, "low <= high"),  # reversed range
        ({"color_jitter": {"hue": (-0.6, 0.0)}}, "hue"),  # outside its limits
        ({"planckian_jitter": {"temperature": (1900, 6500)}}, "temperature"),  # NaN blue below ~1900 K
        ({"motion_blur": {"kernel_size": (4, 4)}}, "odd size"),
        ({"coarse_dropout": {"patches": (0, 0)}}, "patches"),
        ({"jpeg_compression": {"quality": (30.5, 75)}}, "integers"),
        ({"rotation": {"padding_mode": "wrap"}}, "padding_mode"),
    ],
)
def test_bad_config_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        Augmentor(config)


def test_clean_probability_outside_0_1_is_rejected():
    with pytest.raises(ValueError, match="clean_probability"):
        Augmentor(clean_probability=1.5)


# --- Augmentor: which samples are augmented ---


def test_clean_probability_one_leaves_every_sample_untouched():
    x = torch.rand(8, 2, 3, 16, 16)
    assert torch.equal(Augmentor(clean_probability=1.0)(x, generator=generator()), x)


def test_all_probabilities_zero_is_identity():
    x = torch.rand(2, 2, 3, 16, 16)
    assert torch.equal(augment_all(off_config())(x), x)


@pytest.mark.parametrize("name", list(DEFAULT_CONFIG))
def test_each_transform_on_its_own_changes_the_image(name):
    config = off_config()
    config[name] = {"probability": 1.0}
    if name == "crop":
        config[name]["ratio"] = (0.8, 0.9)  # a ratio near 1 can keep the whole small image
    images = torch.rand(8, 1, 3, 16, 16)
    output = augment_all(config)(images, generator=generator())
    assert not torch.equal(output, images)


# --- Augmentor: what reaches the pixels ---


def test_one_draw_is_shared_by_all_frames_and_cameras():
    config = off_config()
    for name in ("crop", "rotation", "color_jitter"):
        config[name]["probability"] = 1.0
    images = image().expand(1, 3, 2, 3, 32, 32).contiguous()  # 3 frames, 2 cameras, all the same
    output = augment_all(config)(images, generator=generator())
    assert not torch.equal(output, images)
    assert torch.equal(output, output[:, :1, :1].expand_as(output))


def test_noise_is_independent_per_frame():
    config = off_config()
    config["gaussian_noise"]["probability"] = 1.0
    images = image().expand(1, 2, 1, 3, 32, 32).contiguous()
    output = augment_all(config)(images, generator=generator())
    assert not torch.equal(output[0, 0], output[0, 1])


def test_same_generator_seed_gives_the_same_output():
    images = torch.rand(4, 2, 2, 3, 32, 32)
    assert torch.equal(Augmentor()(images, generator=generator()), Augmentor()(images, generator=generator()))


# --- Hydra config ---


def compose(*overrides: str, output_dir: Path):
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra.compose("train", overrides=[f"output_dir={output_dir}", *overrides])


def test_code_defaults_match_the_default_yaml(tmp_path):
    """configs/augmentation/default.yaml stays in step with DEFAULT_CONFIG in ops.py."""
    augmentor = hydra.utils.instantiate(compose("augmentation=default", output_dir=tmp_path).augmentation)
    assert augmentor.config == Augmentor().config
    assert augmentor.clean_probability == Augmentor().clean_probability


def test_hydra_overrides_reach_the_augmentor(tmp_path):
    cfg = compose(
        "augmentation.config.crop.ratio=[0.80, 0.90]",
        "augmentation.config.jpeg_compression.probability=0.10",
        "augmentation.clean_probability=0.5",
        output_dir=tmp_path,
    )
    augmentor = hydra.utils.instantiate(cfg.augmentation)
    assert augmentor.clean_probability == 0.5
    assert augmentor.config["crop"]["ratio"] == (0.80, 0.90)
    assert augmentor.config["jpeg_compression"]["probability"] == 0.10


def test_hydra_callback_takes_the_seed_from_train_yaml(tmp_path):
    callback = hydra.utils.instantiate(compose("seed=7", output_dir=tmp_path).callbacks.augmentation)
    assert isinstance(callback.augmentor, Augmentor)
    assert callback.seed == 7


# --- Callback ---


class Trainer:
    def __init__(self, global_rank: int = 0, current_epoch: int = 0) -> None:
        self.global_rank = global_rank
        self.current_epoch = current_epoch


def train_images(callback: AugmentationCallback, images: torch.Tensor, trainer: Trainer, batch_idx: int = 0):
    batch_dict = {"images": images.clone()}
    callback.on_train_batch_start(trainer, None, batch_dict, batch_idx)
    return batch_dict["images"]


def test_callback_augments_the_train_batch():
    images = torch.rand(2, 2, 2, 3, 16, 24)  # batch, frames, cameras, C, H, W: as the datamodule gives them
    result = train_images(AugmentationCallback(augment_all(), seed=0), images, Trainer())
    assert result.shape == images.shape
    assert not torch.equal(result, images)


def test_callback_repeats_exactly_for_the_same_seed_rank_epoch_and_batch():
    images = torch.rand(2, 2, 2, 3, 16, 24)
    first = train_images(AugmentationCallback(augment_all(), seed=0), images, Trainer(1, 3), batch_idx=5)
    second = train_images(AugmentationCallback(augment_all(), seed=0), images, Trainer(1, 3), batch_idx=5)
    assert torch.equal(first, second)


def test_callback_draws_differ_across_seeds_ranks_epochs_and_batches():
    images = torch.rand(2, 2, 2, 3, 16, 24)
    callback = AugmentationCallback(augment_all(), seed=0)
    reference = train_images(callback, images, Trainer(0, 0), batch_idx=0)
    assert not torch.equal(reference, train_images(AugmentationCallback(augment_all(), seed=1), images, Trainer()))
    assert not torch.equal(reference, train_images(callback, images, Trainer(global_rank=1)))
    assert not torch.equal(reference, train_images(callback, images, Trainer(current_epoch=1)))
    assert not torch.equal(reference, train_images(callback, images, Trainer(), batch_idx=1))


def test_batch_seeds_are_unique_across_ranks_epochs_and_batches():
    callback = AugmentationCallback(augment_all(), seed=0)
    seeds = {callback.batch_seed(rank, epoch, batch_idx) for rank in range(8) for epoch in range(10) for batch_idx in range(100)}
    assert len(seeds) == 8 * 10 * 100
