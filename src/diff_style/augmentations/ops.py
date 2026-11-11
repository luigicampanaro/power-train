"""GPU image augmentation for training, tests, and the gallery notebook.

All public operations take ``(B, N, 3, H, W)`` images. ``N`` packs the previous frames
and cameras of each sample. ``Augmentor`` draws one parameter set per sample and calls
the operations. The notebook calls the same operations with explicit parameters.

The operations stay on the image device. The forward pass has no Python loop over the
batch and never waits for the GPU, except ``jpeg_compression`` (a CPU round trip, off by default).
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from diff_style.augmentations.regions import coarse_dropout, patch_brightness

_LUMA = torch.tensor([0.2989, 0.587, 0.114])  # same weights as torchvision rgb_to_grayscale
# CIE XYZ -> linear sRGB
_XYZ_TO_RGB = torch.tensor(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)
PLANCKIAN_RANGE = (1667, 25000)  # kelvin range of the Planckian locus approximation below

# Probabilities apply to the augmented samples only (see Augmentor.clean_probability).
# Sources of the values are in configs/augmentation/default.yaml.
DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "crop": {"probability": 1.0, "ratio": (0.90, 1.00)},
    "rotation": {"probability": 0.30, "degrees": 5.0, "padding_mode": "reflection"},
    "color_jitter": {
        "probability": 1.0,
        "brightness": (0.80, 1.20),
        "contrast": (0.80, 1.20),
        "saturation": (0.80, 1.20),
        "hue": (-0.05, 0.05),
    },
    "sharpness": {"probability": 0.30, "factor": (0.50, 1.50)},
    "planckian_jitter": {"probability": 0.10, "temperature": (3000, 15000)},
    "gaussian_noise": {"probability": 0.10, "sigma": (0.01, 0.03)},
    "motion_blur": {"probability": 0.05, "kernel_size": (3, 7), "angle": (-180.0, 180.0)},
    "jpeg_compression": {"probability": 0.0, "quality": (30, 75)},
    "coarse_dropout": {"probability": 0.10, "patches": (1, 2), "area": (0.02, 0.05), "fill": 0.0},
    "patch_brightness": {"probability": 0.10, "factor": (0.50, 0.80), "size": (0.10, 0.25), "softness": 0.5},
    "gamma_correction": {"probability": 0.10, "gamma": (0.80, 1.20)},
}

# Allowed values of each setting, (lowest, highest). For a (low, high) range both ends must be
# inside, and low <= high. Every section also has a probability in [0, 1].
LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "crop": {"ratio": (0.0, 1.0)},
    "rotation": {"degrees": (0.0, 180.0)},
    "color_jitter": {"brightness": (0.0, math.inf), "contrast": (0.0, math.inf),
                     "saturation": (0.0, math.inf), "hue": (-0.5, 0.5)},
    "sharpness": {"factor": (0.0, math.inf)},
    "planckian_jitter": {"temperature": (2000, PLANCKIAN_RANGE[1])},  # below ~1900 K blue is negative: NaN
    "gaussian_noise": {"sigma": (0.0, 1.0)},
    "motion_blur": {"kernel_size": (1, 31), "angle": (-180.0, 180.0)},
    "jpeg_compression": {"quality": (1, 100)},
    "coarse_dropout": {"patches": (1, 100), "area": (0.0, 1.0), "fill": (0.0, 1.0)},
    "patch_brightness": {"factor": (0.0, math.inf), "size": (0.0, 1.0), "softness": (0.0, 1.0)},
    "gamma_correction": {"gamma": (0.0, math.inf)},
}
INTEGER_SETTINGS = {("planckian_jitter", "temperature"), ("motion_blur", "kernel_size"),
                    ("jpeg_compression", "quality"), ("coarse_dropout", "patches")}
PADDING_MODES = ("zeros", "border", "reflection")


def check_config(config: Mapping[str, Any]) -> None:
    """Raise ValueError for an unknown section or key, or a value outside ``LIMITS``."""
    for name, section in config.items():
        if name not in DEFAULT_CONFIG:
            raise ValueError(f"unknown augmentation {name!r}, expected one of {list(DEFAULT_CONFIG)}")
        unknown = set(section) - set(DEFAULT_CONFIG[name])
        if unknown:
            raise ValueError(f"{name}: unknown settings {sorted(unknown)}, expected {sorted(DEFAULT_CONFIG[name])}")
        if not 0.0 <= section["probability"] <= 1.0:
            raise ValueError(f"{name}.probability must be in [0, 1], got {section['probability']}")
    for name, settings in LIMITS.items():
        for key, (lowest, highest) in settings.items():
            value = config[name][key]
            low, high = value if isinstance(value, tuple) else (value, value)
            if not lowest <= low <= high <= highest:
                raise ValueError(f"{name}.{key} must be inside [{lowest}, {highest}] with low <= high, got {value}")
            if (name, key) in INTEGER_SETTINGS and not (isinstance(low, int) and isinstance(high, int)):
                raise ValueError(f"{name}.{key} must be integers, got {value}")
    if config["rotation"]["padding_mode"] not in PADDING_MODES:
        raise ValueError(f"rotation.padding_mode must be one of {PADDING_MODES}, got {config['rotation']['padding_mode']!r}")
    low, high = config["motion_blur"]["kernel_size"]
    if low | 1 > high:  # the kernels are the odd sizes in [low, high]
        raise ValueError(f"motion_blur.kernel_size must include an odd size, got {(low, high)}")


MOTION_BINS = 72


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):  # list, tuple, or Hydra ListConfig
        return tuple(plain(item) for item in value)
    return value


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: plain(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge(result[key], value)
        else:
            result[key] = plain(value)
    return result


def check_image(image: torch.Tensor) -> None:
    if image.ndim < 4 or image.shape[-3] != 3:
        raise ValueError(f"expected (B, ..., 3, H, W), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError(f"expected a floating point image, got {image.dtype}")


def _param(value: float | torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((batch,), float(value), device=device, dtype=dtype)


def _int_param(value: int | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.int64).reshape(-1)
    return torch.full((batch,), int(value), device=device, dtype=torch.int64)


def uniform(shape: tuple[int, ...], low: float, high: float, device: torch.device,
             generator: torch.Generator | None) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(low, high, generator=generator)


def select(count: int, probability: float, device: torch.device,
            generator: torch.Generator | None) -> torch.Tensor | None:
    """Indices of the samples a transform gets, or ``None`` for none.

    ``count * probability`` samples, rounded down or up at random on the host (see
    ``random_round``): the expected number is exact for any batch size, and the caller knows the
    subset size without synchronising with the device. The samples are a uniform random subset
    without repetition.
    """
    return split(count, probability, device, generator)[0]


def split(count: int, probability: float, device: torch.device,
          generator: torch.Generator | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(selected, rest)``: the indices ``select`` returns and the other indices.

    Both parts come from one permutation, so both sizes are known on the host, without a
    device synchronisation. ``selected`` is ``None`` when no sample is selected, and ``rest``
    is ``None`` when there is no other sample (or none is selected: nothing to put back).
    """
    if probability <= 0.0:
        return None, None
    keep = count if probability >= 1.0 else random_round(count * probability, generator)
    if keep <= 0:
        return None, None
    if keep >= count:
        return torch.arange(count, device=device), None
    permutation = torch.randperm(count, device=device, generator=generator)
    return permutation[:keep], permutation[keep:]


def random_round(value: float, generator: torch.Generator | None) -> int:
    """``value`` rounded up with probability ``value - floor(value)``, else down.

    The expected result is exactly ``value``, so a rate like 0.1 of 22 samples stays 0.1 on
    average instead of always 2 / 22. The draw is on the host, so it never waits for the GPU:
    a CUDA generator seeds a CPU one from its seed, its offset (which advances with every draw)
    and ``value``, then moves its offset on: the next call rounds afresh even when no device
    draw happened in between (a ``select`` of no sample draws nothing).
    """
    whole = math.floor(value)
    if value == whole:
        return whole
    if generator is not None and generator.device.type != "cpu":
        offset = generator.get_offset()
        seed = hash((generator.initial_seed(), offset, value)) & 0xFFFF_FFFF_FFFF_FFFF
        generator.set_offset(offset + 4)  # Philox offsets are multiples of 4; host state only, no sync
        generator = torch.Generator().manual_seed(seed)
    return whole + int(torch.rand((), generator=generator).item() < value - whole)


def replace(images: torch.Tensor, indices: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    if indices.numel() == images.shape[0]:
        return result
    images.index_copy_(0, indices, result)
    return images


def crop_rotate(
    images: torch.Tensor,
    top: int | torch.Tensor,
    left: int | torch.Tensor,
    height: int | torch.Tensor,
    width: int | torch.Tensor,
    angle: float | torch.Tensor,
    output_size: tuple[int, int],
    padding_mode: str = "reflection",
) -> torch.Tensor:
    """Crop and rotate in one sampling pass. Boxes are inside the input image.

    One box and one angle per sample: the frames and cameras of a sample share them, so they
    are channels of one grid sample. The grid is per sample, not per frame or camera.
    """
    batch, input_height, input_width = images.shape[0], images.shape[-2], images.shape[-1]
    channels = images.shape[-3] * math.prod(images.shape[1:-3])  # 3 x cameras x frames
    device, dtype = images.device, images.dtype
    top = _int_param(top, batch, device)
    left = _int_param(left, batch, device)
    crop_height = _int_param(height, batch, device)
    crop_width = _int_param(width, batch, device)
    radians = torch.deg2rad(_param(angle, batch, device, dtype))
    cos, sin = torch.cos(radians).reshape(batch, 1), torch.sin(radians).reshape(batch, 1)

    # Map each output pixel to the input, in grid_sample coordinates ([-1, 1] across each side).
    # The rotation is done in pixels, so a non-square image is rotated, not sheared.
    crop_w = crop_width.to(dtype).reshape(batch, 1)
    crop_h = crop_height.to(dtype).reshape(batch, 1)
    shift_x = (2.0 * left.to(dtype).reshape(batch, 1) + crop_w) / input_width - 1.0
    shift_y = (2.0 * top.to(dtype).reshape(batch, 1) + crop_h) / input_height - 1.0
    a00 = (cos * crop_w / input_width).reshape(batch, 1, 1)
    a01 = (-sin * crop_h / input_width).reshape(batch, 1, 1)
    a02 = shift_x.reshape(batch, 1, 1)
    a10 = (sin * crop_w / input_height).reshape(batch, 1, 1)
    a11 = (cos * crop_h / input_height).reshape(batch, 1, 1)
    a12 = shift_y.reshape(batch, 1, 1)

    output_height, output_width = output_size
    base_x = (2.0 * (torch.arange(output_width, device=device, dtype=dtype) + 0.5) / output_width - 1.0)
    base_y = (2.0 * (torch.arange(output_height, device=device, dtype=dtype) + 0.5) / output_height - 1.0)
    grid = torch.empty(batch, output_height, output_width, 2, device=device, dtype=dtype)
    grid[..., 0] = a00 * base_x.reshape(1, 1, output_width) + a01 * base_y.reshape(1, output_height, 1) + a02
    grid[..., 1] = a10 * base_x.reshape(1, 1, output_width) + a11 * base_y.reshape(1, output_height, 1) + a12
    flat = images.reshape(batch, channels, input_height, input_width)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return out.reshape(*images.shape[:-2], output_height, output_width)


def _adjust_hue(image: torch.Tensor, hue: float | torch.Tensor) -> torch.Tensor:
    """Shift the HSV hue like torchvision ``adjust_hue``, written per pixel so it compiles into one kernel."""
    if not isinstance(hue, torch.Tensor) and hue == 0.0:
        return image
    original_dtype = image.dtype
    image = image.to(torch.float32)
    red, green, blue = image.split(1, dim=-3)
    value = torch.maximum(torch.maximum(red, green), blue)
    delta = value - torch.minimum(torch.minimum(red, green), blue)
    safe = torch.where(delta == 0, 1.0, delta)
    h = torch.where(value == red, (green - blue) / safe,
                    torch.where(value == green, (blue - red) / safe + 2.0, (red - green) / safe + 4.0))
    h = torch.where(delta == 0, 0.0, h)  # gray pixels have no hue
    saturation = torch.where(value == 0, 0.0, delta / value.clamp_min(1e-12))
    h6 = (h / 6.0 + hue).remainder(1.0) * 6.0
    # HSV -> RGB for the three channels at once: offsets 5, 3, 1 give red, green, blue
    k = (torch.arange(5.0, 0.0, -2.0, device=image.device).reshape(3, 1, 1) + h6).remainder(6.0)
    rgb = value - value * saturation * torch.clamp(torch.minimum(k, 4.0 - k), 0.0, 1.0)
    return rgb.clamp(0.0, 1.0).to(original_dtype)


@functools.cache
def _luma(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The luma weights on ``device``; cached, so no blocking host-to-device copy per call."""
    return _LUMA.to(device=device, dtype=dtype).reshape(3, 1, 1)


_COMPILE_COLOR = True  # set to False when compilation fails: color_jitter then runs eagerly


@functools.cache
def _compiled_color_jitter():
    return torch.compile(_color_jitter, dynamic=True)


def color_jitter(images: torch.Tensor, brightness: float | torch.Tensor = 1.0,
                 contrast: float | torch.Tensor = 1.0, saturation: float | torch.Tensor = 1.0,
                 hue: float | torch.Tensor = 0.0) -> torch.Tensor:
    """Brightness, contrast, saturation, and hue, exactly like torchvision applied in this order.

    Each step is clamped to [0, 1], like torchvision. On CUDA, the whole function is compiled
    into a few fused kernels.
    """
    global _COMPILE_COLOR
    luma = _luma(images.device, images.dtype)
    if _COMPILE_COLOR and images.is_cuda:
        try:
            return _compiled_color_jitter()(images, brightness, contrast, saturation, hue, luma)
        except Exception as error:
            _COMPILE_COLOR = False
            warnings.warn(f"color_jitter compilation failed, running eagerly (slower): {error!r}", stacklevel=2)
    return _color_jitter(images, brightness, contrast, saturation, hue, luma)


def _color_jitter(images: torch.Tensor, brightness: float | torch.Tensor, contrast: float | torch.Tensor,
                  saturation: float | torch.Tensor, hue: float | torch.Tensor,
                  luma: torch.Tensor) -> torch.Tensor:
    batch, device, dtype = images.shape[0], images.device, images.dtype
    b = _param(brightness, batch, device, dtype).reshape(-1, 1, 1, 1, 1)
    c = _param(contrast, batch, device, dtype).reshape(-1, 1, 1, 1, 1)
    s = _param(saturation, batch, device, dtype).reshape(-1, 1, 1, 1, 1)
    out = (images * b).clamp(0.0, 1.0)
    mean = (out * luma).sum(dim=-3, keepdim=True).mean(dim=(-2, -1), keepdim=True)  # contrast: gray mean
    out = torch.lerp(mean, out, c).clamp(0.0, 1.0)
    out = torch.lerp((out * luma).sum(dim=-3, keepdim=True), out, s).clamp(0.0, 1.0)  # saturation: gray
    hue_value = hue.reshape(-1, 1, 1, 1, 1) if isinstance(hue, torch.Tensor) else hue
    return _adjust_hue(out, hue_value)


def sharpness(images: torch.Tensor, factor: float | torch.Tensor) -> torch.Tensor:
    """Like torchvision ``adjust_sharpness``: blend with a smoothed image, border pixels unchanged."""
    batch, frames, channels, height, width = images.shape
    flat = images.reshape(batch * frames, channels, height, width)
    kernel = _blur_kernel(images.device, images.dtype).expand(channels, 1, 3, 3)
    blurred = flat.clone()  # the border keeps the original pixels, so sharpness leaves it unchanged
    blurred[..., 1:-1, 1:-1] = F.conv2d(flat, kernel, groups=channels)
    scale = _param(factor, batch, images.device, images.dtype).repeat_interleave(frames).reshape(-1, 1, 1, 1)
    # lerp(flat, blurred, 1 - scale) = flat + (scale - 1) * (flat - blurred), in one fused kernel
    return torch.lerp(flat, blurred, 1.0 - scale).reshape_as(images).clamp(0.0, 1.0)


@functools.cache
def _blur_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The 3 x 3 smoothing kernel of torchvision ``adjust_sharpness`` (PIL SMOOTH): 1s around a 5, over 13."""
    kernel = torch.tensor([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]]) / 13.0
    return kernel.reshape(1, 1, 3, 3).to(device=device, dtype=dtype)


def planckian_rgb(temperature: torch.Tensor) -> torch.Tensor:
    """Linear sRGB color of a black body at ``temperature`` kelvin, shape ``(..., 3)``.

    Chromaticity from the Planckian locus approximation of Kim et al. (2002), valid in
    ``PLANCKIAN_RANGE``, then CIE XYZ (Y = 1) to linear sRGB.
    """
    t = temperature.to(torch.float32)
    x = torch.where(
        t <= 4000,
        -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910,
        -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390,
    )
    y = torch.where(
        t <= 2222,
        -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683,
        torch.where(
            t <= 4000,
            -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867,
            3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483,
        ),
    )
    xyz = torch.stack((x / y, torch.ones_like(x), (1.0 - x - y) / y), dim=-1)
    return xyz @ _xyz_to_rgb(t.device).T


@functools.cache
def _xyz_to_rgb(device: torch.device) -> torch.Tensor:
    """The CIE XYZ to linear sRGB matrix on ``device``; cached, so no blocking copy per call."""
    return _XYZ_TO_RGB.to(device)


@functools.cache
def _white_6500(device: torch.device) -> torch.Tensor:
    return planckian_rgb(torch.tensor(6500.0, device=device))


def planckian_jitter(images: torch.Tensor, temperature: int | torch.Tensor) -> torch.Tensor:
    """Change the light color to a black body at ``temperature`` kelvin. 6500 K is identity.

    Per-channel gains are the light color over the 6500 K color, with green fixed at 1.
    The gains act on linear light, so on sRGB-encoded images they are raised to 1 / 2.2.
    Below 6500 K the image gets warmer (more red), above it gets cooler (more blue).
    """
    batch = images.shape[0]
    light = planckian_rgb(_param(temperature, batch, images.device, torch.float32))
    gain = light / _white_6500(images.device)
    gain = (gain / gain[:, 1:2]) ** (1.0 / 2.2)
    return (images * gain.reshape(batch, 1, 3, 1, 1).to(images)).clamp(0.0, 1.0)


def gamma_correction(images: torch.Tensor, gamma: float | torch.Tensor) -> torch.Tensor:
    batch = images.shape[0]
    return images.clamp(0.0, 1.0).pow(_param(gamma, batch, images.device, images.dtype).reshape(-1, 1, 1, 1, 1))


def gaussian_noise(images: torch.Tensor, sigma: float | torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    batch = images.shape[0]
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    return (images + _param(sigma, batch, images.device, images.dtype).reshape(-1, 1, 1, 1, 1) * noise).clamp(0.0, 1.0)


def motion_kernel(size: int, angle: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.zeros(size, size, device=device, dtype=dtype)
    center = (size - 1) / 2
    radians = math.radians(angle)
    for index in range(size):
        step = index - center
        x = center + step * math.cos(radians)
        y = center + step * math.sin(radians)
        x0, y0 = math.floor(x), math.floor(y)
        dx, dy = x - x0, y - y0
        for xx, yy, weight in (
            (x0, y0, (1 - dx) * (1 - dy)), (x0 + 1, y0, dx * (1 - dy)),
            (x0, y0 + 1, (1 - dx) * dy), (x0 + 1, y0 + 1, dx * dy),
        ):
            if 0 <= xx < size and 0 <= yy < size:
                kernel[yy, xx] += weight
    return kernel / kernel.sum()


@functools.cache
def motion_bank(sizes: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Kernels of every size and ``MOTION_BINS`` angles, ``(sizes, bins, largest, largest)``; cached."""
    largest = sizes[-1]
    bank = torch.zeros(len(sizes), MOTION_BINS, largest, largest)
    for size_index, size in enumerate(sizes):
        offset = (largest - size) // 2
        for bin_index in range(MOTION_BINS):
            angle = -180.0 + (bin_index + 0.5) * 360.0 / MOTION_BINS
            bank[size_index, bin_index, offset:offset + size, offset:offset + size] = motion_kernel(
                size, angle, torch.device("cpu"), torch.float32
            )
    return bank.to(device=device, dtype=dtype)


def motion_blur(images: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    batch, frames, channels, height, width = images.shape
    size = kernels.shape[-1]
    groups = batch * frames * channels
    flat = images.reshape(1, groups, height, width)
    weight = kernels[:, None, None].expand(batch, frames, channels, size, size)
    weight = weight.reshape(groups, 1, size, size)
    out = F.conv2d(F.pad(flat, (size // 2,) * 4, mode="replicate"), weight, groups=groups)
    return out.reshape_as(images).clamp(0.0, 1.0)


def jpeg_compression(images: torch.Tensor, quality: int | torch.Tensor) -> torch.Tensor:
    """JPEG round trip per sample on the CPU, like torchvision ``JPEG``: slow, and waits for the GPU."""
    batch, frames, _, height, width = images.shape
    qualities = _int_param(quality, batch, images.device).tolist()
    flat = images.reshape(batch * frames, 3, height, width)
    decoded = []
    for index, value in enumerate(qualities):
        chunk = flat[index * frames:(index + 1) * frames]
        uint8 = (chunk.clamp(0.0, 1.0) * 255).round().to(torch.uint8)
        decoded.append(v2.JPEG(quality=value)(uint8.cpu()).to(images.device, images.dtype) / 255.0)
    return torch.cat(decoded).reshape_as(images)


__all__ = [
    "DEFAULT_CONFIG",
    "LIMITS",
    "check_config",
    "MOTION_BINS",
    "check_image",
    "coarse_dropout",
    "color_jitter",
    "crop_rotate",
    "gamma_correction",
    "gaussian_noise",
    "jpeg_compression",
    "merge",
    "motion_blur",
    "motion_bank",
    "motion_kernel",
    "patch_brightness",
    "plain",
    "planckian_rgb",
    "PLANCKIAN_RANGE",
    "planckian_jitter",
    "random_round",
    "replace",
    "select",
    "sharpness",
    "split",
    "uniform",
]

