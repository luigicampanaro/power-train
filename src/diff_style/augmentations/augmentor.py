"""Sample one augmentation parameter per sample and apply the operations in ``ops``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from diff_style.augmentations.ops import (
    DEFAULT_CONFIG,
    MOTION_BINS,
    check_config,
    check_image,
    coarse_dropout,
    color_jitter,
    crop_rotate,
    gamma_correction,
    gaussian_noise,
    jpeg_compression,
    merge,
    motion_blur,
    motion_bank,
    patch_brightness,
    planckian_jitter,
    replace,
    select,
    sharpness,
    split,
    uniform,
)


class Augmentor(nn.Module):
    """Apply the configured augmentations with one parameter draw per sample.

    Input and output shape is ``(B, ..., 3, H, W)``. All previous frames and cameras of
    one sample use the same draw.

    A ``clean_probability`` fraction of the samples is left untouched, so the network keeps
    seeing real images like at validation and on the robot. The other samples get each
    transform with its own ``probability``. Each fraction of a batch is rounded down or up at
    random to whole samples, so every rate is exact on average for any batch size.
    Set a ``probability`` to 0 to deactivate a transform.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, clean_probability: float = 0.3) -> None:
        super().__init__()
        self.config = merge(DEFAULT_CONFIG, config or {})
        check_config(self.config)
        if not 0.0 <= clean_probability <= 1.0:
            raise ValueError(f"clean_probability must be in [0, 1], got {clean_probability}")
        self.clean_probability = float(clean_probability)

    def forward(self, images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        check_image(images)
        if generator is not None and (
            generator.device.type != images.device.type
            or (images.is_cuda and generator.device.index not in (None, images.device.index))
        ):
            raise ValueError(f"generator device {generator.device} does not match image device {images.device}")
        if not any(section["probability"] > 0 for section in self.config.values()):
            return images

        shape = images.shape
        flat = images.reshape(shape[0], math.prod(shape[1:-3]), *shape[-3:])  # (B, frames * cameras, 3, H, W)
        # Pick the samples to augment once; only those go through the transforms below
        indices, rest = split(shape[0], 1.0 - self.clean_probability, images.device, generator)
        if indices is None:
            return images
        everyone = rest is None
        out = flat if everyone else flat.index_select(0, indices)
        if everyone and self.config["crop"]["probability"] == 0 and self.config["rotation"]["probability"] == 0:
            out = out.clone()  # the transforms below write into `out`: never into the caller's images
        out = self._crop_rotate(out, generator)
        out = self._color_jitter(out, generator)
        out = self._sharpness(out, generator)
        out = self._planckian_jitter(out, generator)
        out = self._gamma_correction(out, generator)
        out = self._gaussian_noise(out, generator)
        out = self._motion_blur(out, generator)
        out = self._jpeg_compression(out, generator)
        out = self._coarse_dropout(out, generator)
        out = self._patch_brightness(out, generator)
        if not everyone:
            out = _scatter_back(flat, indices, rest, out)  # put the augmented samples next to the clean ones
        return out.reshape(shape)

    def _crop_rotate(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        crop, rotation = self.config["crop"], self.config["rotation"]
        if crop["probability"] == 0 and rotation["probability"] == 0:
            return images
        batch, _, _, height, width = images.shape
        device = images.device
        # Each sample is cropped and rotated with its own probability; otherwise ratio 1 and angle 0
        ratios = uniform((batch,), *crop["ratio"], device, generator)
        ratios = torch.where(torch.rand(batch, device=device, generator=generator) < crop["probability"], ratios, 1.0)
        angles = uniform((batch,), -rotation["degrees"], rotation["degrees"], device, generator)
        angles = torch.where(torch.rand(batch, device=device, generator=generator) < rotation["probability"], angles, 0.0)
        crop_height = (height * ratios).round().clamp(1, height).long()
        crop_width = (width * ratios).round().clamp(1, width).long()
        top = (torch.rand(batch, device=device, generator=generator) * (height - crop_height + 1)).floor().long()
        left = (torch.rand(batch, device=device, generator=generator) * (width - crop_width + 1)).floor().long()
        padding = rotation["padding_mode"] if rotation["probability"] > 0 else "border"
        return crop_rotate(images, top, left, crop_height, crop_width, angles, (height, width), padding)

    def _color_jitter(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["color_jitter"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        count, device = subset.shape[0], images.device
        result = color_jitter(
            subset,
            uniform((count,), *cfg["brightness"], device, generator),
            uniform((count,), *cfg["contrast"], device, generator),
            uniform((count,), *cfg["saturation"], device, generator),
            uniform((count,), *cfg["hue"], device, generator),
        )
        return replace(images, indices, result)

    def _sharpness(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["sharpness"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        factor = uniform((subset.shape[0],), *cfg["factor"], images.device, generator)
        return replace(images, indices, sharpness(subset, factor))

    def _planckian_jitter(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["planckian_jitter"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        temperature = torch.randint(cfg["temperature"][0], cfg["temperature"][1] + 1, (subset.shape[0],),
                                    device=images.device, generator=generator)
        return replace(images, indices, planckian_jitter(subset, temperature))

    def _gamma_correction(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["gamma_correction"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        gamma = uniform((subset.shape[0],), *cfg["gamma"], images.device, generator)
        return replace(images, indices, gamma_correction(subset, gamma))

    def _gaussian_noise(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["gaussian_noise"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        sigma = uniform((subset.shape[0],), *cfg["sigma"], images.device, generator)
        return replace(images, indices, gaussian_noise(subset, sigma, generator))

    def _motion_blur(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["motion_blur"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        sizes = tuple(range(cfg["kernel_size"][0] | 1, cfg["kernel_size"][1] + 1, 2))
        bank = motion_bank(sizes, images.device, images.dtype)
        size_index = torch.randint(0, len(sizes), (subset.shape[0],), device=images.device, generator=generator)
        angles = uniform((subset.shape[0],), *cfg["angle"], images.device, generator)
        angle_index = ((angles + 180.0) * MOTION_BINS / 360.0).floor().long().remainder(MOTION_BINS)
        return replace(images, indices, motion_blur(subset, bank[size_index, angle_index]))

    def _jpeg_compression(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["jpeg_compression"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        quality = torch.randint(cfg["quality"][0], cfg["quality"][1] + 1, (subset.shape[0],),
                                device=images.device, generator=generator)
        return replace(images, indices, jpeg_compression(subset, quality))

    def _coarse_dropout(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["coarse_dropout"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        count, device, height, width = subset.shape[0], images.device, images.shape[-2], images.shape[-1]
        counts = torch.randint(
            cfg["patches"][0], cfg["patches"][1] + 1, (count,), device=device, generator=generator
        )
        side = uniform(
            (count, cfg["patches"][1]),
            math.sqrt(cfg["area"][0]),
            math.sqrt(cfg["area"][1]),
            device,
            generator,
        )
        box_height = (height * side).round().clamp(1, height).long()
        box_width = (width * side).round().clamp(1, width).long()
        top = (
            torch.rand(count, cfg["patches"][1], device=device, generator=generator)
            * (height - box_height + 1)
        ).floor().long()
        left = (
            torch.rand(count, cfg["patches"][1], device=device, generator=generator)
            * (width - box_width + 1)
        ).floor().long()
        return replace(images, indices, coarse_dropout(subset, (top, left, box_height, box_width), counts, cfg["fill"]))

    def _patch_brightness(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        cfg = self.config["patch_brightness"]
        indices, subset = _pick(images, cfg["probability"], generator)
        if indices is None:
            return images
        count, device, height, width = subset.shape[0], images.device, images.shape[-2], images.shape[-1]
        box_height = (
            height * uniform((count,), *cfg["size"], device, generator)
        ).round().clamp(1, height).long()
        box_width = (
            width * uniform((count,), *cfg["size"], device, generator)
        ).round().clamp(1, width).long()
        top = (
            torch.rand(count, device=device, generator=generator) * (height - box_height + 1)
        ).floor().long()
        left = (
            torch.rand(count, device=device, generator=generator) * (width - box_width + 1)
        ).floor().long()
        factor = uniform((count,), *cfg["factor"], device, generator)
        boxes = (top, left, box_height, box_width)
        return replace(images, indices, patch_brightness(subset, boxes, factor, cfg["softness"]))


def _pick(images: torch.Tensor, probability: float,
          generator: torch.Generator | None) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``(indices, subset)``: the samples a transform gets (see ``select``), ``indices`` ``None`` for none."""
    indices = select(images.shape[0], probability, images.device, generator)
    if indices is None or indices.numel() == images.shape[0]:
        return indices, images
    return indices, images.index_select(0, indices)


def _scatter_back(images: torch.Tensor, indices: torch.Tensor, rest: torch.Tensor,
                  result: torch.Tensor) -> torch.Tensor:
    """A copy of ``images`` with ``result`` on ``indices``; ``rest`` is copied unchanged.

    Writing both parts into a new tensor moves fewer bytes than copying all of ``images`` and
    then overwriting the augmented rows. ``indices`` and ``rest`` are known on the host.
    """
    out = torch.empty_like(images)
    out.index_copy_(0, indices, result)
    out.index_copy_(0, rest, images.index_select(0, rest))
    return out
