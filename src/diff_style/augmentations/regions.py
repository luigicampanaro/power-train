"""Box-based augmentations: erase or brighten one or more regions of an image.

All operations take ``(B, N, 3, H, W)`` images and stay on the image device. ``coarse_dropout``
erases whole patches; ``patch_brightness`` scales one soft-edged patch. ``Augmentor`` draws the
boxes; the gallery notebook passes them explicitly.
"""

from __future__ import annotations

import torch


def coarse_dropout(images: torch.Tensor, boxes: tuple[torch.Tensor, ...],
                   counts: torch.Tensor, fill: float) -> torch.Tensor:
    batch, _, _, height, width = images.shape
    top, left, box_height, box_width = boxes
    mask = torch.zeros(batch, 1, height, width, dtype=torch.bool, device=images.device)
    rows = torch.arange(height, device=images.device).reshape(1, height)
    columns = torch.arange(width, device=images.device).reshape(1, width)
    for index in range(top.shape[1]):
        valid = (index < counts).reshape(batch, 1, 1)
        row = (rows >= top[:, index:index + 1]) & (rows < (top[:, index] + box_height[:, index]).reshape(batch, 1))
        column = (columns >= left[:, index:index + 1]) & (
            columns < (left[:, index] + box_width[:, index]).reshape(batch, 1)
        )
        mask |= (row.reshape(batch, height, 1) & column.reshape(batch, 1, width) & valid).unsqueeze(1)
    return torch.where(mask.unsqueeze(1), fill, images)


def patch_brightness(images: torch.Tensor, boxes: tuple[torch.Tensor, ...],
                     factor: torch.Tensor, softness: float) -> torch.Tensor:
    """Multiply one box per sample by ``factor``, fading to no change at the box border.

    ``softness`` is the faded part of each half side of the box: 0 is a hard edge, 1 fades all
    the way to the center. The box center always gets the full ``factor``, whatever the box size.
    """
    batch, _, _, height, width = images.shape
    top, left, box_height, box_width = (value.reshape(batch, 1) for value in boxes)

    def ramp(start: torch.Tensor, size: torch.Tensor, length: int) -> torch.Tensor:
        centers = torch.arange(length, device=images.device) + 0.5  # pixel centers
        inside = torch.minimum(centers - start, start + size - centers)  # distance to the nearer edge, > 0 inside
        fade = (softness * size / 2).clamp_min(1e-6)  # width of the fade, in pixels
        return (inside / fade).clamp(0.0, 1.0)  # (batch, length): 0 outside, 1 past the fade

    weight = ramp(top, box_height, height).reshape(batch, height, 1) * ramp(left, box_width, width).reshape(batch, 1, width)
    mask = 1.0 + weight * (factor.reshape(batch, 1, 1).to(weight) - 1.0)
    return (images * mask.reshape(batch, 1, 1, height, width).to(images)).clamp(0.0, 1.0)


__all__ = ["coarse_dropout", "patch_brightness"]
