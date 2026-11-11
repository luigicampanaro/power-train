import torch
import torch.nn.functional as F


def preprocess_images(images: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Raw camera frames to model input: uint8 (..., 3, H, W) -> float32 in [0, 1], resized to ``image_size``.

    Training, validation, test, and robot inference must all call this, so the model always
    sees the same input.
    """
    images = images.float() / 255.0
    if tuple(images.shape[-2:]) == tuple(image_size):
        return images
    flat = images.reshape(-1, *images.shape[-3:])
    flat = F.interpolate(flat, size=tuple(image_size), mode="bilinear", align_corners=False, antialias=True)
    flat.clamp_(0.0, 1.0)  # the antialiased resize can round 255 up to 1.0000002
    return flat.reshape(*images.shape[:-2], *image_size)
