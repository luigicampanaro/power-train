"""Benchmark the augmentation overhead against an identity augmentor.

The default frame size is ``image_size`` from ``configs/train.yaml``: the datamodule
resizes raw frames to it before augmentation.

Examples:
    uv run python scripts/benchmark_augmentation.py
    uv run python scripts/benchmark_augmentation.py --height 480 --width 640
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import hydra
import torch

from diff_style.augmentations import DEFAULT_CONFIG, Augmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def time_forward(augmentor, images: torch.Tensor, generator: torch.Generator, iters: int) -> float:
    for _ in range(5):
        augmentor(images, generator=generator)
    if images.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        augmentor(images, generator=generator)
    if images.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose("train", overrides=["output_dir=/tmp/diff_style_benchmark"])
    augmentor = hydra.utils.instantiate(cfg.augmentation).to(device)
    identity = Augmentor({name: {"probability": 0.0} for name in DEFAULT_CONFIG}).to(device)
    height = args.height or int(cfg.image_size[0])
    width = args.width or int(cfg.image_size[1])

    images = torch.rand(
        args.batch_size,
        args.frames,
        args.cameras,
        3,
        height,
        width,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    identity_ms = time_forward(identity, images, generator, args.iters)
    augmented_ms = time_forward(augmentor, images, generator, args.iters)

    print(f"device       {device}")
    print(f"shape        {tuple(images.shape)}")
    print(f"identity     {identity_ms:.3f} ms")
    print(f"augmented    {augmented_ms:.3f} ms")
    print(f"overhead     {augmented_ms - identity_ms:.3f} ms")
    print(f"throughput   {args.batch_size / (augmented_ms / 1e3):.0f} samples/s")


if __name__ == "__main__":
    main()
