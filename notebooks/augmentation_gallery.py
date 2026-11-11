import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import base64
    import math
    from pathlib import Path

    import hydra
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torchvision.io import encode_jpeg, read_image

    from diff_style.augmentations.ops import (
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
        sharpness,
    )
    from diff_style.models.preprocess import preprocess_images

    CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
    ASSET = Path(__file__).resolve().parent.parent / "assets" / "robomimic_can.png"

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        CFG = hydra.compose("train", overrides=["output_dir=/tmp/diff_style_gallery"])
    AUG = hydra.utils.instantiate(CFG.augmentation)

    RAW_SIZE = tuple(CFG.frame_size)
    IMAGE_SIZE = tuple(CFG.image_size)
    IMAGE = preprocess_images(read_image(str(ASSET)), IMAGE_SIZE)  # same preprocessing as training
    HEIGHT, WIDTH = IMAGE.shape[-2:]
    BATCH = IMAGE.unsqueeze(0).unsqueeze(0)

    def image_html(image: torch.Tensor):
        display_height = max(1, round(120 * HEIGHT / WIDTH))
        pixels = image.detach().clamp(0, 1).permute(1, 2, 0).mul(255).round()
        tensor = F.interpolate(
            pixels.permute(2, 0, 1).unsqueeze(0),
            size=(display_height, 120),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        jpeg = encode_jpeg(tensor.to(torch.uint8), quality=92)
        assert isinstance(jpeg, torch.Tensor)  # one image in, one encoded image out
        payload = jpeg.numpy().tobytes()
        data = base64.b64encode(payload).decode()
        return mo.Html(f'<img src="data:image/jpeg;base64,{data}" width="120" alt="augmentation" />')

    def row(title: str, items):
        cells = [("original", IMAGE)] + list(items)
        cells = [
            mo.vstack([image_html(image), mo.md(f"`{label}`")], align="center")
            for label, image in cells
        ]
        return mo.vstack(
            [mo.md(f"### {title}"), mo.hstack(cells, justify="start", align="start")],
            align="start",
        )

    def training(name: str, key: str) -> str:
        """The training range of a setting and its probability, read from the Hydra config."""
        value = AUG.config[name][key]
        text = f"{value[0]} to {value[1]}" if isinstance(value, tuple) else f"{value}"
        return f"training {text}, p={AUG.config[name]['probability']}"

    def parameter_row(title: str, values, label, apply):
        return row(title, [(label.format(value=value), apply(value)) for value in values])

    def view(op, *args, **kwargs):
        return op(BATCH, *args, **kwargs)[0, 0].clamp(0, 1)

    def simple(op):
        return lambda value: view(op, value)

    def int_box(top, left, height, width):
        return tuple(torch.tensor([[value]], dtype=torch.int64) for value in (top, left, height, width))

    def random_box(size: float, count: int = 1):
        side = math.sqrt(size)
        height = max(1, round(HEIGHT * side))
        width = max(1, round(WIDTH * side))
        generator = torch.Generator().manual_seed(0)
        top = torch.randint(0, HEIGHT - height + 1, (1, count), generator=generator)
        left = torch.randint(0, WIDTH - width + 1, (1, count), generator=generator)
        return (
            top,
            left,
            torch.full((1, count), height, dtype=torch.int64),
            torch.full((1, count), width, dtype=torch.int64),
        )

    def crop_ratio(value: float):
        height = max(1, round(HEIGHT * value))
        width = max(1, round(WIDTH * value))
        return view(
            crop_rotate,
            (HEIGHT - height) // 2,
            (WIDTH - width) // 2,
            height,
            width,
            0.0,
            (HEIGHT, WIDTH),
            "border",
        )

    def crop_x(value: float):
        height = round(HEIGHT * 0.8)
        width = round(WIDTH * 0.8)
        return view(
            crop_rotate,
            (HEIGHT - height) // 2,
            round((WIDTH - width) * value),
            height,
            width,
            0.0,
            (HEIGHT, WIDTH),
            "border",
        )

    def crop_y(value: float):
        height = round(HEIGHT * 0.8)
        width = round(WIDTH * 0.8)
        return view(
            crop_rotate,
            round((HEIGHT - height) * value),
            (WIDTH - width) // 2,
            height,
            width,
            0.0,
            (HEIGHT, WIDTH),
            "border",
        )

    def rotation(value: float):
        return view(crop_rotate, 0, 0, HEIGHT, WIDTH, value, (HEIGHT, WIDTH), "reflection")

    def motion(size: int, angle: float):
        kernel = motion_kernel(size, angle, BATCH.device, BATCH.dtype)
        return view(motion_blur, kernel[None])

    def patch(value: float, size: float = 0.25, softness: float = 0.5):
        height = max(1, round(HEIGHT * size))
        width = max(1, round(WIDTH * size))
        boxes = int_box((HEIGHT - height) // 2, (WIDTH - width) // 2, height, width)
        return view(patch_brightness, boxes, torch.tensor([value]), softness)


@app.cell
def _():
    mo.md(
        f"""
        # Augmentation gallery

        Each row uses one public operation from the training path. The first tile is the
        original image. The other tiles show one parameter value. The last section uses the
        full Hydra config with all augmentations together.

        The gallery uses the model input size **{HEIGHT} x {WIDTH}** from
        `configs/train.yaml`. Raw dataset frames are **{RAW_SIZE[0]} x {RAW_SIZE[1]}**.
        The datamodule resizes raw frames before augmentation.
        """
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(
                f"crop ratio ({training('crop', 'ratio')})",
                (0.50, 0.70, 0.90, 0.95),
                "ratio={value:.2f}",
                crop_ratio,
            ),
            parameter_row(
                "horizontal position (ratio 0.80)",
                (0.0, 0.33, 0.66, 1.0),
                "left={value:.2f}",
                crop_x,
            ),
            parameter_row(
                "vertical position (ratio 0.80)",
                (0.0, 0.33, 0.66, 1.0),
                "top={value:.2f}",
                crop_y,
            ),
        ]
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(f"rotation ({training('rotation', 'degrees')} deg, +/-)", (-15.0, -5.0, 5.0, 15.0), "{value:+.0f} deg", rotation),
        ]
    )
    return


@app.cell
def _():
    def color(parameter):
        return lambda value: view(color_jitter, **{parameter: value})

    mo.vstack(
        [
            parameter_row(
                f"brightness ({training('color_jitter', 'brightness')})",
                (0.60, 0.80, 1.20, 1.40),
                "{value:.2f}",
                color("brightness"),
            ),
            parameter_row(
                f"contrast ({training('color_jitter', 'contrast')})",
                (0.60, 0.80, 1.20, 1.40),
                "{value:.2f}",
                color("contrast"),
            ),
            parameter_row(
                f"saturation ({training('color_jitter', 'saturation')})",
                (0.50, 0.75, 1.25, 1.50),
                "{value:.2f}",
                color("saturation"),
            ),
            parameter_row(
                f"hue ({training('color_jitter', 'hue')})",
                (-0.08, -0.05, 0.05, 0.08),
                "{value:+.2f}",
                color("hue"),
            ),
        ]
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(
                f"sharpness ({training('sharpness', 'factor')})",
                (0.0, 0.5, 1.5, 2.0),
                "{value:.2f}",
                simple(sharpness),
            ),
            parameter_row(
                f"planckian temperature ({training('planckian_jitter', 'temperature')})",
                (3000, 5000, 9000, 15000),
                "{value} K",
                simple(planckian_jitter),
            ),
            parameter_row(
                f"gaussian noise sigma ({training('gaussian_noise', 'sigma')})",
                (0.01, 0.05, 0.10, 0.20),
                "{value:.2f}",
                simple(gaussian_noise),
            ),
            parameter_row(
                f"gamma ({training('gamma_correction', 'gamma')})",
                (0.5, 0.8, 1.2, 2.0),
                "{value:.2f}",
                simple(gamma_correction),
            ),
        ]
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(
                f"motion blur kernel ({training('motion_blur', 'kernel_size')})",
                (3, 5, 7, 9),
                "kernel={value}",
                lambda value: motion(value, 30.0),
            ),
            parameter_row(
                "motion blur angle (kernel 7)",
                (0, 45, 90, 135),
                "{value} deg",
                lambda value: motion(7, value),
            ),
            parameter_row(
                f"JPEG quality ({training('jpeg_compression', 'quality')})",
                (10, 30, 50, 75),
                "{value}",
                simple(jpeg_compression),
            ),
        ]
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(
                f"dropout patches ({training('coarse_dropout', 'patches')})",
                (1, 2, 4, 8),
                "{value} patch(es)",
                lambda value: view(coarse_dropout, random_box(0.03, 8), torch.tensor([value]), 0.0),
            ),
            parameter_row(
                f"dropout area ({training('coarse_dropout', 'area')})",
                (0.02, 0.05, 0.10, 0.20),
                "{value:.2f}",
                lambda value: view(coarse_dropout, random_box(value), torch.tensor([1]), 0.0),
            ),
            parameter_row(
                f"dropout fill ({training('coarse_dropout', 'fill')})",
                (0.0, 0.33, 0.66, 1.0),
                "{value:.2f}",
                lambda value: view(coarse_dropout, random_box(0.05), torch.tensor([1]), value),
            ),
        ]
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            parameter_row(
                f"patch factor ({training('patch_brightness', 'factor')})",
                (0.2, 0.5, 0.8, 1.2),
                "{value:.2f}",
                patch,
            ),
            parameter_row(
                f"patch size ({training('patch_brightness', 'size')})",
                (0.05, 0.10, 0.20, 0.35),
                "{value:.2f}",
                lambda value: patch(0.6, size=value),
            ),
            parameter_row(
                f"patch softness ({training('patch_brightness', 'softness')})",
                (0.0, 0.25, 0.5, 1.0),
                "{value:.2f}",
                lambda value: patch(0.6, softness=value),
            ),
        ]
    )
    return


@app.cell
def _():
    rows, per_row = 3, 6
    outputs = AUG(BATCH.repeat(rows * per_row, 1, 1, 1, 1), generator=torch.Generator().manual_seed(0))

    def tile(image, label):
        return mo.vstack([image_html(image), mo.md(f"`{label}`")], align="center")

    mo.vstack(
        [mo.md(f"### all augmentations with the Hydra config ({AUG.clean_probability:.0%} of the samples stay clean)"), tile(IMAGE, "original")]
        + [
            mo.hstack(
                [
                    tile(outputs[index, 0], f"sample {index + 1}")
                    for index in range(start, start + per_row)
                ],
                justify="start",
                align="start",
            )
            for start in range(0, rows * per_row, per_row)
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
