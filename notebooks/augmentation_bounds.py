import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import base64
    import math
    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path

    import h5py
    import hydra
    import marimo as mo
    import numpy as np
    import torch
    import yaml
    from torchvision.io import encode_jpeg, encode_png

    from diff_style.augmentations import Augmentor
    from diff_style.augmentations.ops import (
        INTEGER_SETTINGS,
        LIMITS,
        PADDING_MODES,
        check_config,
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
    from diff_style.datasets.episode_dataset import list_episodes
    from diff_style.models.preprocess import preprocess_images

    CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
    OUTPUT = Path(__file__).resolve().parent / "augmentation_selected.yaml"
    AUGMENTATION = "tuned"  # configs/augmentation/tuned.yaml: where the controls start
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        CFG = hydra.compose("train", overrides=["output_dir=/tmp/diff_style_augmentation_bounds",
                                                f"augmentation={AUGMENTATION}"])
    TRAINING = hydra.utils.instantiate(CFG.augmentation)

    IMAGE_SIZE = tuple(CFG.image_size)
    HEIGHT, WIDTH = IMAGE_SIZE
    EPISODES = list_episodes(CFG.datamodule.dataset_dir)
    with h5py.File(EPISODES[0], "r", locking=False) as _file:
        CAMERAS = sorted(_file["observations/images"])

    def episode_length(path: Path) -> int:
        with h5py.File(path, "r", locking=False) as file:
            return len(file["action"])

    LENGTHS = [episode_length(path) for path in EPISODES]

    # Each setting as "section.key": the start values of all controls, from the augmentation config
    DEFAULTS = {
        f"{name}.{key}": value for name, section in TRAINING.config.items() for key, value in section.items()
    } | {"clean_probability": TRAINING.clean_probability}

    def load_settings(path: Path):
        """The settings of a file this notebook wrote, or None if `path` is not such a file."""
        if not path.exists():
            return None
        saved = yaml.safe_load(path.read_text())
        if not isinstance(saved, dict) or "config" not in saved:
            return None
        settings = {f"{name}.{key}": tuple(value) if isinstance(value, list) else value
                    for name, section in saved["config"].items() for key, value in section.items()}
        return settings | {"clean_probability": saved.get("clean_probability", TRAINING.clean_probability)}

    # A reload or a restart of the notebook keeps the selected settings: start where the last session stopped
    STARTS = load_settings(OUTPUT) or DEFAULTS

    # (start, stop, step) of each slider: wide, to look well past the useful values. A slider grows to
    # include its value. The number boxes accept every value inside LIMITS, at 4 decimals.
    SPANS = {
        "crop.ratio": (0.20, 1.00, 0.005),
        "rotation.degrees": (0.0, 45.0, 0.5),
        "color_jitter.brightness": (0.0, 3.0, 0.01),
        "color_jitter.contrast": (0.0, 3.0, 0.01),
        "color_jitter.saturation": (0.0, 3.0, 0.01),
        "color_jitter.hue": (-0.5, 0.5, 0.005),
        "sharpness.factor": (0.0, 5.0, 0.05),
        "planckian_jitter.temperature": (2000, 25000, 100),
        "gaussian_noise.sigma": (0.0, 0.5, 0.0025),
        "motion_blur.kernel_size": (1, 31, 1),
        "motion_blur.angle": (-180.0, 180.0, 5.0),
        "jpeg_compression.quality": (1, 100, 1),
        "coarse_dropout.patches": (1, 20, 1),
        "coarse_dropout.area": (0.0, 0.5, 0.0025),
        "coarse_dropout.fill": (0.0, 1.0, 0.01),
        "patch_brightness.factor": (0.0, 3.0, 0.01),
        "patch_brightness.size": (0.0, 1.0, 0.01),
        "patch_brightness.softness": (0.0, 1.0, 0.01),
        "gamma_correction.gamma": (0.1, 4.0, 0.01),
    }
    PROBABILITY_SPAN = (0.0, 1.0, 0.01)
    DECIMALS = 4
    JPEG_QUALITY = 95  # the grids of many samples: lossless PNG there is larger than marimo shows
    COLUMNS = [1, 3]  # widths of the control column and the tile column of each line

    def load_frames(episode: int, frame: int) -> torch.Tensor:
        """All cameras of one frame as the model sees them: (cameras, 3, H, W), float in [0, 1]."""
        with h5py.File(EPISODES[episode], "r", locking=False) as file:
            frames = np.stack([file[f"observations/images/{camera}"][frame] for camera in CAMERAS])
        return preprocess_images(torch.from_numpy(frames).permute(0, 3, 1, 2), IMAGE_SIZE)

    def random_frame(generator: torch.Generator) -> tuple[int, int]:
        episode = int(torch.randint(len(EPISODES), (), generator=generator))
        return episode, int(torch.randint(LENGTHS[episode], (), generator=generator))

    def is_integer(path: str) -> bool:
        return tuple(path.split(".")) in INTEGER_SETTINGS

    def limits(path: str) -> tuple[float | None, float | None]:
        """The allowed (lowest, highest) value of a setting, ``None`` for no limit."""
        if "." not in path or path.endswith(".probability"):
            return 0.0, 1.0
        name, key = path.split(".")
        return tuple(None if math.isinf(value) else value for value in LIMITS[name][key])

    def slider(path: str, value):
        """A full-width slider for one setting at ``value``; its span grows to include ``value``."""
        start, stop, step = SPANS.get(path, PROBABILITY_SPAN)
        label = path.split(".")[-1]
        if isinstance(value, tuple):
            low, high = sorted(value)
            return mo.ui.range_slider(start=min(start, low), stop=max(stop, high), step=step, value=[low, high],
                                      label=label, show_value=True, debounce=True, full_width=True)
        return mo.ui.slider(start=min(start, value), stop=max(stop, value), step=step, value=value,
                            label=label, show_value=True, debounce=True, full_width=True)

    def number(path: str, value: float):
        """A box to type one exact value; it sends the value on Enter or when it loses the focus."""
        lowest, highest = limits(path)
        step = 1 if is_integer(path) else 10.0**-DECIMALS
        return mo.ui.number(start=lowest, stop=highest, step=step, value=value, debounce=True)

    def widgets(path: str, value):
        """The controls of one setting: a slider with number boxes, or a dropdown for a name."""
        if isinstance(value, str):
            return mo.ui.dictionary({"choice": mo.ui.dropdown(PADDING_MODES, value=value, label=path.split(".")[-1])})
        if isinstance(value, tuple):
            return mo.ui.dictionary({"slider": slider(path, value), "low": number(path, value[0]),
                                     "high": number(path, value[1])})
        return mo.ui.dictionary({"slider": slider(path, value), "value": number(path, value)})

    def resolve(old, controls: dict):
        """The new value of one setting: the slider if it moved, else the number boxes."""
        if "choice" in controls:
            return controls["choice"]
        if isinstance(old, tuple):
            moved, typed = tuple(controls["slider"]), (controls["low"], controls["high"])
            shown = tuple(sorted(old))
        else:
            moved, typed, shown = controls["slider"], controls["value"], old
        if moved != shown:
            return moved
        empty = typed is None or (isinstance(typed, tuple) and None in typed)
        return old if empty else typed

    def field(element):
        """One setting on screen: its slider (or dropdown), then its number boxes."""
        if "choice" in element:
            return element["choice"]
        boxes = [part for name in ("low", "high", "value") if name in element for part in (mo.md(name), element[name])]
        return mo.vstack([element["slider"], mo.hstack(boxes, justify="start", align="center")], gap=0.25)

    def setting(path: str, value):
        """A control value as the Augmentor expects it: integers or rounded floats, ranges as tuples."""
        if isinstance(value, str):
            return value
        convert = int if is_integer(path) else lambda number: round(float(number), DECIMALS)
        return tuple(map(convert, value)) if isinstance(value, (list, tuple)) else convert(value)

    def typed(settings: dict) -> dict:
        return {path: setting(path, value) for path, value in settings.items()}

    def nested(settings: dict) -> dict:
        """The Augmentor config of the "section.key" settings."""
        config = {name: {} for name in TRAINING.config}
        for path, value in settings.items():
            if "." in path:
                name, key = path.split(".")
                config[name][key] = value
        return config

    def config_error(config: dict) -> str:
        try:
            check_config(config)
        except ValueError as error:
            return str(error)
        return ""

    def image_html(image: torch.Tensor):
        """Lossless PNG at the model input size, so noise and JPEG artifacts stay visible."""
        pixels = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
        data = base64.b64encode(encode_png(pixels).numpy().tobytes()).decode()
        return mo.Html(f'<img src="data:image/png;base64,{data}" width="{WIDTH}" alt="augmentation" />')

    def thumbnail_html(image: torch.Tensor):
        """The same pixels as a JPEG, for the grid of many samples: lossless PNG there is larger than
        the output size that marimo shows (30 samples x 2 cameras is about 5 MB of base64)."""
        pixels = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
        data = base64.b64encode(encode_jpeg(pixels, quality=JPEG_QUALITY).numpy().tobytes()).decode()
        return mo.Html(f'<img src="data:image/jpeg;base64,{data}" width="{WIDTH}" alt="augmentation" />')

    def tile(label: str, image: torch.Tensor):
        return mo.vstack([image_html(image), mo.md(f"`{label}`")], align="center")

    def line(left: list, tiles: list):
        """One line of a section: the controls on the left, next to the tiles they change."""
        return mo.hstack(
            [mo.vstack(left, gap=0.75), mo.hstack(tiles, justify="start", align="start", wrap=True)],
            widths=COLUMNS, align="start", gap=2,
        )

    def title(name: str):
        return mo.md(f"### {name}")

    def first(images: torch.Tensor) -> torch.Tensor:
        return images[0, 0]

    def bounds(values: tuple, apply) -> list:
        """The tiles of the two ends of a range, ``apply`` maps one value to a (1, 1, 3, H, W) image."""
        return [tile(f"{value:g}", first(apply(value))) for value in values]

    def box(ratio: float, place: float) -> tuple[int, int, int, int]:
        """(top, left, height, width) of a box of side ``ratio``: ``place`` 0 is top-left, 1 bottom-right."""
        height, width = max(1, round(HEIGHT * ratio)), max(1, round(WIDTH * ratio))
        return round((HEIGHT - height) * place), round((WIDTH - width) * place), height, width

    @dataclass(frozen=True)
    class Geometry:
        """One crop box and one rotation, like one draw of ``Augmentor._crop_rotate``."""

        ratio: float = 1.0
        place: float = 0.5
        angle: float = 0.0
        padding: str = "border"

        def view(self, image: torch.Tensor) -> torch.Tensor:
            return first(crop_rotate(image, *box(self.ratio, self.place), self.angle, IMAGE_SIZE, self.padding))

    def centered_box(size: float) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor([value]) for value in box(size, 0.5))

    def dropout_boxes(patches: int, area: float) -> tuple[torch.Tensor, ...]:
        """``patches`` boxes of ``area`` each, at the same random places for each call."""
        _, _, height, width = box(math.sqrt(area), 0.0)
        generator = torch.Generator().manual_seed(0)
        top = torch.randint(0, HEIGHT - height + 1, (1, patches), generator=generator)
        left = torch.randint(0, WIDTH - width + 1, (1, patches), generator=generator)
        return top, left, torch.full((1, patches), height), torch.full((1, patches), width)

    def kernel(size: int, angle: float) -> torch.Tensor:
        return motion_kernel(size, angle, torch.device("cpu"), torch.float32)[None]

    def formatted(value) -> str:
        return f"[{value[0]}, {value[1]}]" if isinstance(value, tuple) else f"{value}"

    def yaml_text(settings: dict) -> str:
        """The settings as a Hydra augmentation config, in the layout of configs/augmentation/default.yaml."""
        lines = [
            f"# Written by notebooks/{Path(__file__).name} on {datetime.now():%Y-%m-%d %H:%M}.",
            "# Copy to configs/augmentation/ and select it with augmentation=<file name>.",
            "_target_: diff_style.augmentations.Augmentor",
            f"clean_probability: {settings['clean_probability']}",
            "config:",
        ]
        for name, section in nested(settings).items():
            lines.append(f"  {name}:")
            lines += [f"    {key}: {formatted(value)}" for key, value in section.items()]
        return "\n".join(lines) + "\n"

    def changed(settings: dict) -> dict:
        """The settings that differ from the training config."""
        defaults = typed(DEFAULTS)
        return {path: value for path, value in settings.items() if value != defaults[path]}

    def overrides(settings: dict) -> list[str]:
        """Hydra command-line overrides for the settings that differ from the training config."""
        return [
            f"augmentation.{'' if path == 'clean_probability' else 'config.'}{path}={formatted(value).replace(' ', '')}"
            for path, value in changed(settings).items()
        ]


@app.cell
def _():
    mo.md(
        f"""
        # Augmentation bounds on the tomato picking frames

        Each line shows the two ends of each range on one real frame, next to its controls. Drag
        a slider to explore, or type an exact value in a box (push Enter or click outside the box).
        The sliders go far past the useful values on purpose. The last section runs the full
        `Augmentor` with all settings.

        All settings start at the values of `{OUTPUT}` when that file exists, so a reload or a restart
        of the notebook keeps them, and at `configs/augmentation/{AUGMENTATION}.yaml` otherwise. When
        one setting differs from that config, the notebook writes all settings to `{OUTPUT}` after
        each change. A setting that you do not change keeps its current value in that file.

        Frames come from `{CFG.datamodule.dataset_dir}` ({len(EPISODES)} episodes, cameras
        {", ".join(CAMERAS)}). Each frame is resized to **{HEIGHT} x {WIDTH}** first, like in training.
        """
    )
    return


@app.cell
def _():
    episode = mo.ui.dropdown({path.stem: index for index, path in enumerate(EPISODES)},
                             value=EPISODES[0].stem, label="episode")
    camera = mo.ui.dropdown(CAMERAS, value=CAMERAS[0], label="camera")
    return camera, episode


@app.cell
def _(episode):
    frame = mo.ui.slider(0, LENGTHS[episode.value] - 1, value=LENGTHS[episode.value] // 2, label="frame",
                         show_value=True, debounce=True, full_width=True)
    return (frame,)


@app.cell
def _():
    get_settings, set_settings = mo.state(STARTS, allow_self_loops=True)  # self loops: a slider updates its boxes
    return get_settings, set_settings


@app.cell
def _(get_settings, set_settings):
    def _update(values):
        set_settings(lambda old: {path: resolve(value, values[path]) for path, value in old.items()})

    controls = mo.ui.dictionary({path: widgets(path, value) for path, value in get_settings().items()},
                                on_change=_update)
    return (controls,)


@app.cell
def _(camera, episode, frame):
    frames = load_frames(episode.value, frame.value)  # (cameras, 3, H, W)
    image = frames[CAMERAS.index(camera.value)][None, None]  # (1, 1, 3, H, W): the input of the ops
    original_tile = tile("original", first(image))
    line([title("source frame"), episode, camera, frame], [tile(name, view) for name, view in zip(CAMERAS, frames)])
    return frames, image, original_tile


@app.cell
def _(get_settings):
    settings = typed(get_settings())
    config = nested(settings)
    _error = config_error(config)
    mo.stop(bool(_error), mo.callout(mo.md(f"**Invalid settings:** {_error}"), kind="danger"))
    augmentor = Augmentor(config, settings["clean_probability"])
    return augmentor, config, settings


@app.cell
def _(config, controls, image, original_tile):
    _low, _high = config["crop"]["ratio"]
    _degrees, _padding = config["rotation"]["degrees"], config["rotation"]["padding_mode"]
    _crops = [
        tile(f"{_low:g}, top-left", Geometry(_low, 0.0).view(image)),
        tile(f"{_low:g}, center", Geometry(_low).view(image)),
        tile(f"{_low:g}, bottom-right", Geometry(_low, 1.0).view(image)),
        tile(f"{_high:g}, center", Geometry(_high).view(image)),
    ]
    _rotations = [
        tile(f"-{_degrees:g} deg", Geometry(angle=-_degrees, padding=_padding).view(image)),
        tile(f"+{_degrees:g} deg", Geometry(angle=_degrees, padding=_padding).view(image)),
        tile(f"+{_degrees:g} deg, crop {_low:g} top-left", Geometry(_low, 0.0, _degrees, _padding).view(image)),
        tile(f"-{_degrees:g} deg, crop {_low:g} bottom-right", Geometry(_low, 1.0, -_degrees, _padding).view(image)),
    ]
    mo.vstack([
        line([title("crop"), field(controls["crop.probability"]), field(controls["crop.ratio"])],
             [original_tile, *_crops]),
        line([title("rotation (with the smallest crop on the right)"),
              *(field(controls[f"rotation.{key}"]) for key in ("probability", "degrees", "padding_mode"))],
             [original_tile, *_rotations]),
    ])
    return


@app.cell
def _(config, controls, image, original_tile):
    _jitter = {key: value for key, value in config["color_jitter"].items() if key != "probability"}

    def _color(key):
        return lambda value: color_jitter(image, **{key: value})

    _lines = [
        line([field(controls[f"color_jitter.{key}"])], [original_tile, *bounds(values, _color(key))])
        for key, values in _jitter.items()
    ]
    _lines.append(line([mo.md("all four settings together")], [
        original_tile,
        tile("all low", first(color_jitter(image, *(values[0] for values in _jitter.values())))),
        tile("all high", first(color_jitter(image, *(values[1] for values in _jitter.values())))),
    ]))
    mo.vstack([line([title("color_jitter"), field(controls["color_jitter.probability"])], []), *_lines])
    return


@app.cell
def _(config, controls, image, original_tile):
    _transforms = {
        "sharpness": ("factor", lambda value: sharpness(image, value)),
        "planckian_jitter": ("temperature", lambda value: planckian_jitter(image, value)),
        "gamma_correction": ("gamma", lambda value: gamma_correction(image, value)),
        "gaussian_noise": ("sigma", lambda value: gaussian_noise(image, value, torch.Generator().manual_seed(0))),
        "jpeg_compression": ("quality", lambda value: jpeg_compression(image, value)),
    }
    mo.vstack([
        line([title(name), field(controls[f"{name}.probability"]), field(controls[f"{name}.{key}"])],
             [original_tile, *bounds(config[name][key], apply)])
        for name, (key, apply) in _transforms.items()
    ])
    return


@app.cell
def _(config, controls, image, original_tile):
    _blur = config["motion_blur"]
    _sizes = range(_blur["kernel_size"][0] | 1, _blur["kernel_size"][1] + 1, 2)  # odd sizes only, like training
    _low, _high = _blur["angle"]
    _middle = (_low + _high) / 2
    _tiles = [tile(f"kernel {_sizes[0]}, {_middle:g} deg", first(motion_blur(image, kernel(_sizes[0], _middle))))]
    _tiles += [
        tile(f"kernel {_sizes[-1]}, {angle:g} deg", first(motion_blur(image, kernel(_sizes[-1], angle))))
        for angle in (_low, _middle, _high)
    ]
    line([title("motion_blur"), *(field(controls[f"motion_blur.{key}"]) for key in ("probability", "kernel_size", "angle"))],
         [original_tile, *_tiles])
    return


@app.cell
def _(config, controls, image, original_tile):
    _drop = config["coarse_dropout"]
    _dropouts = [
        tile(f"{patches} x area {area:g}",
             first(coarse_dropout(image, dropout_boxes(patches, area), torch.tensor([patches]), _drop["fill"])))
        for patches in _drop["patches"]
        for area in _drop["area"]
    ]
    _patch = config["patch_brightness"]
    _patches = [
        tile(f"factor {factor:g}, size {size:g}",
             first(patch_brightness(image, centered_box(size), torch.tensor([factor]), _patch["softness"])))
        for factor in _patch["factor"]
        for size in _patch["size"]
    ]
    mo.vstack([
        line([title("coarse_dropout"),
              *(field(controls[f"coarse_dropout.{key}"]) for key in ("probability", "patches", "area", "fill"))],
             [original_tile, *_dropouts]),
        line([title("patch_brightness"),
              *(field(controls[f"patch_brightness.{key}"]) for key in ("probability", "factor", "size", "softness"))],
             [original_tile, *_patches]),
    ])
    return


@app.cell
def _():
    seed = mo.ui.number(start=0, stop=1_000_000, step=1, value=0, label="seed")
    random_frames = mo.ui.switch(value=True, label="random frames from all episodes")
    samples = mo.ui.slider(6, 48, 6, value=18, label="samples", show_value=True, debounce=True, full_width=True)
    return random_frames, samples, seed


@app.cell
def _(frames, random_frames, samples, seed):
    if random_frames.value:
        _generator = torch.Generator().manual_seed(seed.value)
        batch = torch.stack([load_frames(*random_frame(_generator)) for _ in range(samples.value)])
    else:
        batch = frames.expand(samples.value, *frames.shape).clone()
    batch = batch.unsqueeze(1)  # (samples, 1 frame, cameras, 3, H, W), like a training batch
    return (batch,)


@app.cell
def _(augmentor, batch, controls, random_frames, samples, seed):
    _outputs = augmentor(batch, generator=torch.Generator().manual_seed(seed.value))
    _tiles = [
        mo.vstack(
            [*(thumbnail_html(view) for view in _outputs[index, 0]),
             mo.md(f"`sample {index}{' (clean)' if torch.equal(_outputs[index], batch[index]) else ''}`")],
            align="center",
        )
        for index in range(len(batch))
    ]
    line([
        title("all augmentations together"),
        mo.md(f"Each tile is one sample, cameras {', '.join(CAMERAS)} from top to bottom. "
              f"JPEG q{JPEG_QUALITY} here to stay below marimo's output size; the tiles above are lossless."),
        field(controls["clean_probability"]), seed, samples, random_frames,
    ], _tiles)
    return


@app.cell
def _(settings):
    _changed = changed(settings)
    if _changed:
        OUTPUT.write_text(yaml_text(settings))
        _status = f"Written to `{OUTPUT}` at {datetime.now():%H:%M:%S}, changed: {', '.join(_changed)}."
    else:
        _status = f"All settings are the defaults: `{OUTPUT}` is not written."
    _rates = "\n".join(
        f"| {name} | {section['probability']:.0%} | {(1 - settings['clean_probability']) * section['probability']:.1%} |"
        for name, section in nested(settings).items()
    )
    mo.vstack([
        mo.md("## Selected settings"),
        mo.callout(mo.md(_status), kind="success" if _changed else "info"),
        mo.md(f"| transform | probability | share of all samples |\n|---|---|---|\n{_rates}"),
        mo.md(f"```yaml\n{yaml_text(settings)}```"),
        mo.md("The same changes as overrides of `scripts/train.py`:"),
        mo.md(f"```bash\n{' '.join(overrides(settings)) or '# no change from the training config'}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
