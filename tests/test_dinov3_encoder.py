"""DINOv3Encoder against the Hugging Face DINOv3 model it wraps, with the pretrained weights.

Needs the model of configs/model/vision_encoder/dinov3_vits16plus.yaml in the Hugging Face cache; skipped without
it. Its gradcheck and CUDA tests are in tests/test_gradients_and_cuda.py, with every other layer.
"""

import json
from pathlib import Path

import hydra
import pytest
import torch
import torchvision.transforms.v2.functional as TF
from huggingface_hub import try_to_load_from_cache
from omegaconf import OmegaConf

from diff_style.models.vision import VisionEncoder
from diff_style.models.vision.dinov3 import DINOv3Encoder
from references import assert_gradients_match, reference_gradients

CONFIG = OmegaConf.load(Path(__file__).parent.parent / "configs" / "model" / "vision_encoder" /
                        "dinov3_vits16plus.yaml")
DINOV3_CACHED = isinstance(try_to_load_from_cache(CONFIG.model_id, "model.safetensors"), str)
pytestmark = pytest.mark.skipif(not DINOV3_CACHED, reason=f"{CONFIG.model_id} is not in the Hugging Face cache")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PREFIX_TOKENS = 1 + 4  # the class token and the 4 register tokens, before the patch tokens


def make_encoder():
    return DINOv3Encoder(CONFIG.model_id, dim=CONFIG.dim)


def test_tokens_are_the_dinov3_patch_tokens_of_each_normalized_frame():
    """The final-norm tokens of each frame on its own, without the class and register tokens."""
    encoder = make_encoder().eval()
    frames = torch.rand(2, 3, 3, 64, 96)  # (N, T, 3, H, W): 4 x 6 patches of 16 pixels
    tokens = encoder(frames)
    assert tokens.shape == (2, 3, 24, 384)
    for n in range(2):
        for t in range(3):
            image = TF.normalize(frames[n, t], mean=IMAGENET_MEAN, std=IMAGENET_STD)
            expected = encoder.model(pixel_values=image[None]).last_hidden_state[0, PREFIX_TOKENS:]
            assert torch.allclose(tokens[n, t], expected, atol=1e-4)


def test_pixels_are_normalized_as_the_model_card_says():
    """The mean and std of the model's own image processor (preprocessor_config.json in the cache)."""
    with open(try_to_load_from_cache(CONFIG.model_id, "preprocessor_config.json")) as file:
        processor = json.load(file)
    encoder = make_encoder()
    assert encoder.mean.flatten().tolist() == pytest.approx(processor["image_mean"])
    assert encoder.std.flatten().tolist() == pytest.approx(processor["image_std"])


def test_dinov3_follows_the_encoders_mode():
    """from_pretrained returns DINOv3 in eval mode; the encoder puts it in its own mode, as any submodule: a new
    module trains. In training mode only, DINOv3 rescales its RoPE patch coordinates at random (its pretraining
    augmentation): the tokens change from call to call. In eval mode they do not."""
    encoder = make_encoder()
    assert encoder.training and encoder.model.training
    frames = torch.rand(1, 1, 3, 32, 32)
    with torch.no_grad():
        assert not torch.equal(encoder(frames), encoder(frames))
        encoder.eval()
        assert not encoder.model.training
        assert torch.equal(encoder(frames), encoder(frames))
    encoder.train()
    assert encoder.model.training


def test_unfrozen_encoder_trains_with_the_gradients_of_dinov3():
    """Unfrozen, the gradients of the frames and of every weight are those of DINOv3 on the normalized frames."""
    encoder = make_encoder()
    encoder.train()
    assert encoder.model.training
    frames = torch.rand(2, 1, 3, 32, 32, requires_grad=True)
    torch.manual_seed(1)  # the same random RoPE rescale in both: DINOv3 draws it in training mode
    tokens = encoder(frames)
    images = TF.normalize(frames.flatten(0, 1), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    torch.manual_seed(1)
    expected = encoder.model(pixel_values=images).last_hidden_state[:, PREFIX_TOKENS:].unflatten(0, (2, 1))
    assert torch.allclose(tokens, expected, atol=1e-5)
    params = {f"model.{name}": p for name, p in encoder.model.named_parameters() if p.requires_grad}
    ref = reference_gradients(expected, {"frames": frames}, params, torch.randn_like(tokens))
    assert_gradients_match(tokens, ref, {"frames": frames}, encoder)


def test_unfrozen_encoder_trains_every_parameter_that_its_forward_uses():
    """The mask token is used only in DINOv3's masked pretraining: trainable, it would get no gradient, and DDP
    (find_unused_parameters=False, Lightning's default) would stop with an error."""
    encoder = make_encoder()
    encoder(torch.rand(1, 1, 3, 32, 32)).sum().backward()
    trainable = {name: p for name, p in encoder.named_parameters() if p.requires_grad}
    assert [name for name, p in trainable.items() if p.grad is None] == []
    assert set(trainable) == {name for name, _ in encoder.named_parameters()} - {"model.embeddings.mask_token"}


def test_dinov3_encoder_follows_the_vision_contract():
    """A VisionEncoder: the flow-matching network accepts it, and every call is checked against the contract."""
    encoder = make_encoder()
    assert isinstance(encoder, VisionEncoder) and encoder.dim == CONFIG.dim
    with pytest.raises(ValueError, match=r"frames must be \(N, T, 3, H, W\)"):
        encoder(torch.rand(2, 3, 32, 32))  # one image, no time axis


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "configs" / "model" / "vision_encoder")
                                        .glob("*.yaml")), ids=lambda path: path.stem)
def test_every_vision_encoder_config_builds_its_encoder(path):
    """Each file of configs/model/vision_encoder (model/vision_encoder=<file>) builds a VisionEncoder of its dim."""
    config = OmegaConf.load(path)
    if not isinstance(try_to_load_from_cache(config.model_id, "model.safetensors"), str):
        pytest.skip(f"{config.model_id} is not in the Hugging Face cache")
    encoder = hydra.utils.instantiate(config)
    assert isinstance(encoder, VisionEncoder) and encoder.dim == config.dim
    assert encoder(torch.rand(1, 1, 3, 32, 32)).shape == (1, 1, 4, config.dim)


def test_wrong_dim_is_refused():
    with pytest.raises(ValueError, match="384-dim tokens"):
        DINOv3Encoder(CONFIG.model_id, dim=768)
