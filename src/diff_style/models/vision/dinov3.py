"""DINOv3 (Siméoni et al. 2025), the Hugging Face transformers model (DINOv3ViTModel), loaded from the local
Hugging Face cache. The weights are gated: request access on the model's page, then download them."""

import torch
from transformers import DINOv3ViTModel

from diff_style.models.vision.encoder import VisionEncoder

# DINOv3's web-image (LVD-1689M) backbones are trained on ImageNet-normalized images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv3Encoder(VisionEncoder):
    """Frames (N, T, 3, H, W), float in [0, 1] -> patch tokens (N, T, H/16 * W/16, dim), each frame on its own.

    Returns the final-norm patch tokens; the class token and the register tokens before them are dropped. The patch
    tokens already carry their position (DINOv3 uses 2D RoPE inside).
    model_id: a Hugging Face DINOv3 model, e.g. facebook/dinov3-vits16plus-pretrain-lvd1689m. Only the local cache
        is read: no download when the policy is built.
    dim: the token width (384 for ViT-S); checked against the loaded model, so configs can read it.
    Trainable, DINOv3 follows the encoder's mode; in training mode it rescales its RoPE patch coordinates at random
    (its pretraining augmentation). The policy freezes it (submodule_lrs): no gradients, eval mode.
    """

    def __init__(self, model_id: str, dim: int) -> None:
        super().__init__(dim)
        self.model = DINOv3ViTModel.from_pretrained(model_id, local_files_only=True)
        if self.model.config.hidden_size != dim:
            raise ValueError(f"{model_id} gives {self.model.config.hidden_size}-dim tokens, the config says dim={dim}")
        self.model.train(self.training)  # from_pretrained returns it in eval mode: the encoder's mode, as any submodule
        self.num_prefix_tokens = 1 + self.model.config.num_register_tokens  # class token, then register tokens
        # Used only by DINOv3's masked pretraining: trainable, it would get no gradient, which stops DDP
        self.model.embeddings.mask_token.requires_grad_(False)
        # Not saved in checkpoints: constants, the same for every run
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(3, 1, 1), persistent=False)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        n, t = frames.shape[:2]
        x = (frames.flatten(0, 1) - self.mean) / self.std  # (N*T, 3, H, W)
        # Frozen: neither the weights nor the images need gradients, so autograd keeps no graph here
        tokens = self.model(pixel_values=x).last_hidden_state[:, self.num_prefix_tokens:]  # (N*T, patches, dim)
        return tokens.unflatten(0, (n, t))
