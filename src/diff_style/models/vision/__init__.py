"""Vision encoders. Every backbone is a VisionEncoder (encoder.py): it implements encode(frames), (N, T, 3, H, W)
float in [0, 1] -> tokens (N, T_out, L, dim), and gives its token width dim. N is batch x cameras: every camera goes
through the same encoder. An image encoder (DINOv3) encodes each frame on its own, so T_out = T. Each encoder does
its own pixel normalization and gives its tokens their spatial position.

CAUTION: video encoders that merge frames (T_out < T) are not supported yet: the network gives their outputs
incorrect time offsets. See VisionEncoder.
"""

from diff_style.models.vision.encoder import VisionEncoder

__all__ = ["VisionEncoder"]
