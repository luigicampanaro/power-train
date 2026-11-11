import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup from 0 to the optimizer's lr over warmup_steps, then cosine decay to 0 at total_steps.
    Stepped once per optimizer step. The same factors as Hugging Face's get_cosine_schedule_with_warmup
    (used by Diffusion Policy), without making transformers a dependency."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)
