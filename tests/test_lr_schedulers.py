import torch

from diff_style.utils.lr_schedulers import warmup_cosine
from references import REFERENCES


def test_warmup_cosine_matches_hugging_face():
    """Hugging Face's get_cosine_schedule_with_warmup: the lr factor at every step, warmup included."""
    ref = REFERENCES["hf_warmup_cosine"]
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = warmup_cosine(optimizer, warmup_steps=ref["warmup_steps"], total_steps=ref["total_steps"])
    factors = []
    for _ in range(ref["total_steps"]):
        factors.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert torch.allclose(torch.tensor(factors), ref["factors"], atol=1e-7)


def test_warmup_cosine_stays_at_zero_after_the_last_step():
    """Hugging Face's cosine rises again after total_steps; ours stays at 0 if training runs longer."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = warmup_cosine(optimizer, warmup_steps=2, total_steps=10)
    for _ in range(15):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.0
