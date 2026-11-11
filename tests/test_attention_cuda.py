"""CUDA-only: the attention kernels. Every layer and model on CUDA: tests/test_gradients_and_cuda.py."""

from functools import partial

import pytest
import torch

from diff_style.models.layers import Attention, math_attention, sdpa

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
              # torch explains each kernel it skips; these tests skip kernels on purpose
              pytest.mark.filterwarnings("ignore:.*(kernel not used|runtime disabled|not supported)")]


def test_no_mask_runs_on_the_flash_kernel_and_a_mask_cannot():
    """layers/attention.py promises mask=None keeps the flash kernels: with only FlashAttention allowed,
    the unmasked call runs and the masked one (even all True) finds no kernel."""
    torch.manual_seed(0)
    attention = Attention(dim=384, num_heads=6, attention_fn=partial(sdpa, kernels=["FLASH_ATTENTION"]))
    attention = attention.cuda().bfloat16()  # head_dim 64
    x = torch.randn(2, 16, 384, device="cuda", dtype=torch.bfloat16)
    flash = attention(x)
    all_true = torch.ones(16, 16, dtype=torch.bool, device="cuda")
    with pytest.raises(RuntimeError, match="No available kernel"), \
            pytest.warns(UserWarning, match="Flash Attention does not support non-null attn_mask"):
        attention(x, mask=all_true)
    attention.attention_fn = math_attention
    assert torch.allclose(flash.float(), attention(x).float(), atol=2e-2)


@pytest.mark.parametrize("kernel, masked", [("FLASH_ATTENTION", False), ("EFFICIENT_ATTENTION", False),
                                           ("EFFICIENT_ATTENTION", True), ("CUDNN_ATTENTION", False)])
def test_gpu_kernels_match_the_written_out_softmax(kernel, masked):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, length, 64, device="cuda", dtype=torch.bfloat16) for length in [16, 68, 68])
    mask = None
    if masked:
        mask = torch.rand(2, 1, 16, 68, device="cuda") > 0.5
        mask[..., 0] = True
    expected = math_attention(q.float(), k.float(), v.float(), mask)
    try:
        out = sdpa(q, k, v, mask, kernels=[kernel])
    except RuntimeError as error:  # cuDNN attention needs a recent GPU and cuDNN build
        pytest.skip(f"{kernel} can't run here: {str(error)[:80]}")
    assert torch.allclose(out.float(), expected, atol=2e-2)


def test_flash_kernel_runs_causal_scaled_attention():
    """Neither needs a mask tensor, so the flash kernel takes both."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 16, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = sdpa(q, k, v, is_causal=True, scale=0.05, kernels=["FLASH_ATTENTION"])
    expected = math_attention(q.float(), k.float(), v.float(), is_causal=True, scale=0.05)
    assert torch.allclose(out.float(), expected, atol=2e-2)


@pytest.mark.parametrize("kernel", ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"])
def test_gpu_kernels_apply_dropout_without_changing_the_mean(kernel):
    """The fast kernels support attention dropout (flash too, unlike a mask): one draw differs from
    the output without dropout, the mean over many draws is the same."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 6, length, 64, device="cuda", dtype=torch.bfloat16) for length in [16, 68, 68])
    draws = sdpa(*(t.expand(2000, -1, -1, -1) for t in (q, k, v)), dropout_p=0.5, kernels=[kernel]).float()
    exact = math_attention(q.float(), k.float(), v.float())[0]
    assert not torch.allclose(draws[0], exact, atol=5e-2)
    assert torch.allclose(draws.mean(0), exact, atol=5e-2)

