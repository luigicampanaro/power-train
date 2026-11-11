"""FlowMatchingPolicy.predict_actions for one sample, in one CUDA graph: the preprocessing of the frames, the
observation encoder and every Euler step of the sampler, replayed as one launch.

At batch size 1 the GPU work is thousands of small kernels (about 7000 for 10 Euler steps), and launching them from
Python takes longer than running them. A CUDA graph launches them all at once. compile_for_inference first fuses them
into fewer kernels. On an RTX 4070 Ti SUPER, with the train.yaml model (DINOv3 ViT-S+/16, 2 cameras x 2 frames,
10 Euler steps): about 30 ms eager, 13 ms graphed, 5 ms compiled (with freezing) and graphed.

The computation is the policy's validation in bf16-mixed training: the datamodule's preprocessing in float32, then
predict_actions under bf16 autocast. The graph draws new noise on every call. It is captured on the first call, for
the shapes of that call's inputs: later inputs must have the same shapes (the same cameras, frame size and horizons).
"""

import torch
from torch._inductor import config as inductor_config

from diff_style.models.preprocess import preprocess_images
from diff_style.policies.flow_matching import FlowMatchingPolicy

INPUT_KEYS = ("images", "qpos", "past_actions")
WARMUP_CALLS = 3  # before the capture: compile, the cuBLAS workspaces and the autotuning run outside the graph


def compile_for_inference(policy: FlowMatchingPolicy) -> None:
    """torch.compile the policy's network, with Inductor's freezing: the weights become constants, so their casts to
    bf16 under autocast are done once, at compile time, and not again at every Euler step (graphed: 8 ms -> 5 ms).
    The weights must not change after the first call, which compiles (20 s to a few minutes, DINOv3 included).

    CAUTION: freezing is set for the whole process, for every module compiled after this call. In torch 2.5 the
    per-module option, policy.compile(options={"freezing": True}), compiles without freezing (measured: 8 ms)."""
    inductor_config.freezing = True
    policy.compile()


class GraphedPolicy:
    def __init__(self, policy: FlowMatchingPolicy, image_size: tuple[int, int]) -> None:
        """policy: on a CUDA device, in eval mode, compiled or not. image_size: (H, W) of the model input, the
        datamodule's (PolicySpec.image_size)."""
        self.policy = policy
        self.image_size = tuple(image_size)
        self.graph: torch.cuda.CUDAGraph | None = None

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """batch: one sample, as ObservationHistory.batch gives it: images (1, T2, C, 3, H, W) uint8, qpos (1, T1, Dq),
        past_actions (1, T1, Da), raw units, on any device. Returns the next action_horizon actions (H, Da), raw
        units, float32, on the policy's device: a new tensor on each call."""
        if self.graph is None:
            self.graph = self._capture(batch)
        self._copy_inputs(batch)
        self.graph.replay()
        return self.actions.clone()

    def _capture(self, batch: dict[str, torch.Tensor]) -> torch.cuda.CUDAGraph:
        """Static input buffers of the shapes of batch, then the graph of _predict on them. Its output, self.actions,
        is a static buffer too: each replay writes into it. The warm-up and the capture draw noise: the random state
        is put back after them, so the first call draws the noise an eager call would."""
        self.inputs = {key: torch.empty_like(batch[key], device=self.policy.device) for key in INPUT_KEYS}
        self._copy_inputs(batch)
        random_state = torch.cuda.get_rng_state(self.policy.device)
        graph = self._warm_up_and_record()
        torch.cuda.set_rng_state(random_state, self.policy.device)
        return graph

    def _warm_up_and_record(self) -> torch.cuda.CUDAGraph:
        side_stream = torch.cuda.Stream(self.policy.device)  # torch's recipe: the warm-up runs on a side stream
        side_stream.wait_stream(torch.cuda.current_stream(self.policy.device))
        with torch.cuda.stream(side_stream):
            for _ in range(WARMUP_CALLS):
                self._predict()
        torch.cuda.current_stream(self.policy.device).wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.actions = self._predict()
        return graph

    def _copy_inputs(self, batch: dict[str, torch.Tensor]) -> None:
        for key, buffer in self.inputs.items():
            if batch[key].shape != buffer.shape:  # copy_ would broadcast some shapes without an error
                raise ValueError(f"{key}: the graph was captured for shape {tuple(buffer.shape)}, got "
                                 f"{tuple(batch[key].shape)}")
            buffer.copy_(batch[key])

    def _predict(self) -> torch.Tensor:
        images = preprocess_images(self.inputs["images"], self.image_size)  # float32, as the datamodule's
        with torch.autocast(self.policy.device.type, dtype=torch.bfloat16):
            actions = self.policy.predict_actions({"images": images, "qpos": self.inputs["qpos"],
                                                   "past_actions": self.inputs["past_actions"]})
        return actions[0].float()
