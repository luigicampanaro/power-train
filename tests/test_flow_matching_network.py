"""FlowMatchingNetwork: the token layout, the source and time embeddings, the input and vision-encoder checks, the
init, and the velocity's dependence on each input. A tiny stand-in replaces DINOv3, so these tests run without it.
The policy around the network (flow path, sampler, training, configs) is tested in tests/test_flow_matching.py."""

import warnings

import pytest
import torch
from torch import nn

from diff_style.models.flow_matching_network import FlowMatchingNetwork
from diff_style.models.layers import INIT_STD, AdaLNModulation
from diff_style.models.perceiver import Perceiver
from diff_style.models.transformer import Transformer
from diff_style.models.vision import VisionEncoder
from test_vision_encoder import FrameMergingEncoder

C, T1, T2, H, K, DIM = 2, 3, 2, 4, 5, 32  # cameras, 1D / image history, action horizon, tokens per image, width


class PixelEncoder(VisionEncoder):
    """Vision encoder stand-in: one token per pixel, (N, T, 3, H, W) -> (N, T, H*W, dim)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(dim)
        self.proj = nn.Linear(3, dim)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.proj(frames.flatten(-2).transpose(-1, -2))


class ZeroPerceiver(nn.Module):
    """Perceiver stand-in: K zero tokens per image, so only the embeddings are left."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(x), K, DIM)


def network_parts() -> dict:
    """The arguments of a small FlowMatchingNetwork, without dropout."""
    return {
        "vision_encoder": PixelEncoder(),
        "perceiver": Perceiver(input_dim=8, num_latents=K, dim=DIM, num_heads=4),
        "obs_encoder": Transformer(DIM, depth=1, num_heads=4),
        "action_decoder": Transformer(DIM, depth=2, num_heads=4, cross_attention=True, cond_dim=DIM),
        "num_cameras": C, "qpos_dim": 7, "action_dim": 6, "obs_horizon_1d": T1, "obs_horizon_2d": T2,
        "action_horizon": H,
    }


def make_network(seed=0, **dropouts):
    """A small network; dropouts: FlowMatchingNetwork's camera_dropout, embedding_dropout, ..."""
    torch.manual_seed(seed)
    return FlowMatchingNetwork(**network_parts(), **dropouts)


def make_batch(b=3):
    return {"images": torch.rand(b, T2, C, 3, 4, 4), "qpos": torch.randn(b, T1, 7),
            "past_actions": torch.randn(b, T1, 6), "actions": torch.randn(b, H, 6)}


def test_shapes():
    net, batch = make_network(), make_batch()
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    assert obs.shape == (3, C * T2 * K + 2 * T1, DIM)
    assert net(batch["actions"], torch.rand(3), obs).shape == (3, H, 6)


def test_a_video_encoder_that_merges_frames_runs_with_a_warning():
    """T_out = T2 / 2 = 1 frame per camera: K tokens per camera. The shapes are correct, but the time offsets of such
    an encoder are not supported yet (see VisionEncoder), so the network warns."""
    torch.manual_seed(0)
    net = FlowMatchingNetwork(**network_parts() | {"vision_encoder": FrameMergingEncoder()})
    batch = make_batch()
    with pytest.warns(UserWarning, match="video encoders that merge frames are not supported yet"):
        obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    assert obs.shape == (3, C * (T2 // 2) * K + 2 * T1, DIM)
    assert net(batch["actions"], torch.rand(3), obs).shape == (3, H, 6)


def test_an_image_encoder_gives_no_warning():
    """One output per frame (T_out = T2): the time offsets are correct."""
    net, batch = make_network(), make_batch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])


def test_the_vision_encoder_must_be_a_vision_encoder():
    """A plain nn.Module does not promise the contract (frames -> tokens, dim): refused when the network is built."""
    with pytest.raises(TypeError, match="VisionEncoder"):
        FlowMatchingNetwork(**network_parts() | {"vision_encoder": nn.Linear(3, 8)})


def test_the_vision_encoder_width_must_be_the_perceiver_input():
    with pytest.raises(ValueError, match="16-dim tokens, the Perceiver takes 8"):
        FlowMatchingNetwork(**network_parts() | {"vision_encoder": PixelEncoder(dim=16)})


def test_qpos_and_past_actions_must_have_the_same_number_of_steps():
    """past_actions[:, k] pairs with qpos[:, k]. One past action would broadcast against the time offsets of T1 steps,
    without an error."""
    net, batch = make_network(), make_batch()
    with pytest.raises(ValueError, match="same number of steps"):
        net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"][:, :1])


def test_embedding_tables_start_at_trunc_normal_init_std(trunc_normal_stds):
    """source_embedding and time_embedding from trunc_normal(INIT_STD), as the Perceiver's latents and ViT's pos_embed:
    small tags on the content tokens, not nn.Embedding's N(0, 1)."""
    net = make_network()
    for table in (net.source_embedding, net.time_embedding):
        assert trunc_normal_stds.get(id(table)) == INIT_STD


def test_embedding_tables_are_plain_parameters():
    """Used as tensors (all of the sources, a range of time offsets), not looked up as a vocabulary: one row per
    source (C cameras, qpos, past actions), and one per time offset (-max_past .. H - 1)."""
    state = make_network().state_dict()
    assert state["source_embedding"].shape == (C + 2, DIM)
    assert state["time_embedding"].shape == (max(T1, T2 - 1) + H, DIM)


@pytest.mark.parametrize("start, stop", [(-max(T1, T2 - 1) - 1, 0), (0, H + 1)])
def test_time_offsets_outside_the_table_raise(start, stop):
    """A slice would wrap a negative start to the end of the table, and cut a stop past it, without an error."""
    with pytest.raises(ValueError, match="outside the time table"):
        make_network().time(start, stop)


def test_input_projections_start_at_the_scale_of_the_embeddings():
    """qpos_proj, past_action_proj and action_proj from trunc_normal(0.02) with biases 0, as the embeddings
    trunc_normal(0.02) and the Perceiver: with PyTorch's default (std 1 / sqrt(3 x 7)) the qpos tokens start ~17x the
    image tokens."""
    net = make_network()
    projections = [net.qpos_proj, net.past_action_proj, net.action_proj]
    weights = torch.cat([proj.weight.flatten() for proj in projections])  # 608 values: the std within 3%
    assert weights.std().item() == pytest.approx(0.02, rel=0.15)
    assert not any(proj.bias.any() for proj in projections)


def test_every_token_gets_its_source_and_time_offset():
    """With the content zeroed, the tokens are just their embeddings. Channel 0 of the time table
    holds the row (offset + max_past), channel 1 of the source table holds the source id."""
    net = make_network()
    net.obs_encoder = nn.Identity()
    net.perceiver = ZeroPerceiver()
    with torch.no_grad():
        for proj in [net.qpos_proj, net.past_action_proj]:
            proj.weight.zero_()
            proj.bias.zero_()
        net.time_embedding.zero_()
        net.time_embedding[:, 0] = torch.arange(len(net.time_embedding))
        net.source_embedding.zero_()
        net.source_embedding[:, 1] = torch.arange(C + 2)
    batch = make_batch()
    tokens = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])[0]
    offsets = (tokens[:, 0] - net.max_past).long().tolist()
    sources = tokens[:, 1].long().tolist()
    expected = ([(c, t - (T2 - 1)) for c in range(C) for t in range(T2) for _ in range(K)]  # images: camera-major
                + [(C, t - (T1 - 1)) for t in range(T1)]  # qpos: offsets -(T1-1) .. 0
                + [(C + 1, t - T1) for t in range(T1)])  # past actions: one step earlier, -T1 .. -1
    assert list(zip(sources, offsets)) == expected


def tokens_one_at_a_time(net: FlowMatchingNetwork, batch: dict) -> torch.Tensor:
    """The obs tokens before the obs encoder, built without a reshape of the batch: each camera's video through the
    encoder alone, each image through the Perceiver alone, each token's tags read from the tables by row. The offsets
    come from the dataset (EpisodeDataset): images[:, i] and qpos[:, i] are the steps t - T + 1 + i, past_actions[:, i]
    is the action of step t - T1 + i (one step before qpos[:, i])."""
    images, qpos, past_actions = batch["images"], batch["qpos"], batch["past_actions"]
    source, time = net.source_embedding, lambda offset: net.time_embedding[net.max_past + offset]
    samples = []
    for s in range(len(images)):
        tokens = []
        for c in range(C):
            outputs = net.vision_encoder(images[s, :, c][None])[0]  # (T_out, L, vision dim): camera c's video alone
            for i, features in enumerate(outputs):  # output i is at step t - T_out + 1 + i
                tokens.append(net.perceiver(features[None])[0] + source[c] + time(i - len(outputs) + 1))
        for i in range(T1):
            tokens.append((net.qpos_proj(qpos[s, i]) + source[C] + time(i - T1 + 1))[None])
        for i in range(T1):
            tokens.append((net.past_action_proj(past_actions[s, i]) + source[C + 1] + time(i - T1))[None])
        samples.append(torch.cat(tokens))
    return torch.stack(samples)


@pytest.mark.filterwarnings("ignore:the vision encoder gave")  # the video case: tested in its own test
@pytest.mark.parametrize("vision_encoder", [PixelEncoder, FrameMergingEncoder], ids=["image", "video"])
def test_obs_tokens_are_each_signal_plus_its_own_source_and_time(vision_encoder):
    """Every token against tokens_one_at_a_time: the right image, qpos or action, with its own camera or source and
    its own step, in the order camera-major, then qpos, then past actions. In the video case (T_out = 1), this checks
    the content and the order of the tokens: the time offsets follow the same convention, which is correct only for
    one output per frame (see VisionEncoder)."""
    torch.manual_seed(0)
    net = FlowMatchingNetwork(**network_parts() | {"vision_encoder": vision_encoder()}).eval()
    net.obs_encoder = nn.Identity()
    batch = make_batch()
    tokens = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    torch.testing.assert_close(tokens, tokens_one_at_a_time(net, batch))


def test_each_image_reaches_only_its_own_tokens():
    """Changing camera 1 at frame 0 changes only the K tokens of (camera 1, frame 0), in its own sample."""
    net = make_network()
    net.obs_encoder = nn.Identity()  # no mixing between tokens after the Perceiver
    batch = make_batch()
    before = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    batch["images"][1, 0, 1] = torch.rand(3, 4, 4)
    after = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    changed = (after - before).abs().amax(-1) > 1e-6  # (B, tokens)
    first = (1 * T2 + 0) * K  # camera 1, frame 0
    expected = torch.zeros_like(changed)
    expected[1, first:first + K] = True
    assert torch.equal(changed, expected)


def test_samples_are_independent():
    net = make_network()
    torch.nn.init.normal_(net.velocity_head.weight)  # out of the zero init
    batch, tau = make_batch(), torch.rand(3)
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    velocity = net(batch["actions"], tau, obs)
    batch["images"][0], batch["qpos"][0], batch["actions"][0] = 0, 0, 0
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    assert torch.allclose(net(batch["actions"], tau, obs)[1:], velocity[1:], atol=1e-5)


class RecordingBackend:
    """A torch.compile backend that runs each graph as traced and records the tensors that each graph receives. Torch
    2.5 gives a compiled module's parameters to its graph as inputs: these are the parameters that run compiled."""

    def __init__(self) -> None:
        self.tensor_ids = set()

    def __call__(self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.tensor_ids.update(id(tensor) for tensor in example_inputs)
        return graph_module.forward

    def eager_parameters(self, module: nn.Module) -> set[str]:
        return {name for name, p in module.named_parameters() if id(p) not in self.tensor_ids}


# The glue of encode_observations, outside every module that it calls: eager, a few small operations
EAGER_GLUE = {"source_embedding", "qpos_proj.weight", "qpos_proj.bias", "past_action_proj.weight",
              "past_action_proj.bias"}


def test_compile_covers_forward_and_the_modules_of_encode_observations():
    """Module.compile covers __call__ only, and encode_observations is a plain method: net.compile() also compiles each
    module that encode_observations calls. The action decoder is in the graph of forward."""
    torch._dynamo.reset()  # a new cache, so no earlier test uses up the recompile limit
    net, batch, backend = make_network(), make_batch(), RecordingBackend()
    net.vision_encoder.requires_grad_(False)  # frozen modules too
    net.compile(backend=backend)
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    net(batch["actions"], torch.rand(3), obs)
    assert backend.eager_parameters(net) == EAGER_GLUE


def test_velocity_depends_on_the_flow_time():
    net = make_network()
    torch.nn.init.normal_(net.velocity_head.weight)
    for module in net.action_decoder.modules():  # out of the adaLN zero init, so tau reaches the actions
        if isinstance(module, AdaLNModulation):
            torch.nn.init.normal_(module.linear.weight, std=0.1)
    batch = make_batch()
    obs = net.encode_observations(batch["images"], batch["qpos"], batch["past_actions"])
    early, late = net(batch["actions"], torch.full((3,), 0.1), obs), net(batch["actions"], torch.full((3,), 0.9), obs)
    assert not torch.allclose(early, late, atol=1e-3)
