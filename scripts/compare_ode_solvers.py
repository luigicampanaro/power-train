"""Compare ODE solvers and step counts for sampling actions (flow matching: noise at tau = 0 -> actions at tau = 1),
on the val and test episodes of a finished run. torchdiffeq is not a project dependency; pin torch to the project's
build, or uv pulls the latest torch for torchdiffeq, which breaks the project's torchvision:

    uv run --with torchdiffeq==0.2.5 --with matplotlib --with torch==2.7.1+cu128 \
        --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
        python scripts/compare_ode_solvers.py outputs/<run>

For every sample, the observations are encoded once and every solver starts from the same noise: the differences
come from the solvers only. float32, no autocast, so the solvers' own error is not hidden under bf16 rounding;
"euler 10 bf16" is the sampling of training and validation (bf16 autocast), for reference.
Reported, in raw units: the error of the whole chunk and of its first action against the recorded actions, the
distance to the exact ODE solution (dopri5 at rtol = atol = 1e-6), the network evaluations (NFE) per chunk, and the
latency of one chunk at batch 1 (the robot's case), observations encoded included.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchdiffeq import odeint

from diff_style.datamodules.episode_datamodule import EpisodeDataModule
from diff_style.datasets.episode_dataset import EpisodeDataset, list_episodes
from diff_style.models.preprocess import preprocess_images
from diff_style.policies.flow_matching import FlowMatchingPolicy, euler
from plot_val_rollout import best_checkpoint, check_never_trained_on

# name: (torchdiffeq method, fixed steps or None for adaptive, rtol/atol of the adaptive ones)
SOLVERS = {
    **{f"euler {n}": ("euler", n, None) for n in [5, 10, 20, 50, 100]},
    **{f"midpoint {n}": ("midpoint", n, None) for n in [5, 10, 20]},
    **{f"rk4 {n}": ("rk4", n, None) for n in [5, 10, 20]},
    "dopri5 1e-3": ("dopri5", None, 1e-3),
    "exact (dopri5 1e-6)": ("dopri5", None, 1e-6),
}


class Velocity(torch.nn.Module):
    """dx/dtau = net(x, tau, obs_tokens), counting its evaluations."""

    def __init__(self, policy: FlowMatchingPolicy, obs_tokens: torch.Tensor) -> None:
        super().__init__()
        self.policy, self.obs_tokens, self.nfe = policy, obs_tokens, 0

    def forward(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.policy.net(x, tau.expand(len(x)).to(x.dtype), self.obs_tokens).to(x.dtype)


def solve(policy: FlowMatchingPolicy, obs_tokens: torch.Tensor, noise: torch.Tensor, solver: str):
    """Normalized actions (B, H, D) at tau = 1, and the NFE."""
    method, steps, tol = SOLVERS[solver]
    velocity = Velocity(policy, obs_tokens)
    times = torch.tensor([0.0, 1.0], device=noise.device)
    if steps is None:
        x = odeint(velocity, noise, times, method=method, rtol=tol, atol=tol)[-1]
    else:
        x = odeint(velocity, noise, times, method=method, options={"step_size": 1.0 / steps})[-1]
    return x, velocity.nfe


@torch.no_grad()
def sample_all(policy, ds: EpisodeDataset, image_size, batch_size: int, generator: torch.Generator) -> dict:
    """{solver: (T, H, D) raw-unit chunks} for every step of the episode, plus "euler 10 bf16"; and NFE per solver."""
    out, nfes = {s: [] for s in [*SOLVERS, "euler 10 bf16"]}, {s: [] for s in SOLVERS}
    H, D = policy.hparams["action_horizon"], policy.hparams["action_dim"]
    for start in range(0, len(ds), batch_size):
        samples = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        batch = {k: torch.stack([s[k] for s in samples]).cuda() for k in ["images", "qpos", "past_actions"]}
        batch["images"] = preprocess_images(batch["images"], image_size)
        noise = torch.randn(len(samples), H, D, generator=generator).cuda()
        obs_tokens = policy.encode_observations(batch)
        for solver in SOLVERS:
            x, nfe = solve(policy, obs_tokens, noise, solver)
            out[solver].append(policy.action_normalizer.denormalize(x).cpu())
            nfes[solver].append(nfe)
            if solver == "euler 10":  # torchdiffeq's Euler is the policy's sampler (float32)
                ours = euler(lambda x, tau: policy.net(x, tau, obs_tokens), noise, 10)
                assert torch.allclose(x, ours, atol=1e-4), (x - ours).abs().max()
        with torch.autocast("cuda", dtype=torch.bfloat16):  # as in validation: bf16, the same noise
            tokens = policy.encode_observations(batch)
            x = euler(lambda x, tau: policy.net(x, tau, tokens), noise.to(tokens.dtype), 10)
        out["euler 10 bf16"].append(policy.action_normalizer.denormalize(x.float()).cpu())
    return {s: torch.cat(v).numpy() for s, v in out.items()}, nfes


@torch.no_grad()
def latency_ms(policy, ds: EpisodeDataset, image_size, solver: str, repeats: int = 20) -> float:
    """One chunk at batch 1, from the preprocessed frames: encode + solve."""
    sample = ds[len(ds) // 2]
    batch = {k: sample[k][None].cuda() for k in ["images", "qpos", "past_actions"]}
    batch["images"] = preprocess_images(batch["images"], image_size)
    noise = torch.randn(1, policy.hparams["action_horizon"], policy.hparams["action_dim"], device="cuda")
    times = []
    for i in range(repeats + 3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        solve(policy, policy.encode_observations(batch), noise, solver)
        torch.cuda.synchronize()
        if i >= 3:  # warmup
            times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best", help="best, last, or a file in <run>/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra/config.yaml")
    dataset_dir = Path(cfg.datamodule.dataset_dir)
    paths = list_episodes(dataset_dir)
    split = EpisodeDataModule(dataset_dir, args.run_dir / "unused.pt", train_fraction=cfg.datamodule.train_fraction,
                              val_fraction=cfg.datamodule.val_fraction, test_fraction=cfg.datamodule.test_fraction,
                              seed=cfg.datamodule.seed).split_episodes(len(paths))
    ckpt = best_checkpoint(args.run_dir) if args.checkpoint == "best" else \
        args.run_dir / "checkpoints" / ("last.ckpt" if args.checkpoint == "last" else args.checkpoint)
    policy = FlowMatchingPolicy.load_from_checkpoint(ckpt, map_location="cuda").eval()
    groups = {"val": split["val"].tolist(), "test": split["test"].tolist()}
    check_never_trained_on(cfg, split, dataset_dir, groups["val"] + groups["test"], policy, allowed=("val", "test"))

    generator = torch.Generator().manual_seed(args.seed)
    results = {g: {} for g in groups}  # group -> solver -> list of (chunk error, first error, distance to exact)
    nfe_all = {s: [] for s in SOLVERS}
    for g, episodes in groups.items():
        for n, ep in enumerate(episodes):
            ds = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon, episodes=[ep])
            chunks, nfes = sample_all(policy, ds, list(cfg.image_size), 64, generator)
            truth = ds.actions[0].numpy()
            T, H, _ = chunks["euler 10"].shape
            # recorded actions t ... t + H - 1 of every chunk, the last one repeated after the episode's end (as the
            # dataset's windows); a (T, H) index
            target = truth[np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)]
            exact = chunks["exact (dopri5 1e-6)"]
            for solver, pred in chunks.items():
                results[g].setdefault(solver, []).append(
                    (np.abs(pred - target).reshape(-1), np.abs(pred[:, 0] - truth).reshape(-1),
                     np.abs(pred - exact).reshape(-1)))
            for solver in SOLVERS:
                nfe_all[solver] += nfes[solver]
            print(f"\r{g}: {n + 1}/{len(episodes)} episodes", end="", flush=True)
        print()

    ds = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon,
                        episodes=groups["val"][:1])
    print(f"\n{ckpt.name}, {len(groups['val'])} val + {len(groups['test'])} test episodes. Errors: mean absolute, "
          f"raw units, all joints")
    print(f"{'solver':<20} {'NFE':>5} {'ms/chunk':>9} | {'val chunk':>9} {'val 1st':>8} | {'test chunk':>10} "
          f"{'test 1st':>9} | {'to exact':>9}")
    for solver in [*SOLVERS, "euler 10 bf16"]:
        nfe = np.mean(nfe_all[solver]) if solver in SOLVERS else 10
        ms = latency_ms(policy, ds, list(cfg.image_size), solver) if solver in SOLVERS else float("nan")
        row = []
        for g in groups:
            chunk, first, to_exact = (np.concatenate(x).mean() for x in zip(*results[g][solver]))
            row.append((chunk, first, to_exact))
        to_exact = np.mean([r[2] for r in row])
        print(f"{solver:<20} {nfe:5.0f} {ms:9.1f} | {row[0][0]:9.4f} {row[0][1]:8.4f} | {row[1][0]:10.4f} "
              f"{row[1][1]:9.4f} | {to_exact:9.5f}")


if __name__ == "__main__":
    main()
