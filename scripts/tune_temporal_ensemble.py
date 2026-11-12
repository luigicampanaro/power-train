"""Tune the decay m of the temporal ensemble (diff_style/inference/temporal_ensemble.py), for each offset f, on the
val episodes of a finished run; report on its test episodes (used for nothing else: an unbiased error).

    uv run python scripts/tune_temporal_ensemble.py outputs/<run> [--checkpoint best|last|<file>]

Open loop, as scripts/plot_val_rollout.py: the policy predicts a chunk at every step from the recorded observations
(cached in <run>/chunks_<checkpoint>.npz), the ensemble plays them step by step, and its action at t is compared with
the recorded action of t + f. f is a speed choice (the robot is commanded f steps ahead), not an accuracy one: m is
tuned for each f. Error: mean absolute error in raw units, every joint. Smoothness: mean |second difference| of the
sent actions (lower = smoother). "newest only" is no ensemble: the newest chunk's action f.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from diff_style.datamodules.episode_datamodule import EpisodeDataModule
from diff_style.datasets.episode_dataset import EpisodeDataset, list_episodes
from diff_style.inference.temporal_ensemble import TemporalEnsemble
from diff_style.policies.flow_matching import FlowMatchingPolicy
from plot_val_rollout import best_checkpoint, check_never_trained_on, predict_every_step

OFFSETS = [0, 1, 2, 3, 5, 8]
DECAYS = [-0.2, -0.1, -0.05, -0.01, 0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.13, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def ensemble(pred: np.ndarray, offset: int, decay: float) -> np.ndarray:
    """TemporalEnsemble on every step of one episode, vectorized: (T, H, D) chunks -> (T, D) actions."""
    T, H, _ = pred.shape
    window = H - offset
    ages = np.arange(window)
    t, i = np.meshgrid(np.arange(T), ages, indexing="ij")  # (T, window): step, age of the chunk
    exists = t - i >= 0
    actions = pred[np.clip(t - i, 0, None), offset + i]  # (T, window, D): chunk t-i's prediction of t + f
    weights = np.exp(-decay * ages)[None] * exists
    return (weights[..., None] * actions).sum(1) / weights.sum(1, keepdims=True)


def scores(chunks: dict, truths: dict, episodes: list[int], offset: int, decay: float | None) -> tuple[float, float]:
    """(error, smoothness) over the episodes. decay None: newest only."""
    errors, jerks = [], []
    for ep in episodes:
        pred, truth = chunks[ep], truths[ep]
        sent = pred[:, offset] if decay is None else ensemble(pred, offset, decay)
        target = truth[np.minimum(np.arange(len(truth)) + offset, len(truth) - 1)]
        errors.append(np.abs(sent - target).reshape(-1))
        jerks.append(np.abs(np.diff(sent, n=2, axis=0)).reshape(-1))
    return float(np.concatenate(errors).mean()), float(np.concatenate(jerks).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best", help="best, last, or a file in <run>/checkpoints")
    parser.add_argument("--seed", type=int, default=0, help="seed of the sampling noise")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra/config.yaml")
    dataset_dir = Path(cfg.datamodule.dataset_dir)
    paths = list_episodes(dataset_dir)
    split = EpisodeDataModule(dataset_dir, args.run_dir / "unused.pt", train_fraction=cfg.datamodule.train_fraction,
                              val_fraction=cfg.datamodule.val_fraction, test_fraction=cfg.datamodule.test_fraction,
                              seed=cfg.datamodule.seed).split_episodes(len(paths))
    val, test = split["val"].tolist(), split["test"].tolist()
    ckpt = best_checkpoint(args.run_dir) if args.checkpoint == "best" else \
        args.run_dir / "checkpoints" / ("last.ckpt" if args.checkpoint == "last" else args.checkpoint)
    policy = FlowMatchingPolicy.load_from_checkpoint(ckpt, map_location="cuda").eval()
    check_never_trained_on(cfg, split, dataset_dir, val + test, policy, allowed=("val", "test"))  # tuned, reported on

    cache = args.run_dir / f"chunks_{ckpt.stem}_seed{args.seed}.npz"
    truths = {}
    for ep in val + test:
        truths[ep] = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon,
                                    episodes=[ep]).actions[0].numpy()
    if cache.exists():
        chunks = {int(k): v for k, v in np.load(cache).items()}
    else:
        torch.manual_seed(args.seed)
        chunks = {}
        for n, ep in enumerate(val + test):
            ds = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon, episodes=[ep])
            chunks[ep] = predict_every_step(policy, ds, list(cfg.image_size), batch_size=64)
            print(f"\rpredicted {n + 1}/{len(val + test)} episodes", end="", flush=True)
        np.savez(cache, **{str(k): v for k, v in chunks.items()})
        print(f"\nsaved {cache}")

    # The vectorized ensemble is TemporalEnsemble's (tested against ACT, LeRobot, ml/act)
    pred = chunks[val[0]]
    for f, m in [(0, 0.05), (5, 0.13), (3, -0.1)]:
        reference = TemporalEnsemble(pred.shape[1], f, m)
        expected = np.stack([reference(torch.from_numpy(c).double()).numpy() for c in pred])
        assert np.allclose(ensemble(pred, f, m), expected, atol=1e-6), (f, m)

    print(f"\n{ckpt.name}: m tuned on {len(val)} val episodes, reported on {len(test)} test episodes")
    print(f"{'f':>3} {'best m':>7} | {'val err':>8} | {'test err':>8} {'smooth':>7} | {'newest only: test err':>21} "
          f"{'smooth':>7} | {'m=0.05 test err':>15}")
    for f in OFFSETS:
        best = min(DECAYS, key=lambda m: scores(chunks, truths, val, f, m)[0])
        val_err = scores(chunks, truths, val, f, best)[0]
        test_err, test_jerk = scores(chunks, truths, test, f, best)
        new_err, new_jerk = scores(chunks, truths, test, f, None)
        paper_err = scores(chunks, truths, test, f, 0.05)[0]
        print(f"{f:>3} {best:>7} | {val_err:8.4f} | {test_err:8.4f} {test_jerk:7.4f} | {new_err:21.4f} {new_jerk:7.4f} "
              f"| {paper_err:15.4f}")
    print("\nerror on the val episodes for every m (rows: f)")
    print("    m " + " ".join(f"{m:>6}" for m in DECAYS))
    for f in OFFSETS:
        print(f"f={f:<3} " + " ".join(f"{scores(chunks, truths, val, f, m)[0]:6.4f}" for m in DECAYS))


if __name__ == "__main__":
    main()
