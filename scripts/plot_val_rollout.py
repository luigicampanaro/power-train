"""Predicted vs recorded actions, in raw units, on one validation episode of a finished run.

    uv run python scripts/plot_val_rollout.py outputs/<run> [--checkpoint best|last|<file>] [--episode <number>]

Rebuilds the run's train/val/test split from its own saved config (.hydra/config.yaml) and checks that the episode
was never trained on: it must be in the val split, the train split rebuilt here must give the normalization stats that
the checkpoint stores (so it is the split the run trained on), and no train episode may share its source recording
(episodes.yaml) or its action data.

At every timestep t, the policy gets the recorded observations (teacher forcing: open loop, not a closed-loop rollout
on the robot) and predicts the next action_horizon actions. Plots, per joint: the recorded actions, the first
predicted action at every t, and every predicted chunk; also the error. Saves <run>/val_rollout_episode_<n>.png
and prints the errors.
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from diff_style.datamodules.episode_datamodule import EpisodeDataModule
from diff_style.datasets.episode_dataset import EpisodeDataset, list_episodes
from diff_style.models.preprocess import preprocess_images
from diff_style.policies.flow_matching import FlowMatchingPolicy
from diff_style.inference.temporal_ensemble import TemporalEnsemble


def best_checkpoint(run_dir: Path) -> Path:
    """The best checkpoint by val/action_mse, read from the ModelCheckpoint state in last.ckpt."""
    last = torch.load(run_dir / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    for key, state in last["callbacks"].items():
        if "val/action_mse" in key and state.get("best_model_path"):
            path = Path(state["best_model_path"])
            print(f"best checkpoint by val/action_mse: {path.name} ({float(state['best_model_score']):.4f})")
            return run_dir / "checkpoints" / path.name
    raise RuntimeError("no val/action_mse checkpoint state in last.ckpt")


def check_never_trained_on(cfg, split: dict, dataset_dir: Path, episodes: list[int], policy: FlowMatchingPolicy,
                           allowed: tuple[str, ...] = ("val",)) -> None:
    """Each of the episodes (positions in the sorted list) is in one of the `allowed` splits and was never trained on."""
    paths = list_episodes(dataset_dir)
    train = split["train"].tolist()
    allowed_episodes = {ep for name in allowed for ep in split[name].tolist()}
    for episode in episodes:
        assert episode in allowed_episodes and episode not in train, f"position {episode}: not only in {allowed}"

    # 1. The split rebuilt here is the run's: the stats of its train episodes are the ones in the checkpoint
    train_ds = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon, episodes=train)
    stats = EpisodeDataModule.compute_stats(train_ds)
    for key, normalizer in [("qpos", policy.qpos_normalizer), ("actions", policy.action_normalizer)]:
        for name in ["mean", "std"]:
            saved = getattr(normalizer, name).cpu()
            assert torch.allclose(stats[key][name], saved, rtol=0, atol=1e-6), f"{key} {name}: not the run's split"
    print(f"split check: train stats of the rebuilt split == checkpoint stats ({len(train)} train episodes)")

    # 2. No train episode is the same recording under another name, or has the same actions
    sources = yaml.safe_load((dataset_dir / "episodes.yaml").read_text())["episodes"]
    train_sources = {sources[paths[ep].stem] for ep in train}

    def digest(actions: torch.Tensor) -> str:
        return hashlib.sha256(actions.numpy().tobytes()).hexdigest()

    train_digests = {digest(a) for a in train_ds.actions}
    checked = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon, episodes=episodes)
    for episode, actions in zip(episodes, checked.actions):
        source = sources[paths[episode].stem]
        assert source not in train_sources, f"{source} ({paths[episode].name}) is also a train episode"
        assert digest(actions) not in train_digests, f"{paths[episode].name}: same actions as a train episode"
    names = ", ".join(f"{paths[ep].stem} ({sources[paths[ep].stem]})" for ep in episodes[:3])
    print(f"leak check: {len(episodes)} episode(s) [{names}{', ...' if len(episodes) > 3 else ''}] in {'/'.join(allowed)}; "
          f"no train episode has their recording or actions")


@torch.no_grad()
def predict_every_step(policy: FlowMatchingPolicy, ds: EpisodeDataset, image_size, batch_size: int) -> np.ndarray:
    """(T, action_horizon, action_dim) raw-unit predictions, one chunk per timestep of the episode."""
    chunks = []
    for start in range(0, len(ds), batch_size):
        samples = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        batch = {k: torch.stack([s[k] for s in samples]).cuda() for k in ["images", "qpos", "past_actions"]}
        batch["images"] = preprocess_images(batch["images"], image_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):  # the run's precision (bf16-mixed)
            chunks.append(policy.predict_actions(batch).float().cpu())
    return torch.cat(chunks).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best", help="best, last, or a file in <run>/checkpoints")
    parser.add_argument("--episode", type=int, help="episode number (episode_<n>.hdf5); default: the first val one")
    parser.add_argument("--chunk_every", type=int, default=8, help="draw the predicted chunk every this many steps")
    parser.add_argument("--seed", type=int, default=0, help="seed of the sampling noise")
    parser.add_argument("--ensemble_offset", type=int, default=None,
                        help="also plot the temporal ensemble (inference/temporal_ensemble.py) with this offset f: "
                             "0 = ACT's, > 0 = proleptic (PTE). Off by default")
    parser.add_argument("--ensemble_decay", type=float, default=0.05, help="m of the ensemble weights exp(-m * i)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra/config.yaml")
    dataset_dir = Path(cfg.datamodule.dataset_dir)
    paths = list_episodes(dataset_dir)
    datamodule = EpisodeDataModule(dataset_dir, args.run_dir / "unused.pt", train_fraction=cfg.datamodule.train_fraction,
                                   val_fraction=cfg.datamodule.val_fraction, test_fraction=cfg.datamodule.test_fraction,
                                   seed=cfg.datamodule.seed)
    split = datamodule.split_episodes(len(paths))
    numbers = [int(p.stem.removeprefix("episode_")) for p in paths]
    episode = split["val"][0] if args.episode is None else numbers.index(args.episode)  # position in the sorted list

    if args.checkpoint == "best":
        ckpt = best_checkpoint(args.run_dir)
    else:
        ckpt = args.run_dir / "checkpoints" / ("last.ckpt" if args.checkpoint == "last" else args.checkpoint)
    policy = FlowMatchingPolicy.load_from_checkpoint(ckpt, map_location="cuda").eval()  # the EMA weights

    check_never_trained_on(cfg, split, dataset_dir, [episode], policy)

    ds = EpisodeDataset(dataset_dir, cfg.obs_horizon_1d, cfg.obs_horizon_2d, cfg.action_horizon, episodes=[episode])
    torch.manual_seed(args.seed)
    pred = predict_every_step(policy, ds, list(cfg.image_size), batch_size=64)  # (T, H, D)
    truth = ds.actions[0].numpy()  # (T, D)
    T, H, D = pred.shape

    # Errors. First action: pred[t, 0] vs truth[t]. Whole chunk: pred[t, h] vs truth[t + h], within the episode
    first_err = pred[:, 0] - truth
    chunk_err = np.concatenate([pred[t, :min(H, T - t)] - truth[t:t + H] for t in range(T)])
    std = policy.action_normalizer.std.cpu().numpy()
    print(f"\n{paths[episode].name}: {T} steps, checkpoint {ckpt.name}, open loop on the recorded observations")
    print(f"{'joint':>6} {'range':>8} {'MAE 1st':>9} {'RMSE 1st':>9} {'max 1st':>9} {'MAE chunk':>10} "
          f"{'RMSE chunk':>11} {'MAE/std':>8}")
    for j in range(D):
        print(f"{j:>6} {np.ptp(truth[:, j]):8.4f} {np.abs(first_err[:, j]).mean():9.4f} "
              f"{np.sqrt((first_err[:, j] ** 2).mean()):9.4f} {np.abs(first_err[:, j]).max():9.4f} "
              f"{np.abs(chunk_err[:, j]).mean():10.4f} {np.sqrt((chunk_err[:, j] ** 2).mean()):11.4f} "
              f"{np.abs(first_err[:, j]).mean() / std[j]:8.3f}")
    print(f"all joints: MAE first action {np.abs(first_err).mean():.4f}, MAE whole chunk {np.abs(chunk_err).mean():.4f}, "
          f"normalized MSE whole chunk {((chunk_err / std) ** 2).mean():.4f} (val/action_mse-like)")

    ens = ens_err = None
    if args.ensemble_offset is not None:
        # Played step by step, as on the robot. Its action at t targets the recorded action of t + f (the last one
        # after the episode's end): compared and drawn there
        f = args.ensemble_offset
        ensemble = TemporalEnsemble(H, offset=f, decay=args.ensemble_decay)
        ens = np.stack([ensemble(torch.from_numpy(pred[t])).numpy() for t in range(T)])
        target = np.minimum(np.arange(T) + f, T - 1)
        ens_err = ens - truth[target]
        print(f"temporal ensemble f={f}, m={args.ensemble_decay}: MAE per joint "
              f"{np.round(np.abs(ens_err).mean(0), 4).tolist()}, all joints {np.abs(ens_err).mean():.4f} "
              f"(vs truth at t + f)")

    fig, axes = plt.subplots(D, 2, figsize=(16, 2.6 * D), sharex=True, gridspec_kw={"width_ratios": [3, 1.3]})
    steps = np.arange(T)
    for j in range(D):
        ax = axes[j, 0]
        for t in range(0, T, args.chunk_every):
            h = min(H, T - t)
            ax.plot(steps[t:t + h], pred[t, :h, j], color="tab:orange", alpha=0.35, lw=1,
                    label="predicted chunk" if t == 0 else None)
        ax.plot(steps, truth[:, j], color="black", lw=2, label="recorded (ground truth)")
        ax.plot(steps, pred[:, 0, j], color="tab:blue", lw=1.3, label="predicted, first action")
        if ens is not None:
            ax.plot(steps + args.ensemble_offset, ens[:, j], color="tab:green", lw=1.3,
                    label=f"temporal ensemble f={args.ensemble_offset} m={args.ensemble_decay} (at t + f)")
        ax.set_ylabel(f"joint {j}")
        ax.text(0.01, 0.92, f"MAE {np.abs(first_err[:, j]).mean():.4f}  max {np.abs(first_err[:, j]).max():.4f}",
                transform=ax.transAxes, fontsize=8, va="top")
        err_ax = axes[j, 1]
        err_ax.plot(steps, first_err[:, j], color="tab:red", lw=1)
        if ens is not None:
            err_ax.plot(steps + args.ensemble_offset, ens_err[:, j], color="tab:green", lw=1)
        err_ax.axhline(0, color="gray", lw=0.6)
        err_ax.set_ylabel("pred - true")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].set_title(f"{args.run_dir.name}, {ckpt.name}: val episode {paths[episode].stem}, raw units, open loop")
    axes[0, 1].set_title("error of the first predicted action")
    axes[-1, 0].set_xlabel("timestep")
    axes[-1, 1].set_xlabel("timestep")
    fig.tight_layout()
    suffix = "" if ens is None else f"_ensemble_f{args.ensemble_offset}_m{args.ensemble_decay}"
    out = args.run_dir / f"val_rollout_{paths[episode].stem}{suffix}.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
