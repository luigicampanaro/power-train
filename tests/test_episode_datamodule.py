import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    from types import SimpleNamespace

    import hydra
    import pytest
    import torch
    from omegaconf import OmegaConf

    from diff_style.datamodules.episode_datamodule import EpisodeDataModule
    from diff_style.datasets.episode_dataset import EpisodeDataset
    from diff_style.models.normalizer import Normalizer
    from test_episode_dataset import write_episodes, write_many_episodes  # tests/ is on sys.path under pytest and marimo

    CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@app.function
def make_datamodule(tmp_path):
    """The two fake episodes of test_episode_dataset: episode_2 (3 steps) and episode_00010 (4 steps),
    action = 100 * episode + 10 * t + joint, qpos = -action. Small enough to compute the stats by hand
    (the three-way split is tested on write_many_episodes). seed=0 permutation [0, 1]: episode_2 goes
    to val, episode_00010 is the train episode."""
    datamodule = EpisodeDataModule(
        write_episodes(tmp_path), stats_path=tmp_path / "run" / "stats.pt", obs_horizon_1d=3, obs_horizon_2d=2, action_horizon=4,
        train_fraction=0.5, val_fraction=0.5, test_fraction=0.0, batch_size=3, num_workers=0,
    )
    datamodule.setup("fit")
    return datamodule


@app.function
def test_split_by_whole_episode(tmp_path):
    """10 episodes split 0.6/0.2/0.2: each episode lands in exactly one split, with all its steps."""
    datamodule = EpisodeDataModule(
        write_many_episodes(tmp_path, 10), stats_path=tmp_path / "run" / "stats.pt",
        train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, batch_size=4, num_workers=0,
    )
    datamodule.setup("fit")
    # seed=0 permutation [4, 6, 2, 7, 3, 5, 9, 0, 8, 1]: the first 2 to val, the next 2 to test, the rest to train
    assert [p.stem for p in datamodule.train_ds.paths] == ["episode_0", "episode_1", "episode_3", "episode_5", "episode_8", "episode_9"]
    assert [p.stem for p in datamodule.val_ds.paths] == ["episode_4", "episode_6"]
    assert [p.stem for p in datamodule.test_ds.paths] == ["episode_2", "episode_7"]
    # Val and test give every step of their own episodes once, in order (drop_last is off there). Joint 0 of the first
    # action = 100 * episode + 10 * t
    first_actions = lambda loader: torch.cat([batch["actions"][:, 0, 0] for batch in loader]).int().tolist()
    assert first_actions(datamodule.val_dataloader()) == [400, 410, 420, 600, 610, 620]
    assert first_actions(datamodule.test_dataloader()) == [200, 210, 220, 700, 710, 720]
    # Train has drop_last (one batch shape for torch.compile): from 18 steps, 4 full batches of 4 and the partial one
    # is dropped. Shuffled, so the 2 left out change every epoch; every batch has the same shape
    train_batches = [batch["actions"][:, 0, 0].int().tolist() for batch in datamodule.train_dataloader()]
    assert [len(batch) for batch in train_batches] == [4, 4, 4, 4]
    train = [step for batch in train_batches for step in batch]
    train_steps = {100 * episode + 10 * t for episode in [0, 1, 3, 5, 8, 9] for t in [0, 1, 2]}
    assert len(set(train)) == 16 and set(train) < train_steps  # 2 of the 18 train steps are left out
    assert train != sorted(train)  # shuffled: 16 samples in order by chance once in 16!
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["images"].shape == (4, 1, 2, 3, 2, 4)  # (batch, obs_horizon_2d default 1, cameras, C, H, W)
    assert batch["qpos"].shape == (4, 2, 7)  # (batch, obs_horizon_1d, D)
    assert batch["past_actions"].shape == (4, 2, 7)  # (batch, obs_horizon_1d, D)
    assert batch["actions"].shape == (4, 16, 7)  # (batch, action_horizon, D)


@app.function
def test_batches_have_separate_1d_and_2d_observation_horizons(tmp_path):
    datamodule = EpisodeDataModule(
        write_episodes(tmp_path), stats_path=tmp_path / "run" / "stats.pt", obs_horizon_1d=4, obs_horizon_2d=2,
        action_horizon=3, train_fraction=0.5, val_fraction=0.5, test_fraction=0.0, batch_size=2, num_workers=0,
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))  # val = episode_2, in order
    assert batch["images"].shape == (2, 2, 2, 3, 2, 4)  # (batch, obs_horizon_2d, cameras, C, H, W)
    assert batch["qpos"].shape == (2, 4, 7)  # (batch, obs_horizon_1d, D)
    assert batch["past_actions"].shape == (2, 4, 7)  # (batch, obs_horizon_1d, D)
    assert batch["actions"].shape == (2, 3, 7)  # (batch, action_horizon, D)
    for i in range(2):  # each batch item is exactly the matching dataset sample
        for key in ["images", "qpos", "past_actions", "actions"]:
            assert torch.equal(batch[key][i], datamodule.val_ds[i][key])


@app.function
def test_every_batch_becomes_float_0_1_at_image_size_after_the_transfer():
    """on_after_batch_transfer runs in every stage (train, val, test, predict) on the GPU: the loaders
    send raw uint8 frames, the model gets float [0, 1] at image_size. 255 -> 1.0 checks the scale.
    Only the images change."""
    datamodule = EpisodeDataModule("unused", "unused", image_size=(4, 8))
    qpos = torch.zeros(2, 2, 7)
    batch = {"images": torch.full((2, 1, 2, 3, 2, 4), 255, dtype=torch.uint8), "qpos": qpos}
    batch = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)
    assert batch["images"].dtype == torch.float32
    assert torch.allclose(batch["images"], torch.ones(2, 1, 2, 3, 4, 8))  # (batch, frames, cameras, C, 4, 8)
    assert batch["qpos"] is qpos


@app.function
def test_resizing_a_raw_frame_stays_inside_0_1():
    """A white 480 x 640 camera frame resized to 224 x 224: the resize rounds to 1.0000002 unless clamped."""
    datamodule = EpisodeDataModule("unused", "unused", image_size=(224, 224))
    batch = {"images": torch.full((1, 1, 1, 3, 480, 640), 255, dtype=torch.uint8)}
    images = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)["images"]
    assert images.max() <= 1.0
    assert torch.allclose(images, torch.ones_like(images))


@app.function
def test_split_sizes_follow_the_fractions():
    """Only the episode count matters, so no files. The expected splits hold whatever the global
    (training) seed is: the split has its own generator, np.random.default_rng(seed)."""
    def split(num_episodes, train, val, test, seed=0):
        datamodule = EpisodeDataModule("unused", "unused", train_fraction=train, val_fraction=val, test_fraction=test, seed=seed)
        return {name: eps.tolist() for name, eps in datamodule.split_episodes(num_episodes).items()}

    # 5 episodes, permutation [2, 4, 3, 0, 1]: 0.1 * 5 = 0.5 rounds to 0, but a fraction > 0 gets at least 1
    assert split(5, 0.8, 0.1, 0.1) == {"train": [0, 1, 3], "val": [2], "test": [4]}  # the default config
    assert split(5, 0.9, 0.1, 0.0) == {"train": [0, 1, 3, 4], "val": [2], "test": []}  # fraction 0: none
    # 13 episodes, permutation [10, 2, 7, 4, ...]: rounded, not int()/ceil(): 0.2 * 13 = 2.6 -> 3 val, 0.1 * 13 = 1.3 -> 1 test
    assert split(13, 0.7, 0.2, 0.1) == {"train": [0, 1, 3, 5, 6, 8, 9, 11, 12], "val": [2, 7, 10], "test": [4]}
    # seed=1: permutation of 5 is [4, 0, 1, 2, 3]
    assert split(5, 0.8, 0.1, 0.1, seed=1) == {"train": [1, 2, 3], "val": [4], "test": [0]}


@app.function
def test_bad_fractions_raise():
    EpisodeDataModule("unused", "unused", train_fraction=0.7, val_fraction=0.2, test_fraction=0.1)  # sums to 0.9999999999999999: ok
    with pytest.raises(ValueError, match="sum to 1"):
        EpisodeDataModule("unused", "unused", train_fraction=0.9, val_fraction=0.1, test_fraction=0.1)
    with pytest.raises(ValueError, match="no train episodes"):  # 2 episodes: 1 val + 1 test, 0 left
        EpisodeDataModule("unused", "unused", train_fraction=0.0, val_fraction=0.5, test_fraction=0.5).split_episodes(2)


@app.function
def test_stats_from_raw_train_steps_only(tmp_path):
    """Train = episode_00010 only: action[:, joint] = 1000 + joint + [0, 10, 20, 30] -> mean 1015 + joint,
    std sqrt((15² + 5² + 5² + 15²) / 3) = sqrt(500 / 3) = 12.91. With episode_2 (val) in the stats, or
    over the windows (which repeat the edge steps), they would be different."""
    stats = make_datamodule(tmp_path).stats
    assert stats["actions"]["mean"].tolist() == [1015, 1016, 1017, 1018, 1019, 1020, 1021]
    assert stats["qpos"]["mean"].tolist() == [-1015, -1016, -1017, -1018, -1019, -1020, -1021]
    for key in ["actions", "qpos"]:
        assert torch.allclose(stats[key]["std"], torch.full((7,), (500 / 3) ** 0.5))
    # The file copy holds the same stats
    saved = torch.load(tmp_path / "run" / "stats.pt", weights_only=True)
    assert all(torch.equal(saved[key][s], stats[key][s]) for key in stats for s in ["mean", "std"])
    # With both episodes, all 7 steps are pooled: joint 0 = 200 210 220 | 1000 1010 1020 1030 -> mean
    # 4690 / 7 = 670 (the mean of the two episode means would be (210 + 1015) / 2 = 612.5)
    stats = EpisodeDataModule.compute_stats(EpisodeDataset(write_episodes(tmp_path)))
    assert stats["actions"]["mean"].tolist() == [670, 671, 672, 673, 674, 675, 676]


@app.function
def test_constant_joint_std_is_clamped():
    """A joint that never moves has std 0: clamped, so normalizing it doesn't divide by 0. The fake
    episodes have no constant joint, so this one feeds compute_stats a single in-memory episode."""
    steps = [torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])]  # joint 0 constant, joint 1 std 1
    stats = EpisodeDataModule.compute_stats(SimpleNamespace(qpos=steps, actions=steps))
    assert torch.allclose(stats["actions"]["std"], torch.tensor([0.01, 1.0]))


@app.function
def test_every_rank_computes_the_stats_only_rank_zero_writes_them(tmp_path):
    """Under DDP every rank runs setup() and needs the stats for its copy of the model, but only
    global rank 0 writes stats.pt: two processes saving the same .tmp file could leave a corrupt copy."""
    def datamodule(rank0):
        dm = EpisodeDataModule(write_episodes(tmp_path), stats_path=tmp_path / "run" / "stats.pt",
                               train_fraction=0.5, val_fraction=0.5, test_fraction=0.0, num_workers=0)
        dm.trainer = SimpleNamespace(is_global_zero=rank0)  # stands in for the pl.Trainer of each rank
        dm.setup("fit")
        return dm

    assert datamodule(rank0=False).stats["actions"]["mean"].tolist()[0] == 1015  # computed on every rank
    assert not (tmp_path / "run").exists()  # but written only by rank 0
    datamodule(rank0=True)
    assert (tmp_path / "run" / "stats.pt").exists()


@app.function
def test_normalizer_uses_the_stats_and_keeps_them_in_the_state_dict():
    """Hand-made stats with round numbers, so the results can be checked by eye."""
    normalizer = Normalizer(2)
    normalizer.set_stats({"mean": torch.tensor([10.0, 5.0]), "std": torch.tensor([4.0, 0.01])})
    assert normalizer.normalize(torch.tensor([[14.0, 5.0]])).tolist() == [[1, 0]]  # (14 - 10) / 4, (5 - 5) / 0.01
    assert normalizer.denormalize(torch.tensor([[1.0, 0.0]])).tolist() == [[14, 5]]
    # Buffers are in the state_dict, so they are saved in (and loaded from) the checkpoint
    loaded = Normalizer(2)
    loaded.load_state_dict(normalizer.state_dict())
    assert loaded.mean.tolist() == [10, 5] and torch.allclose(loaded.std, torch.tensor([4, 0.01]))


@app.function
def test_hydra_config_builds_the_datamodule(tmp_path):
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose("train", overrides=[f"datamodule.dataset_dir={write_episodes(tmp_path)}", f"output_dir={tmp_path}"])
    datamodule = hydra.utils.instantiate(OmegaConf.to_container(cfg, resolve=True)["datamodule"])
    assert isinstance(datamodule, EpisodeDataModule)
    assert datamodule.hparams.obs_horizon_1d == cfg.obs_horizon_1d
    assert datamodule.hparams.obs_horizon_2d == cfg.obs_horizon_2d
    assert datamodule.hparams.image_size == list(cfg.image_size)
    assert datamodule.stats_path == tmp_path / "stats.pt"  # <run_dir>/stats.pt, run_dir = output_dir


@app.function
def test_the_run_folder_is_hydras_run_folder_outputs_timestamp():
    """run_dir is the folder Hydra made for the run, not a new ${now:...} per process: Lightning passes that folder
    (hydra.run.dir) to the processes it starts, so every process of a machine gets the same run_dir."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose("train", return_hydra_config=True)
    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert unresolved["run_dir"] == "${output_dir}" and unresolved["output_dir"] == "${hydra:runtime.output_dir}"
    assert unresolved["hydra"]["run"]["dir"] == "outputs/${now:%Y-%m-%d_%H-%M-%S}"


@app.function
def test_saved_hparams_are_a_plain_config_that_rebuilds_the_datamodule(tmp_path):
    """The checkpoint stores datamodule.hparams: exactly the arguments, as plain values (str, not Path),
    so they load with weights_only=True and rebuild the same datamodule to repeat the training."""
    hparams = dict(make_datamodule(tmp_path).hparams)  # built from Python with Path arguments
    assert hparams == {
        "dataset_dir": str(tmp_path), "stats_path": str(tmp_path / "run" / "stats.pt"), "obs_horizon_1d": 3,
        "obs_horizon_2d": 2, "action_horizon": 4, "image_size": [224, 224], "train_fraction": 0.5, "val_fraction": 0.5, "test_fraction": 0.0,
        "batch_size": 3, "num_workers": 0, "seed": 0,
    }
    yaml = OmegaConf.to_yaml({"_target_": "diff_style.datamodules.episode_datamodule.EpisodeDataModule", **hparams})
    assert dict(hydra.utils.instantiate(OmegaConf.create(yaml)).hparams) == hparams


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Episode datamodule

    The `test_*` functions above run under pytest (`uv run pytest`) on the two fake episodes of
    `test_episode_dataset`, and the split on 10 episodes. The next cell prints which split each episode
    lands in with its actions, the stats computed from the train steps only, every batch each loader
    gives, and then the 10-episode split.
    """)
    return


@app.cell
def _():
    import tempfile

    with tempfile.TemporaryDirectory() as _folder:
        _datamodule = make_datamodule(Path(_folder))
        for _name in ["train", "val"]:
            _ds = getattr(_datamodule, f"{_name}_ds")
            for _path, _actions in zip(_ds.paths, _ds.actions):
                print(f"\n===== {_name}: {_path.stem}, actions (T, 7):")
                print(_actions.int())  # .int() only to print without decimals
        print()
        for _key, _stats in _datamodule.stats.items():
            print(f"{_key:8s} mean {_stats['mean'].tolist()}   std {[round(_s, 2) for _s in _stats['std'].tolist()]}")

        print("\nEvery batch of one pass per loader (train shuffled), joint 0 only:")
        print("action = 100 * episode + 10 * t + joint, qpos = -action (e.g. 1020: episode_00010, t=2)")
        for _name in ["train", "val"]:
            for _i, _batch in enumerate(getattr(_datamodule, f"{_name}_dataloader")()):
                print(f"\n----- {_name} batch {_i}")
                print("qpos (batch, obs_horizon_1d):")
                print(_batch["qpos"][:, :, 0].int())
                print("past_actions (batch, obs_horizon_1d):")
                print(_batch["past_actions"][:, :, 0].int())
                print("actions (batch, action_horizon):")
                print(_batch["actions"][:, :, 0].int())

    with tempfile.TemporaryDirectory() as _folder:
        _datamodule = EpisodeDataModule(
            write_many_episodes(Path(_folder), 10), stats_path=Path(_folder) / "stats.pt",
            train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, batch_size=4, num_workers=0,
        )
        _datamodule.setup("fit")
        print("\n===== 10 episodes, split 0.6 / 0.2 / 0.2")
        print("joint 0 of each sample's first action, per loader (action = 100 * episode + 10 * t):")
        for _name in ["train", "val", "test"]:
            _loader = getattr(_datamodule, f"{_name}_dataloader")()
            _first = torch.cat([_batch["actions"][:, 0, 0] for _batch in _loader]).int().tolist()
            print(f"{_name:5s} {[_p.stem for _p in getattr(_datamodule, f'{_name}_ds').paths]}")
            print(f"      {_first}")
    return


if __name__ == "__main__":
    app.run()
