import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import pickle
    import re
    import sys

    import h5py
    import numpy as np
    import pytest
    import torch
    from torch.utils.data import DataLoader

    from diff_style.datasets.episode_dataset import EpisodeDataset


@app.function
def frame(t, cam, episode):
    """2x4 image where every pixel is different: pixel (row, col) = (R, G, B)
    = (t, 10 * cam + row, 10 * episode + col). E.g. [1, 11, 103]: t=1, cam 1, row 1, episode 10, col 3."""
    row, col = np.indices((2, 4))
    return np.stack([np.full((2, 4), t), 10 * cam + row, 10 * episode + col], axis=-1).astype(np.uint8)


@app.function
def write_episode(path, action, gripper, main):
    """One episode with the same keys, dtypes and shapes as the real recordings, except
    fewer steps and 2x4 images. qpos = -action, so the two can't be confused."""
    action = np.array(action, dtype=np.float64)  # (T, 7)
    zeros = np.zeros(len(action))
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = False
        f["action"] = action
        f["observations/qpos"] = -action
        f["observations/qvel"] = np.zeros_like(action)
        f["observations/images/gripper"] = np.stack(gripper)  # (T, 2, 4, 3) uint8
        f["observations/images/main"] = np.stack(main)
        f["value_assignment/rew_total"] = zeros
        f["value_assignment/v_target"] = zeros
        f["value_assignment/v_target_normalized"] = zeros


@app.function
def write_episodes(folder):
    """Two episodes where every number says where it comes from:
    action = 100 * episode + 10 * t + joint (e.g. 1023: episode 10, t=2, joint 3).
    Cameras: gripper=0, main=1 (see `frame`)."""
    write_episode(
        folder / "episode_2.hdf5",
        action=[
            [200, 201, 202, 203, 204, 205, 206],  # t=0
            [210, 211, 212, 213, 214, 215, 216],  # t=1
            [220, 221, 222, 223, 224, 225, 226],  # t=2
        ],
        gripper=[frame(0, 0, 2), frame(1, 0, 2), frame(2, 0, 2)],
        main=[frame(0, 1, 2), frame(1, 1, 2), frame(2, 1, 2)],
    )
    write_episode(
        folder / "episode_00010.hdf5",  # zero padded: must still sort after episode_2
        action=[
            [1000, 1001, 1002, 1003, 1004, 1005, 1006],  # t=0
            [1010, 1011, 1012, 1013, 1014, 1015, 1016],  # t=1
            [1020, 1021, 1022, 1023, 1024, 1025, 1026],  # t=2
            [1030, 1031, 1032, 1033, 1034, 1035, 1036],  # t=3
        ],
        gripper=[frame(0, 0, 10), frame(1, 0, 10), frame(2, 0, 10), frame(3, 0, 10)],
        main=[frame(0, 1, 10), frame(1, 1, 10), frame(2, 1, 10), frame(3, 1, 10)],
    )
    return folder


@app.function
def write_many_episodes(folder, num_episodes):
    """episode_0 ... episode_<num_episodes - 1>, 3 steps each, written like write_episodes with the same
    numbers: action = 100 * episode + 10 * t + joint (e.g. 410: episode_4, t=1). At most 25 episodes,
    so the image pixels (10 * episode + col) fit in uint8."""
    for ep in range(num_episodes):
        write_episode(
            folder / f"episode_{ep}.hdf5",
            action=[[100 * ep + 10 * t + joint for joint in range(7)] for t in range(3)],
            gripper=[frame(t, 0, ep) for t in range(3)],
            main=[frame(t, 1, ep) for t in range(3)],
        )
    return folder


@app.function
def test_episodes_in_numeric_order_one_sample_per_step(tmp_path):
    dataset = EpisodeDataset(write_episodes(tmp_path))
    assert [p.name for p in dataset.paths] == ["episode_2.hdf5", "episode_00010.hdf5"]
    assert len(dataset) == 3 + 4


@app.function
def test_episodes_picks_some_episodes_by_position(tmp_path):
    """`episodes` are positions in the sorted list [episode_2, episode_00010], as the datamodule's split gives them."""
    folder = write_episodes(tmp_path)  # written once: the episode files stay open while a dataset reads them
    dataset = EpisodeDataset(folder, obs_horizon_1d=3, obs_horizon_2d=3, action_horizon=4, episodes=[1])
    assert [p.name for p in dataset.paths] == ["episode_00010.hdf5"]
    assert len(dataset) == 4  # its 4 steps only
    sample = dataset[1]  # episode_00010, t=1: qpos, actions and images all from that episode
    assert sample["qpos"][:, 0].tolist() == [-1000, -1000, -1010]  # steps 0, 0, 1
    assert sample["actions"][:, 0].tolist() == [1010, 1020, 1030, 1030]  # steps 1, 2, 3, 3
    assert sample["images"][:, :, :, 0, 0].tolist() == [  # (t, 10 * cam, 10 * episode)
        [[0, 0, 100], [0, 10, 100]],
        [[0, 0, 100], [0, 10, 100]],
        [[1, 0, 100], [1, 10, 100]],
    ]
    assert len(EpisodeDataset(folder, episodes=[])) == 0  # an empty split, e.g. test_fraction=0


@app.function
def test_many_episodes_follow_the_same_numbers(tmp_path):
    dataset = EpisodeDataset(write_many_episodes(tmp_path, 10), obs_horizon_1d=1, obs_horizon_2d=1, action_horizon=1)
    assert [p.stem for p in dataset.paths] == [f"episode_{ep}" for ep in range(10)]
    assert len(dataset) == 10 * 3
    sample = dataset[3 * 4 + 1]  # episode_4, t=1
    assert sample["actions"][0].tolist() == [410, 411, 412, 413, 414, 415, 416]
    assert sample["qpos"][0].tolist() == [-410, -411, -412, -413, -414, -415, -416]
    assert sample["images"][0, :, :, 0, 0].tolist() == [[1, 0, 40], [1, 10, 40]]  # (t, 10 * cam, 10 * episode)


@app.function
def test_windows_repeat_first_and_last_step(tmp_path):
    dataset = EpisodeDataset(write_episodes(tmp_path), obs_horizon_1d=3, action_horizon=4)
    first = dataset[0]  # episode_2, t=0
    assert first["qpos"].dtype == first["actions"].dtype == torch.float32
    assert first["qpos"][:, 0].tolist() == [-200, -200, -200]  # steps 0, 0, 0
    assert first["actions"][:, 0].tolist() == [200, 210, 220, 220]  # steps 0, 1, 2, 2
    later = dataset[4]  # episode_00010, t=1
    assert later["qpos"][:, 0].tolist() == [-1000, -1000, -1010]  # steps 0, 0, 1
    assert later["actions"][:, 0].tolist() == [1010, 1020, 1030, 1030]  # steps 1, 2, 3, 3
    assert later["actions"][0].tolist() == [1010, 1011, 1012, 1013, 1014, 1015, 1016]  # all joints, in order


@app.function
def test_past_actions_end_before_the_current_step(tmp_path):
    """Row t holds the next action. Past actions end at t-1, so action[t] is never in the past window."""
    def action(t):
        return [1000 + 10 * t + joint for joint in range(7)]

    zero = [0] * 7
    folder = write_episodes(tmp_path)  # written once: the episode files stay open while a dataset reads them
    dataset = EpisodeDataset(folder, obs_horizon_1d=4, obs_horizon_2d=1, action_horizon=2)
    expected = [
        [zero, zero, zero, zero],  # t=0: no previous row
        [zero, zero, zero, action(0)],  # t=1
        [zero, zero, action(0), action(1)],  # t=2
        [zero, action(0), action(1), action(2)],  # t=3
    ]
    assert dataset[3]["past_actions"].dtype == torch.float32
    for t, past_actions in enumerate(expected):
        sample = dataset[3 + t]  # episode_00010 starts at index 3
        assert sample["past_actions"].shape == (4, 7)  # (obs_horizon_1d, D)
        assert sample["past_actions"].tolist() == past_actions
        assert sample["actions"][0].tolist() == action(t)
        if t:
            assert not torch.equal(sample["past_actions"][-1], sample["actions"][0])

    one_step = EpisodeDataset(folder, obs_horizon_1d=1, obs_horizon_2d=1, action_horizon=1)
    assert [one_step[3 + t]["past_actions"][0, 0].item() for t in range(4)] == [0, 1000, 1010, 1020]


@app.function
def test_1d_and_2d_observation_horizons_are_independent(tmp_path):
    """1D observations (qpos) use `obs_horizon_1d`; images use `obs_horizon_2d`. Both windows end at t."""
    dataset = EpisodeDataset(write_episodes(tmp_path), obs_horizon_1d=4, obs_horizon_2d=2, action_horizon=1)
    sample = dataset[4]  # episode_00010, t=1
    assert sample["images"].shape == (2, 2, 3, 2, 4)  # (obs_horizon_2d, cameras, C, H, W)
    assert sample["images"][:, :, :, 0, 0].tolist() == [
        [[0, 0, 100], [0, 10, 100]],  # t-1
        [[1, 0, 100], [1, 10, 100]],  # t
    ]
    assert sample["qpos"][:, 0].tolist() == [-1000, -1000, -1000, -1010]  # steps 0, 0, 0, 1


@app.function
def test_all_three_horizons_are_independent(tmp_path):
    """The 1D, 2D and action windows have their own lengths, at a middle step and at the last step."""
    dataset = EpisodeDataset(write_episodes(tmp_path), obs_horizon_1d=4, obs_horizon_2d=2, action_horizon=5)
    sample = dataset[5]  # episode_00010, t=2
    assert sample["qpos"][:, 0].tolist() == [-1000, -1000, -1010, -1020]  # steps 0, 0, 1, 2
    assert sample["images"].shape == (2, 2, 3, 2, 4)  # (obs_horizon_2d, cameras, C, H, W)
    assert sample["images"][:, :, :, 0, 0].tolist() == [
        [[1, 0, 100], [1, 10, 100]],  # t-1
        [[2, 0, 100], [2, 10, 100]],  # t
    ]
    assert sample["actions"][:, 0].tolist() == [1020, 1030, 1030, 1030, 1030]  # steps 2, 3, 3, 3, 3
    last = dataset[6]  # episode_00010, t=3
    assert last["qpos"][:, 0].tolist() == [-1000, -1010, -1020, -1030]
    assert last["images"][:, :, :, 0, 0].tolist() == [
        [[2, 0, 100], [2, 10, 100]],
        [[3, 0, 100], [3, 10, 100]],
    ]
    assert last["actions"][:, 0].tolist() == [1030, 1030, 1030, 1030, 1030]


@app.function
def test_1d_window_ends_at_the_current_step(tmp_path):
    """Four 1D steps ending at t. Missing steps before the start repeat step 0."""
    dataset = EpisodeDataset(write_episodes(tmp_path), obs_horizon_1d=4, obs_horizon_2d=2, action_horizon=4)
    expected_qpos_joint0 = [
        [-1000, -1000, -1000, -1000],  # t=0: steps 0, 0, 0, 0
        [-1000, -1000, -1000, -1010],  # t=1: steps 0, 0, 0, 1
        [-1000, -1000, -1010, -1020],  # t=2: steps 0, 0, 1, 2
        [-1000, -1010, -1020, -1030],  # t=3: steps 0, 1, 2, 3
    ]
    for t, expected in enumerate(expected_qpos_joint0):
        assert dataset[3 + t]["qpos"][:, 0].tolist() == expected  # episode_00010 starts at index 3


@app.function
def test_non_positive_horizons_raise(tmp_path):
    for keyword in ["obs_horizon_1d", "obs_horizon_2d", "action_horizon"]:
        with pytest.raises(ValueError, match="at least 1"):
            EpisodeDataset(write_episodes(tmp_path), **{keyword: 0})


@app.function
def test_images_are_uint8_time_camera_channel_height_width(tmp_path):
    images = EpisodeDataset(write_episodes(tmp_path), obs_horizon_2d=3)[4]["images"]  # episode_00010, t=1
    assert images.dtype == torch.uint8
    assert images.shape == (3, 2, 3, 2, 4)  # (obs_horizon_2d, cameras, C, H, W)
    # Pixel (row 0, col 0) of every frame: (t, 10 * cam, 10 * episode)
    assert images[:, :, :, 0, 0].tolist() == [
        [[0, 0, 100], [0, 10, 100]],  # t-2 -> before the start: frame 0
        [[0, 0, 100], [0, 10, 100]],  # t-1
        [[1, 0, 100], [1, 10, 100]],  # t
    ]
    # One whole frame (t, main camera), channel by channel: R = t, G = 10 + row, B = 100 + col
    assert images[2, 1].tolist() == [
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[10, 10, 10, 10], [11, 11, 11, 11]],
        [[100, 101, 102, 103], [100, 101, 102, 103]],
    ]


@app.function
def test_survives_pickling(tmp_path):
    """DataLoader workers started with spawn (or on other nodes) get a pickled copy of the dataset."""
    dataset = EpisodeDataset(write_episodes(tmp_path))
    copy = pickle.loads(pickle.dumps(dataset))
    for key, value in dataset[0].items():
        assert torch.equal(copy[0][key], value)


@app.function
def legacy_sample(dataset, index):
    """The sample as the HDF5 reader before the raw reads made it (h5py per sample, the version that leaked):
    the reference every sample of the new reader must equal, value, dtype, shape and memory layout."""
    def clamp(start, stop, last):
        return [min(max(i, 0), last) for i in range(start, stop)]

    ep, t = dataset.index[index].tolist()
    last = len(dataset.actions[ep]) - 1
    obs_1d_start = t - dataset.obs_horizon_1d + 1
    obs_1d_ts = clamp(obs_1d_start, t + 1, last)
    obs_2d_ts = clamp(t - dataset.obs_horizon_2d + 1, t + 1, last)
    past_action_ts = clamp(obs_1d_start, t + 1, len(dataset.padded_actions[ep]) - 1)
    act_ts = clamp(t, t + dataset.action_horizon, last)
    with h5py.File(dataset.paths[ep], "r", locking=False) as f:
        images = [[f[f"observations/images/{cam}"][i] for cam in dataset.camera_names] for i in obs_2d_ts]
    return {
        "images": torch.from_numpy(np.array(images)).permute(0, 1, 4, 2, 3),
        "qpos": dataset.qpos[ep][obs_1d_ts],
        "past_actions": dataset.padded_actions[ep][past_action_ts],
        "actions": dataset.actions[ep][act_ts],
    }


@app.function
def assert_same_sample(sample, reference):
    assert sample.keys() == reference.keys()
    for key, value in reference.items():
        got = sample[key]
        assert (got.dtype, got.shape, got.stride()) == (value.dtype, value.shape, value.stride()), key
        assert torch.equal(got, value), key


@app.function
def write_random_episodes(folder, seed=0):
    """Episodes unlike the hand-made ones: random pixels (every byte matters), 3 cameras, non-square frames,
    different lengths (1 to 9 steps), zero-padded names, and some files with a userblock (the HDF5 data
    starts 512 or 1024 bytes into the file), which moves every frame offset."""
    rng = np.random.default_rng(seed)
    cameras = ["gripper", "main", "top"]
    for ep, (steps, userblock) in enumerate([(5, 0), (1, 512), (9, 0), (2, 1024), (7, 0), (3, 512)]):
        with h5py.File(folder / f"episode_{ep:03d}.hdf5", "w", userblock_size=userblock) as f:
            f["action"] = rng.normal(size=(steps, 7))
            f["observations/qpos"] = rng.normal(size=(steps, 7))
            for cam in cameras:
                f[f"observations/images/{cam}"] = rng.integers(0, 256, (steps, 6, 10, 3), dtype=np.uint8)
    return folder


@app.function
@pytest.mark.parametrize("horizons", [(1, 1, 1), (2, 2, 16), (3, 1, 4), (1, 4, 2), (5, 12, 3)])
def test_same_samples_as_the_hdf5_reader_random_episodes(tmp_path, horizons):
    """Every sample, at every edge of every window, against the old reader."""
    obs_horizon_1d, obs_horizon_2d, action_horizon = horizons
    folder = write_random_episodes(tmp_path)
    for episodes in [None, [4, 1, 2]]:
        dataset = EpisodeDataset(folder, obs_horizon_1d, obs_horizon_2d, action_horizon, episodes=episodes)
        for i in range(len(dataset)):
            assert_same_sample(dataset[i], legacy_sample(dataset, i))


@app.function
def test_same_samples_as_the_hdf5_reader_hand_made_episodes(tmp_path):
    folder = write_many_episodes(tmp_path, 25)
    dataset = EpisodeDataset(folder, obs_horizon_1d=2, obs_horizon_2d=3, action_horizon=4)
    for i in range(len(dataset)):
        assert_same_sample(dataset[i], legacy_sample(dataset, i))


@app.function
def test_every_frame_equals_hdf5(tmp_path):
    """The raw reads give every frame of every camera exactly as h5py reads it, userblock or not."""
    dataset = EpisodeDataset(write_random_episodes(tmp_path))
    for ep, path in enumerate(dataset.paths):
        with h5py.File(path, "r") as f:
            expected = np.stack([f[f"observations/images/{cam}"][()] for cam in dataset.camera_names], axis=1)
        assert np.array_equal(dataset._read_frames(ep, list(range(len(expected)))), expected)


@app.function
def test_same_samples_as_the_hdf5_reader_recorded_episodes(tmp_path):
    """Episodes as the recording converter writes them (act/convert_mcap_to_hdf5.py: the same create_dataset
    calls, attributes and keys) at the real camera size, 480x640 RGB (921,600 bytes per frame)."""
    rng = np.random.default_rng(1)
    for ep, steps in enumerate([3, 5]):
        action = rng.normal(size=(steps, 7))
        with h5py.File(tmp_path / f"episode_{ep}.hdf5", "w") as f:
            f.attrs["sim"] = False
            f.create_dataset("/action", data=action)
            obs_group = f.create_group("observations")
            obs_group.create_dataset("qpos", data=rng.normal(size=(steps, 7)))
            obs_group.create_dataset("qvel", data=rng.normal(size=(steps, 7)))
            images_group = obs_group.create_group("images")
            images_group.create_dataset("gripper", data=rng.integers(0, 256, (steps, 480, 640, 3), dtype=np.uint8))
            images_group.create_dataset("main", data=rng.integers(0, 256, (steps, 480, 640, 3), dtype=np.uint8))
    dataset = EpisodeDataset(tmp_path, obs_horizon_1d=2, obs_horizon_2d=2, action_horizon=16)
    assert dataset.frame_shapes.tolist() == [[480, 640, 3], [480, 640, 3]]
    for i in range(len(dataset)):
        assert_same_sample(dataset[i], legacy_sample(dataset, i))


@app.function
def test_frames_beyond_8_gib_into_the_file(tmp_path):
    """Large files: the frames start past 2^31 and 2^32 bytes, where 32-bit offset arithmetic would wrap. An
    8 GiB userblock puts them there; the file is sparse (the userblock is never written), ~4 KB on disk."""
    rng = np.random.default_rng(2)
    with h5py.File(tmp_path / "episode_0.hdf5", "w", userblock_size=2**33) as f:
        f["action"] = rng.normal(size=(4, 7))
        f["observations/qpos"] = rng.normal(size=(4, 7))
        f["observations/images/gripper"] = rng.integers(0, 256, (4, 6, 10, 3), dtype=np.uint8)
        f["observations/images/main"] = rng.integers(0, 256, (4, 6, 10, 3), dtype=np.uint8)
    dataset = EpisodeDataset(tmp_path, obs_horizon_2d=2)
    assert dataset.image_offsets.min() > 2**33
    for i in range(len(dataset)):
        assert_same_sample(dataset[i], legacy_sample(dataset, i))


@app.function
def test_no_hdf5_after_the_dataset_is_made(tmp_path, monkeypatch):
    """h5py keeps a little memory for good at every File() and f["path"] lookup (the OOM after ~90k steps):
    samples never touch h5py, only the constructor does."""
    dataset = EpisodeDataset(write_random_episodes(tmp_path), obs_horizon_2d=2)

    def no_hdf5(*args, **kwargs):
        raise AssertionError("h5py used while reading a sample")

    monkeypatch.setattr(h5py, "File", no_hdf5)
    monkeypatch.setattr(h5py.h5f, "open", no_hdf5)
    for i in range(len(dataset)):
        dataset[i]


@app.function
@pytest.mark.skipif(sys.platform != "linux", reason="reads RssAnon from /proc/self/status")
def test_memory_stays_flat_over_many_reads(tmp_path):
    """The leak this guards against: ~1.2 KB per sample (h5py.File() + f["path"] per sample), ~23 MB here."""
    def anon_bytes():
        with open("/proc/self/status") as status:
            return 1024 * int(re.search(r"RssAnon:\s+(\d+)", status.read()).group(1))

    dataset = EpisodeDataset(write_many_episodes(tmp_path, 4), obs_horizon_2d=2)
    for i in range(len(dataset)):  # warm up the allocator's pools
        dataset[i]
    before, num_reads = anon_bytes(), 20_000
    for i in range(num_reads):
        dataset[i % len(dataset)]
    assert (anon_bytes() - before) / num_reads < 100  # bytes per sample, a leak per sample would be ~1200


@app.function
def test_no_file_stays_open_between_samples(tmp_path):
    """Any number of episodes: nothing is held open, so no limit on open files and nothing per episode in memory."""
    import os

    dataset = EpisodeDataset(write_random_episodes(tmp_path), obs_horizon_2d=3)
    before = len(os.listdir("/proc/self/fd")) if sys.platform == "linux" else None
    for i in range(len(dataset)):
        dataset[i]
    if before is not None:
        assert len(os.listdir("/proc/self/fd")) == before


@app.function
def test_samples_are_independent_writable_tensors(tmp_path):
    """Each sample gets its own image memory, as before: changing one never changes the file or the next read."""
    dataset = EpisodeDataset(write_random_episodes(tmp_path), obs_horizon_2d=2)
    first = dataset[3]
    reference = first["images"].clone()
    first["images"].zero_()  # in place: would fail on a read-only buffer
    assert torch.equal(dataset[3]["images"], reference)


@app.function
@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_dataloader_workers_give_the_same_samples(tmp_path, context):
    """Forked workers (Linux default) and spawned ones (a pickled copy, as on other platforms or nodes), two
    epochs with persistent workers: every sample equals the old reader's."""
    dataset = EpisodeDataset(write_random_episodes(tmp_path), obs_horizon_1d=2, obs_horizon_2d=2, action_horizon=3)
    expected = [legacy_sample(dataset, i) for i in range(len(dataset))]
    loader = DataLoader(dataset, batch_size=4, num_workers=2, multiprocessing_context=context,
                        persistent_workers=True)
    for _ in range(2):
        got = [{key: value[j] for key, value in batch.items()} for batch in loader for j in range(len(batch["qpos"]))]
        assert len(got) == len(expected)
        for sample, reference in zip(got, expected):
            for key in reference:
                assert torch.equal(sample[key], reference[key]), key


@app.function
def test_survives_pickling_after_reading(tmp_path):
    dataset = EpisodeDataset(write_random_episodes(tmp_path), obs_horizon_2d=2)
    first = dataset[4]
    copy = pickle.loads(pickle.dumps(dataset))
    assert_same_sample(copy[4], first)


@app.function
def write_one_episode(path, images=None, **image_kwargs):
    """2 steps, cameras gripper and main. `images` replaces the main camera's frames; `image_kwargs` go to
    create_dataset for it (chunks, compression, external...)."""
    with h5py.File(path, "w") as f:
        f["action"] = np.zeros((2, 7))
        f["observations/qpos"] = np.zeros((2, 7))
        f["observations/images/gripper"] = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        main = np.zeros((2, 4, 6, 3), dtype=np.uint8) if images is None else images
        if image_kwargs:
            f.create_dataset("observations/images/main", data=main, **image_kwargs)
        else:
            f["observations/images/main"] = main
    return path.parent


@app.function
@pytest.mark.parametrize("layout, message", [
    (dict(chunks=(1, 4, 6, 3)), "not contiguous"),
    (dict(compression="gzip"), "compressed or filtered|not contiguous"),
    (dict(fletcher32=True), "compressed or filtered|not contiguous"),
    (dict(images=np.zeros((2, 4, 6, 3), dtype=np.float32)), "not uint8"),
    (dict(images=np.zeros((2, 4, 6), dtype=np.uint8)), r"not \(T, H, W, C\)"),
    (dict(images=np.zeros((1, 4, 6, 3), dtype=np.uint8)), "1 frames for 2 steps"),
])
def test_frames_that_cannot_be_read_raw_raise_at_the_start(tmp_path, layout, message):
    """A layout the raw reads can't follow raises when the dataset is made, naming the file: never wrong frames."""
    write_one_episode(tmp_path / "episode_0.hdf5", **layout)
    with pytest.raises(ValueError, match=message) as error:
        EpisodeDataset(tmp_path)
    assert "episode_0.hdf5" in str(error.value) and "main" in str(error.value)


@app.function
def test_frames_in_external_files_raise(tmp_path):
    raw = tmp_path / "frames.raw"
    raw.write_bytes(bytes(2 * 4 * 6 * 3))
    with h5py.File(tmp_path / "episode_0.hdf5", "w") as f:
        f["action"] = np.zeros((2, 7))
        f["observations/qpos"] = np.zeros((2, 7))
        f["observations/images/gripper"] = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        f.create_dataset("observations/images/main", shape=(2, 4, 6, 3), dtype=np.uint8, external=[(raw, 0, 144)])
    with pytest.raises(ValueError, match="external files"):
        EpisodeDataset(tmp_path)


@app.function
def test_frames_never_written_raise(tmp_path):
    """Created but never written: no bytes in the file yet (h5py would give zeros, the raw reads garbage)."""
    with h5py.File(tmp_path / "episode_0.hdf5", "w") as f:
        f["action"] = np.zeros((2, 7))
        f["observations/qpos"] = np.zeros((2, 7))
        f["observations/images/gripper"] = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        f.create_dataset("observations/images/main", shape=(2, 4, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not fully written"):
        EpisodeDataset(tmp_path)


@app.function
def test_episodes_with_different_cameras_raise(tmp_path):
    write_one_episode(tmp_path / "episode_0.hdf5")
    with h5py.File(tmp_path / "episode_1.hdf5", "w") as f:
        f["action"] = np.zeros((2, 7))
        f["observations/qpos"] = np.zeros((2, 7))
        f["observations/images/main"] = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same cameras"):
        EpisodeDataset(tmp_path)


@app.function
def test_cameras_with_different_frame_shapes_raise(tmp_path):
    write_one_episode(tmp_path / "episode_0.hdf5", images=np.zeros((2, 8, 6, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="different frame shapes"):
        EpisodeDataset(tmp_path)


@app.function
def test_a_file_cut_short_after_the_dataset_is_made_raises(tmp_path):
    """Files must not change while training. If one is cut short anyway, the read fails loudly (no silent zeros)."""
    folder = write_random_episodes(tmp_path)
    dataset = EpisodeDataset(folder, obs_horizon_2d=1)
    last = dataset.paths[2]  # 9 steps
    end = int(dataset.image_offsets[2].max())  # cut inside the last camera's frames
    with open(last, "r+b") as f:
        f.truncate(end + 100)
    last_step_of_episode_2 = int(np.flatnonzero(dataset.index[:, 0] == 2)[-1])  # its last step: the frame beyond the cut
    with pytest.raises(OSError, match="file changed"):
        dataset[last_step_of_episode_2]


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Episode dataset

    The `test_*` functions above run under pytest (`uv run pytest`) on tiny fake episodes.
    The next cell prints those fake episodes and two samples made from them.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as _folder:
        _folder = write_episodes(Path(_folder))
        for _path in sorted(_folder.glob("*.hdf5")):
            print(f"===== {_path.name}")
            with h5py.File(_path) as _f:
                _f.visititems(lambda name, obj: print(f"{name:40s} {obj.shape} {obj.dtype}") if isinstance(obj, h5py.Dataset) else None)
                print("action (T, 7):")
                print(_f["action"][()])
                print("gripper frame at t=1, channel by channel (C, H, W):")
                print(_f["observations/images/gripper"][1].transpose(2, 0, 1))

        _dataset = EpisodeDataset(_folder, obs_horizon_1d=3, obs_horizon_2d=2, action_horizon=4)
        print("\n===== dataset (obs_horizon_1d=3, obs_horizon_2d=2, action_horizon=4)")
        print("paths:", [p.name for p in _dataset.paths], "| cameras:", _dataset.camera_names)
        print("index (episode, t):", _dataset.index.tolist())
        for _i, _label in [(0, "episode_2, t=0"), (4, "episode_00010, t=1")]:
            _sample = _dataset[_i]
            print(f"\n----- dataset[{_i}] = {_label}")
            print("qpos (obs_horizon_1d, 7):")
            print(_sample["qpos"].int())  # .int() only to print without decimals
            print("past_actions (obs_horizon_1d, 7):")
            print(_sample["past_actions"].int())
            print("actions (action_horizon, 7):")
            print(_sample["actions"].int())
            print("images", tuple(_sample["images"].shape), "pixel (row 0, col 0) per obs step and camera, (R, G, B):")
            print(_sample["images"][:, :, :, 0, 0].tolist())
            print("images[-1, 1]: main camera at t, channel by channel (C, H, W):")
            print(_sample["images"][-1, 1])
    return


if __name__ == "__main__":
    app.run()
