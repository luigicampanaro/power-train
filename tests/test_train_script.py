"""scripts/train.py run as you start it, with a tiny model: each run in its own new folder, outputs/<timestamp> in the
folder it starts in, with every file of the run; and a run on several machines needs one folder name for all of
them, given with hydra.run.dir on every machine."""

import os
import re
import subprocess
import sys
from pathlib import Path

from test_episode_dataset import write_many_episodes

PROJECT = Path(__file__).resolve().parent.parent
TINY = ["compile=false", "model_dim=32", "model.net.perceiver.num_latents=4",
        "model.net.vision_encoder._target_=test_flow_matching_network.PixelEncoder",
        "~model.net.vision_encoder.model_id", "model.net.vision_encoder.dim=8",
        *[f"model.net.{module}.num_heads=4" for module in ["perceiver", "obs_encoder", "action_decoder"]],
        "datamodule.batch_size=8", "datamodule.num_workers=0", "image_size=[8,8]", "trainer.accelerator=cpu",
        "trainer.precision=32", "trainer.max_steps=2", "trainer.val_check_interval=2", "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=1", "+trainer.enable_progress_bar=false", "~callbacks.progress_bar",
        "model.lr_scheduler.warmup_steps=1"]


def train(cwd: Path, *overrides: str) -> subprocess.CompletedProcess:
    """python scripts/train.py with the tiny model, started in cwd; the tests folder holds the PixelEncoder stand-in."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PROJECT / "tests"), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(PROJECT / "scripts" / "train.py"), *TINY, *overrides], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def test_a_run_writes_every_file_into_one_new_folder_outputs_timestamp(tmp_path):
    (tmp_path / "data").mkdir()
    result = train(tmp_path, f"datamodule.dataset_dir={write_many_episodes(tmp_path / 'data', 10)}")
    assert result.returncode == 0, result.stderr[-3000:]
    runs = list((tmp_path / "outputs").iterdir())
    assert len(runs) == 1 and re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", runs[0].name)
    files = {path.relative_to(runs[0]).as_posix() for path in runs[0].rglob("*")}
    assert {".hydra/config.yaml", "stats.pt", "hparams.yaml", "checkpoints/last.ckpt"} <= files
    assert any(name.startswith("events.out.tfevents") for name in files)


def test_a_run_on_several_machines_needs_one_folder_name_given_on_every_machine(tmp_path):
    """Each machine starts its own process, with its own clock: a timestamp would give each machine its own folder."""
    result = train(tmp_path, "trainer.num_nodes=2")
    assert result.returncode != 0 and "hydra.run.dir" in result.stderr
