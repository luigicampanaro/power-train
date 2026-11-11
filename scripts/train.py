"""Train a policy. From the project folder: uv run python scripts/train.py [Hydra overrides].

Each run gets its own new folder, outputs/<date>_<time> (configs/hydra), or hydra.run.dir=outputs/<name>.
On several machines, start the same command on each one, with the same hydra.run.dir and trainer.num_nodes=<machines>,
and Lightning's MASTER_ADDR (the IP of the NODE_RANK=0 machine), MASTER_PORT and NODE_RANK (0, 1, ...) set."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from diff_style.utils.tensorboard import launch_tensorboard


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    check_one_run_folder_for_all_machines(cfg.trainer.num_nodes)
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    if cfg.compile:
        model.compile()
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()]
    # name="" and version="": logs and checkpoints go straight into the run folder
    logger = TensorBoardLogger(cfg.run_dir, name="", version="")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, default_root_dir=cfg.run_dir)
    # TODO: resume a run: trainer.fit(..., ckpt_path=cfg.ckpt_path), with ckpt_path: null in train.yaml. last.ckpt
    # has all the state (live and EMA weights, AdamW, lr schedule, loops); resume with the same max_steps, which the
    # lr schedule follows.
    serve_tensorboard(cfg.run_dir)
    trainer.fit(model, datamodule=datamodule)


def check_one_run_folder_for_all_machines(num_nodes: int) -> None:
    """Each machine starts its own process, with its own clock: without one hydra.run.dir given on every machine, each
    machine would get its own outputs/<date>_<time> folder."""
    folder_given = any(override.startswith("hydra.run.dir=") for override in HydraConfig.get().overrides.hydra)
    if num_nodes > 1 and not folder_given:
        raise ValueError(f"trainer.num_nodes={num_nodes}: give the same hydra.run.dir=outputs/<name> on every machine")


@rank_zero_only
def serve_tensorboard(run_dir: str) -> None:
    """All runs: the parent of the run folder. It stops when the training ends."""
    print(f"TensorBoard: {launch_tensorboard(str(Path(run_dir).parent))}")


if __name__ == "__main__":
    main()
