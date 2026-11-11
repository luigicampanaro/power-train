"""The exponential moving average (EMA) of the weights that train: Lightning's EMAWeightAveraging, without the
frozen parameters."""

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EMAWeightAveraging
from torch import nn
from torch.optim.swa_utils import AveragedModel


class TrainableEMA(EMAWeightAveraging):
    """EMAWeightAveraging (decay, update_every_n_steps, ...: the same options, validation on the average, the
    average copied in at the end of training and saved as the checkpoint's state_dict), but only the parameters
    that train are averaged. A frozen parameter is never averaged: it stays bit-identical (Lightning's
    0.999 * ema + 0.001 * p rounds even a value that does not change)."""

    def __init__(self, decay: float = 0.999, **kwargs) -> None:
        super().__init__(decay=decay, **kwargs)
        self.decay = decay

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            self._average_model = TrainableAveragedModel(pl_module, self._device or pl_module.device, self.decay)


class TrainableAveragedModel(AveragedModel):
    """A copy of the model. At each update, its trainable parameters (requires_grad, read at every update) become
    the EMA of the model's, lerp(average, current, 1 - decay); its frozen parameters stay as copied; its buffers
    are copied from the model's."""

    def __init__(self, model: nn.Module, device: torch.device | None, decay: float) -> None:
        super().__init__(model, device=device, use_buffers=False)
        self.decay = decay

    @torch.no_grad()
    def update_parameters(self, model: nn.Module) -> None:
        for average, current in zip(self.module.parameters(), model.parameters()):
            if current.requires_grad:
                # Lightning converts the model's precision after the average is built: the average keeps its own
                self._average(average, current.detach().to(average))
        for average, current in zip(self.module.buffers(), model.buffers()):
            average.copy_(current)
        self.n_averaged += 1

    def _average(self, average: torch.Tensor, current: torch.Tensor) -> None:
        if self.n_averaged == 0:
            average.copy_(current)
        else:
            average.lerp_(current, 1 - self.decay)
