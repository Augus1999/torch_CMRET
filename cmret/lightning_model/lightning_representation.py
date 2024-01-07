# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Lightning wrapper of CMRET model.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
from lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..model import CMRETModel
from ..utils.tools import loss_calc


DEFAULT_HPARAM: Dict[str, Union[str, int, float]] = {
    "model_unit": "Hartree",
    "lr_scheduler_factor": 0.9,
    "lr_scheduler_patience": 50,
    "lr_scheduler_interval": "epoch",  # can be "step" as well
    "lr_scheduler_frequency": 1,
    "lr_warmup_step": 10000,
    "max_lr": 1e-3,
    "ema_alpha": 0.05,  # EMA alpha value
}


class CMRET4Training(LightningModule):
    def __init__(
        self, cmret: CMRETModel, hparam: Dict[str, Any] = DEFAULT_HPARAM
    ) -> None:
        """
        A `~lightning.LightningModule` wrapper of CMRET model.\n
        This model is for training only. After training, calling `CMRET4Training(...).export_model(YOUR_WORKDIR)`
        will save the trained model to `YOUR_WORKDIR/trained.pt`, which can be loaded later by calling
        `~cmret.CMRETModel.from_checkpoint(...)`

        :param cmret: `~cmret.model.representation.CMRETModel` instance
        :param hparam: training hyperparameters. See `~cmret.light_model.lightning_representation.DEFAULT_HPARAM`.
        """
        super().__init__()
        self.cmret = cmret
        self.mse_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()
        self.scalar_loss: Optional[float] = None
        self.val_scalar_loss: Optional[float] = None
        self.save_hyperparameters(hparam)
        self.cmret.unit = hparam["model_unit"]

    def training_step(self, batch: Dict[str, Dict[str, Tensor]]) -> Tensor:
        mol, label = batch["mol"], batch["label"]
        nb = mol["batch"].shape[0]
        out = self.cmret.forward(mol)
        loss_dict = loss_calc(out, label, self.mse_fn)
        if "scalar" in loss_dict and "vector" in loss_dict:
            a = self.hparams.ema_alpha
            b = 1 - a
            scalar_loss = loss_dict["scalar"]
            vector_loss = loss_dict["vector"]
            if self.scalar_loss is None:
                self.scalar_loss = scalar_loss.item()
            else:
                scalar_loss = a * scalar_loss + b * self.scalar_loss
                self.scalar_loss = scalar_loss.item()
            loss = 0.2 * scalar_loss + 0.8 * vector_loss
            self.log("train_scalar_loss", scalar_loss.item(), batch_size=nb)
            self.log("train_vector_loss", vector_loss.item(), batch_size=nb)
        elif "scalar" in loss_dict:
            loss = loss_dict["scalar"]
            self.log("train_scalar_loss", loss.item(), batch_size=nb)
        elif "vector" in loss_dict:
            loss = loss_dict["vector"]
            self.log("train_vector_loss", loss.item(), batch_size=nb)
        return loss

    def validation_step(self, batch: Dict[str, Dict[str, Tensor]]) -> None:
        mol, label = batch["mol"], batch["label"]
        nb = mol["batch"].shape[0]
        enable_grad = False
        if hasattr(self.cmret.model.out, "dy"):
            enable_grad = self.cmret.model.out.dy
        with torch.set_grad_enabled(enable_grad):
            out = self.cmret.forward(mol)
            val_loss_dict = loss_calc(out, label, self.mae_fn)
            if "scalar" in val_loss_dict and "vector" in val_loss_dict:
                a = self.hparams.ema_alpha
                b = 1 - a
                val_scalar_loss = val_loss_dict["scalar"].item()
                val_vector_loss = val_loss_dict["vector"].item()
                if self.val_scalar_loss is None:
                    self.val_scalar_loss = val_scalar_loss
                else:
                    val_scalar_loss = a * val_scalar_loss + b * self.val_scalar_loss
                    self.val_scalar_loss = val_scalar_loss
                val_loss = 0.2 * val_scalar_loss + 0.8 * val_vector_loss
                self.log("val_loss", val_loss, batch_size=nb)
            elif "scalar" in val_loss_dict:
                self.log("val_loss", val_loss_dict["scalar"].item(), batch_size=nb)
            elif "vector" in val_loss_dict:
                self.log("val_loss", val_loss_dict["vector"].item(), batch_size=nb)

    def configure_optimizers(self) -> Dict:
        optimizer = Adam(self.parameters(), 1e-8, amsgrad=False)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            self.hparams.lr_scheduler_factor,
            self.hparams.lr_scheduler_patience,
            min_lr=1e-8,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": self.hparams.lr_scheduler_interval,
            "monitor": "val_loss",
            "frequency": self.hparams.lr_scheduler_frequency,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs) -> None:
        optimizer: Adam = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        # warn-up step
        if self.trainer.global_step < self.hparams.lr_warmup_step:
            lr_scale = int(self.trainer.global_step + 1) / self.hparams.lr_warmup_step
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.max_lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad(set_to_none=True)

    def export_model(self, workdir: Path) -> None:
        """
        Export the trained model.

        :param workdir: the directory where the model weights will be stored.
        """
        torch.save(
            {
                "nn": self.cmret.state_dict(),
                "args": self.cmret.args,
                "unit": self.hparams.model_unit,
            },
            f=workdir / "trained.pt",
        )
