# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Lightning wrapper of CMRET model.
"""
from typing import Dict, Any, Optional
from lightning import LightningModule
import torch.nn as nn
from torch import Tensor
from ..model import CMRETModel
from ..utils.tools import loss_calc


class CMRET4Training(LightningModule):
    def __init__(self, cmret: CMRETModel, hparam: Dict[str, Any]) -> None:
        """
        A `~lightning.LightningModule` wrapper of CMRET model.\n
        This model is for training only. After training, calling `CMRET4Training(...).export_model(YOUR_WORKDIR)`
        will save the trained model to `YOUR_WORKDIR/trained.pt`, which can be loaded later by calling
        `~cmret.CMRETModel.from_checkpoint(...)`

        :param cmret: `~cmret.model.representation.CMRETModel` instance
        :param hparam: training hyperparameters
        """
        self.cmret = cmret
        self.mse_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()
        self.val_vector_loss: Optional[float] = None
        self.save_hyperparameters(hparam)
        self.cmret.unit = hparam["unit"]