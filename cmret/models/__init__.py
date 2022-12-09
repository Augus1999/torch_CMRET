# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Pre-trained models
"""
from pathlib import Path
import torch

pretrain = Path(__file__).parent
names = {
    "carbene": pretrain / "carbene.h5",
    "acetylacetone": pretrain / "acac.h5",
}


def trained_model(name: str) -> torch.nn.Module:
    if name in names.keys():
        return torch.load(names[name], map_location="cpu")
    else:
        raise NotImplementedError(
            f"no model named '{name}' in '{pretrain}'. You can train your own model and put it here."
        )


__all__ = ["trained_model"]
