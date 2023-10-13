# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Pre-trained models
"""
from pathlib import Path
import torch

pretrain = Path(__file__).parent
names = {
    "md17": pretrain / "md17.h5",
    "coll": pretrain / "coll.h5",
    "ccsd": pretrain / "ani1ccx.h5",
    "iso17": pretrain / "iso17.h5",
    "dimer": pretrain / "des370k.h5",
    "ani1x": pretrain / "ani1x.h5",
}


def trained_model(name: str) -> torch.nn.Module:
    if name in names.keys():
        return torch.load(names[name], map_location="cpu")
    else:
        raise NotImplementedError(
            f"No model named '{name}' in '{pretrain}'. You can train your own model and put it here.\n"
            f"Recently available models are {list(names.keys())}."
        )


__all__ = ["trained_model"]
