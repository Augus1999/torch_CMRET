# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Pre-trained models.
"""
from pathlib import Path
from ..model import CMRETModel

pretrain = Path(__file__).parent
names = {
    "md17": pretrain / "md17.pt",
    "coll": pretrain / "coll.pt",
    "ccsd": pretrain / "ani1ccx.pt",
    "iso17": pretrain / "iso17.pt",
    "dimer": pretrain / "des370k.pt",
    "ani1x": pretrain / "ani1x.pt",
}


def trained_model(name: str) -> CMRETModel:
    if name in names.keys():
        return CMRETModel.from_checkpoint(names[name])
    else:
        raise NotImplementedError(
            f"No model named '{name}' in '{pretrain}'. You can train your own model and put it here.\n"
            f"Recently available models are {list(names.keys())}."
        )


__all__ = ["trained_model"]
