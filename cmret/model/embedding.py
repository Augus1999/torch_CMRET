# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CMRET embedding modules.
"""
from typing import List
import torch
import torch.nn as nn
from torch import Tensor, float32


ORBITALS = "1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p 6f 7d 7f".split()
POSSIBLE_ELECTRONS = dict(s=2, p=6, d=10, f=14)


def _electron_config(atomic_num: int) -> List[int]:
    """
    Generate electron configuration for a given atomic number.

    :param atomic_num: atomic number
    :return: electron configuration
    """
    electron_count, last_idx, config = 0, -1, []
    for i in ORBITALS:
        if electron_count < atomic_num:
            config.append(POSSIBLE_ELECTRONS[i[-1]])
            electron_count += POSSIBLE_ELECTRONS[i[-1]]
            last_idx += 1
        else:
            config.append(0)
    if electron_count > atomic_num:
        config[last_idx] -= electron_count - atomic_num
    return config


electron_config = torch.tensor([_electron_config(i) for i in range(119)], dtype=float32)


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Nuclear embedding block.

        :param embedding_dim: embedding dimension
        """
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(electron_config)
        self.out = nn.Linear(in_features=22, out_features=embedding_dim, bias=True)

    def forward(self, z: Tensor) -> Tensor:
        """
        :param z: nuclear charges;  shape: (1, n_a)
        :return: embedded tensor;   shape: (1, n_a, embedding_dim)
        """
        return self.out(self.embed(z))


if __name__ == "__main__":
    ...
