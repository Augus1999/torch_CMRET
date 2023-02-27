# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Output models
"""
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from .module import GatedEquivariant


class EquivarientScalar(nn.Module):
    def __init__(
        self,
        n_feature: int = 128,
        n_output: int = 2,
        dy: bool = True,
        return_vector_feature: bool = False,
    ) -> None:
        """
        Equivariant Scalar output block.

        :param n_feature: input feature
        :param n_output: number of output layers
        :param dy: whether to calculater -dy
        :param return_vector_feature: whether to return the vector features
        """
        super().__init__()
        self.dy = dy
        self.return_v = return_vector_feature
        self.block = nn.ModuleList(
            [GatedEquivariant(n_feature=n_feature) for _ in range(n_output)]
        )
        self.out = nn.Linear(in_features=n_feature, out_features=1, bias=True)

    def forward(self, **kargv: Tensor) -> Dict[str, Tensor]:
        s, v, r = kargv["s"], kargv["v"], kargv["r"]
        for layer in self.block:
            s, v = layer(s, v)
        s = self.out(s)
        y = s.sum(dim=-2)
        out = {"energy": y}
        if self.dy:
            dy = grad(
                outputs=y,
                inputs=r,
                grad_outputs=torch.ones_like(y),
                retain_graph=self.training,
                create_graph=self.training,
            )[0]
            out["force"] = -dy
        if self.return_v:
            out["v"] = v
        return out


class EquivariantDipoleMoment(nn.Module):
    def __init__(self, n_feature: int = 128, n_output: int = 2) -> None:
        """
        Equivariant dipole moment output block.

        :param n_feature: input feature
        :param n_output: number of output layers
        """
        super().__init__()
        self.block = nn.ModuleList(
            [GatedEquivariant(n_feature=n_feature) for _ in range(n_output)]
        )

    def forward(self, **kargv: Tensor) -> Dict[str, Tensor]:
        z, s, v, r = kargv["z"], kargv["s"], kargv["v"], kargv["r"]
        mass_centre = (z.unsqueeze(dim=-1) * r) / z.sum(dim=-1)[:, None, None]
        for layer in self.block:
            s, v = layer(s, v)
        mu = (v + s.unsequeeze(dim=-2) * (r - mass_centre).unsqueeze(dim=-1)).sum(dim=1)
        mu = torch.linalg.norm(mu, 2, -1)
        return {"dipole moment": mu}
