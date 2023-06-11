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
        s, v, r, batch_mask = kargv["s"], kargv["v"], kargv["r"], kargv["batch"]
        for layer in self.block:
            s, v = layer(s, v)
        s = self.out(s)
        y = (s.repeat(batch_mask.shape[0], 1, 1) * batch_mask).sum(dim=-2)
        out = {"scalar": y}
        if self.dy:
            dy = grad(
                outputs=y,
                inputs=r,
                grad_outputs=torch.ones_like(y),
                retain_graph=self.training,
                create_graph=self.training,
            )[0]
            out["vector"] = -dy
        if self.return_v:
            out["v"] = v.mean(dim=-1)
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
        self.s = nn.Linear(in_features=n_feature, out_features=1, bias=False)
        self.v = nn.Linear(in_features=n_feature, out_features=1, bias=False)

    def forward(self, **kargv: Tensor) -> Dict[str, Tensor]:
        z, s, v, r = kargv["z"], kargv["s"], kargv["v"], kargv["r"]
        batch_mask = kargv["batch"]
        n_b = batch_mask.shape[0]  # batch-size
        r = r.repeat(n_b, 1, 1) * batch_mask
        z = z.unsqueeze(dim=-1).repeat(n_b, 1, 1) * batch_mask
        centre = (z * r).sum(dim=-2, keepdim=True) / z.sum(dim=-2, keepdim=True)
        for layer in self.block:
            s, v = layer(s, v)
        s, v = self.s(s), self.v(v).squeeze(dim=-1)
        v = v.repeat(n_b, 1, 1) * batch_mask
        mu = (v + s * (r - centre)).sum(dim=1, keepdim=True)
        mu = torch.linalg.norm(mu, 2, -1)
        return {"scalar": mu}


class EquivariantPolarizability(nn.Module):
    def __init__(self, n_feature: int = 128, n_output: int = 2) -> None:
        """
        Equivariant polarizability output block.

        :param n_feature: input feature
        :param n_output: number of output layers
        """
        super().__init__()
        self.block = nn.ModuleList(
            [GatedEquivariant(n_feature=n_feature) for _ in range(n_output)]
        )
        self.s = nn.Linear(in_features=n_feature, out_features=1, bias=False)
        self.v = nn.Linear(in_features=n_feature, out_features=1, bias=False)

    def forward(self, **kargv: Tensor) -> Dict[str, Tensor]:
        z, s, v, r = kargv["z"], kargv["s"], kargv["v"], kargv["r"]
        batch_mask = kargv["batch"]
        n_b = batch_mask.shape[0]
        r = r.repeat(n_b, 1, 1) * batch_mask
        z = z.unsqueeze(dim=-1).repeat(n_b, 1, 1) * batch_mask
        centre = (z * r).sum(dim=-2, keepdim=True) / z.sum(dim=-2, keepdim=True)
        for layer in self.block:
            s, v = layer(s, v)
        s, v = self.s(s), self.v(v)
        v = v.repeat(n_b, 1, 1, 1) * batch_mask[:, :, :, None]
        r = (r - centre)[:, :, :, None]
        eye = torch.eye(3, device=s.device)[None, None, :, :]
        alpha = (
            (s.repeat(n_b, 1, 1) * batch_mask).unsqueeze(dim=-1) * eye
            + v @ r.transpose(-1, -2)
            + r @ v.transpose(-1, -2)
        ).sum(dim=1, keepdim=True)
        alpha = torch.linalg.matrix_norm(alpha)
        return {"scalar": alpha}


class ElectronicSpatial(nn.Module):
    def __init__(self, n_feature: int = 128, n_output: int = 2) -> None:
        """
        Electronic spatial extent output block.

        :param n_feature: input feature
        :param n_output: number of output layers
        """
        super().__init__()
        self.block = nn.ModuleList(
            [GatedEquivariant(n_feature=n_feature) for _ in range(n_output)]
        )
        self.s = nn.Linear(in_features=n_feature, out_features=1, bias=False)

    def forward(self, **kargv: Tensor) -> Dict[str, Tensor]:
        z, s, v, r = kargv["z"], kargv["s"], kargv["v"], kargv["r"]
        batch_mask = kargv["batch"]
        n_b = batch_mask.shape[0]
        z = z.unsqueeze(dim=-1).repeat(n_b, 1, 1) * batch_mask
        centre = (z * r).sum(dim=-2, keepdim=True) / z.sum(dim=-2, keepdim=True)
        for layer in self.block:
            s, v = layer(s, v)
        s = self.s(s)
        y = s.repeat(n_b, 1, 1) * batch_mask
        y = (y * (r - centre).pow(2).sum(dim=-1, keepdim=True)).sum(dim=-2)
        return {"scalar": y}
