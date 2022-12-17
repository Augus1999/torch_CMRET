# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model representation
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from .embedding import Embedding
from .output import EquivariantDipoleMoment, EquivarientScalar
from .module import Interaction, Distance, CosinCutOff, RBF1, RBF2, RBF3


__all__ = ["CMRETModel"]


class CMRET(nn.Module):
    def __init__(
        self,
        output: nn.Module,
        cutoff: float = 5.0,
        n_kernel: int = 20,
        n_atom_basis: int = 128,
        n_interaction: int = 5,
        rbf_type: str = "gaussian",
        num_head: int = 1,
        attention_activation: str = "softmax",
        simplified_cfconv: bool = True,
        dy: bool = True,
    ) -> None:
        """
        CMRET upper representaton.

        :param output: output model
        :param cutoff: cutoff radius
        :param n_kernel: number of RBF kernels
        :param n_atom_basis: number of atomic basis
        :param n_interaction: number of interaction blocks
        :param rbf_type: type of rbf basis: 'bessel', 'gaussian' or 'spherical'
                         choosing 'spherical' to allow higher order MP (L = 3)
        :param num_head: number of attention head per layer
        :param attention_activation: attention activation function type
        :param simplified_cfconv: whether using simplified CFConv scheme
        :param dy: whether calculater -dy
        """
        super().__init__()
        self.dy = dy
        self.n = n_atom_basis
        self.sim_cfconv = simplified_cfconv
        self.embedding = Embedding(embedding_dim=n_atom_basis)
        self.distance = Distance()
        if rbf_type == "bessel":
            self.rbf = RBF1(cell=cutoff, n_kernel=n_kernel)
        elif rbf_type == "gaussian":
            self.rbf = RBF2(cell=cutoff, n_kernel=n_kernel)
        elif rbf_type == "spherical":
            self.rbf = RBF3(cell=cutoff, n_kernel=n_kernel)
        else:
            raise NotImplementedError
        self.cutoff = CosinCutOff(cutoff=cutoff)
        self.interaction = nn.ModuleList(
            [
                Interaction(
                    n_feature=n_atom_basis,
                    n_kernel=n_kernel,
                    num_head=num_head,
                    attention_activation=attention_activation,
                )
                for _ in range(n_interaction)
            ]
        )
        self.norm = nn.LayerNorm(normalized_shape=n_atom_basis)
        self.out = output

    def forward(self, mol: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;      shape: (n_b, n_a)
            "R": nuclear coordinates tensor;  shape: (n_b, n_a, 3)
            "Q": total charge tensor;         shape: (n_b, 1) which is optional
            "S": spin state tensor;           shape: (n_b, 1) which is optional
        }
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        z, r = mol["Z"], mol["R"]
        r.requires_grad = self.dy
        v = torch.zeros_like(r).unsqueeze(dim=-1).repeat(1, 1, 1, self.n)
        if "Q" in mol:
            v -= (mol["Q"] / z.sum(dim=-1, keepdim=True))[:, :, None, None]
        if "S" in mol:
            v += (mol["S"] / z.sum(dim=-1, keepdim=True))[:, :, None, None]
        if self.sim_cfconv:
            loop_mask = None
        else:
            loop_mask = torch.eye(z.shape[-1], device=z.device)
            loop_mask = loop_mask[None, :, :].repeat(z.shape[0], 1, 1) == 0
        d, d_vec = self.distance(r)
        cutoff, mask = self.cutoff(d)
        cutoff, mask = cutoff.unsqueeze(dim=-1), mask.unsqueeze(dim=-1)
        e_ = self.rbf(d=d, d_vec=d_vec)
        if e_.dim() == 4:
            e = cutoff * e_
        else:
            e = cutoff.unsqueeze(dim=-2) * e_
        d_vec_norm = (mask * d_vec / d.unsqueeze(dim=-1)).unsqueeze(dim=-1)
        s = self.embedding(z)
        s_o = 0
        for layer in self.interaction:
            s, o, v = layer(s, v, e, d_vec_norm, mask, loop_mask)
            s_o += o
        s_o = self.norm(s_o)
        return self.out(z=z, s=s_o, v=v, r=r)


class CMRETModel(nn.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        n_kernel: int = 20,
        n_atom_basis: int = 128,
        n_interaction: int = 5,
        n_output: int = 2,
        rbf_type: str = "gaussian",
        num_head: int = 1,
        attention_activation: str = "softmax",
        simplified_cfconv: bool = True,
        output_mode: str = "energy-force",
    ) -> None:
        """
        CMRET model.

        :param cutoff: cutoff radius
        :param n_kernel: number of RBF kernels
        :param n_atom_basis: number of atomic basis
        :param n_interaction: number of interaction blocks
        :param n_output: number of output blocks
        :param rbf_type: type of rbf basis: 'bessel', 'gaussian' or 'spherical'
                         choosing 'spherical' to allow higher order MP (L = 3)
        :param num_head: number of attention head per layer
        :param attention_activation: attention activation function type
        :param simplified_cfconv: whether using simplified CFConv scheme
        :param output_mode: output properties
        """
        super().__init__()
        dy = False
        self.unit: Optional[str] = None
        if output_mode == "energy-force":
            out = EquivarientScalar(n_feature=n_atom_basis, n_output=n_output, dy=True)
            dy = True
        elif output_mode in ("energy", "HOMO", "LUMO"):
            out = EquivarientScalar(n_feature=n_atom_basis, n_output=n_output, dy=False)
        elif output_mode == "dipole moment":
            out = EquivariantDipoleMoment(n_feature=n_atom_basis, n_output=n_output)
        self.model = CMRET(
            output=out,
            cutoff=cutoff,
            n_kernel=n_kernel,
            n_atom_basis=n_atom_basis,
            n_interaction=n_interaction,
            rbf_type=rbf_type,
            num_head=num_head,
            attention_activation=attention_activation,
            simplified_cfconv=simplified_cfconv,
            dy=dy,
        )

    def forward(self, mol: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;      shape: (n_b, n_a)
            "R": nuclear coordinates tensor;  shape: (n_b, n_a, 3)
            "Q": total charge tensor;         shape: (n_b, 1) which is optional
            "S": spin state tensor;           shape: (n_b, 1) which is optional
        }
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        return self.model(mol)

    def pretrained(self, file: Optional[str]) -> nn.Module:
        if file:
            with open(file, mode="rb") as f:
                state_dict = torch.load(f, map_location="cpu")
            self.load_state_dict(state_dict=state_dict["nn"])
            self.unit = state_dict["unit"]
        return self

    @property
    def check_parameter_number(self) -> int:
        """
        Count the number trainable of parameters.

        :return: n_parameters
        """
        number = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return number
