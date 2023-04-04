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
        n_interaction: int = 6,
        rbf_type: str = "gaussian",
        num_head: int = 4,
        temperature_coeff: float = 2.0,
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
        :param temperature_coeff: temperature coefficient
        :param dy: whether calculater -dy
        """
        super().__init__()
        self.dy = dy
        self.n = n_atom_basis
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
                    temperature_coeff=temperature_coeff,
                )
                for _ in range(n_interaction)
            ]
        )
        self.norm = nn.LayerNorm(normalized_shape=n_atom_basis)
        self.out = output

    def forward(
        self,
        mol: Dict[str, Tensor],
        return_attn_matrix: bool = False,
        average_attn_matrix_over_layers: bool = True,
    ) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;      shape: (1, n_a)
            "R": nuclear coordinates tensor;  shape: (1, n_a, 3)
            "batch": batch mask;              shape: (n_b, n_a, 1)
            "Q": total charge tensor;         shape: (n_b, 1) which is optional
            "S": spin state tensor;           shape: (n_b, 1) which is optional
        }
        :param return_attn_matrix: whether to return the attention matrices
        :param average_attn_matrix_over_layers: whether to average the attention matrices over layers
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        z, r, batch = mol["Z"], mol["R"], mol["batch"]
        r.requires_grad = self.dy
        v = torch.zeros_like(r).unsqueeze(dim=-1).repeat(1, 1, 1, self.n)
        if "Q" in mol:
            q_info = mol["Q"]
            z_info = z.repeat(q_info.shape[0], 1) * batch.squeeze(dim=-1)
            q_info = q_info / z_info.sum(dim=-1, keepdim=True)
            v_ = v.repeat(q_info.shape[0], 1, 1, 1) * batch.unsqueeze(dim=-1)
            v_ = v_ - q_info[:, :, None, None]
            v = v_[batch.squeeze(-1) != 0].view(v.shape)
        if "S" in mol:
            s_info = mol["S"]
            z_info = z.repeat(s_info.shape[0], 1) * batch.squeeze(dim=-1)
            s_info = s_info / z_info.sum(dim=-1, keepdim=True)
            v_ = v.repeat(s_info.shape[0], 1, 1, 1) * batch.unsqueeze(dim=-1)
            v_ = v_ + s_info[:, :, None, None]
            v = v_[batch.squeeze(-1) != 0].view(v.shape)
        # --------- compute loop mask that removes the self-loop ----------------
        loop_mask = torch.eye(z.shape[-1], device=z.device)
        loop_mask = loop_mask[None, :, :] == 0
        # -----------------------------------------------------------------------
        s = self.embedding(z)
        o = torch.zeros_like(s)
        # ---- compute batch mask that seperates atoms in different molecules ----
        batch_mask_ = batch.squeeze(-1).transpose(-2, -1) @ batch.squeeze(-1)
        batch_mask = batch_mask_[None, :, :, None]
        batch_mask_ = batch_mask_ == 0
        # ------------------------------------------------------------------------
        d, d_vec = self.distance(r, batch_mask, loop_mask)
        cutoff, mask = self.cutoff(d)
        cutoff, mask = cutoff.unsqueeze(dim=-1), mask.unsqueeze(dim=-1)
        h = batch_mask.shape[1]
        cutoff_mask = batch_mask[loop_mask].view(1, h, h - 1, 1)
        cutoff, mask = cutoff * cutoff_mask, mask * cutoff_mask
        e = self.rbf(d=d, d_vec=d_vec)
        if e.dim() == 4:
            e = cutoff * e
        else:
            e = cutoff.unsqueeze(dim=-2) * e
        _d = d.masked_fill(d == 0, torch.inf)
        d_vec_norm = (mask * d_vec / _d.unsqueeze(dim=-1)).unsqueeze(dim=-1)
        attn = []
        for layer in self.interaction:
            s, o, v, _attn = layer(
                s, o, v, e, d_vec_norm, mask, loop_mask, batch_mask_, return_attn_matrix
            )
            if return_attn_matrix:
                if average_attn_matrix_over_layers:
                    attn.append(_attn.unsqueeze(dim=0))
                else:
                    attn.append(_attn)
        o = self.norm(o)
        out = self.out(z=z, s=o, v=v, r=r, batch=batch)
        if return_attn_matrix:
            if average_attn_matrix_over_layers:
                attn = torch.cat(attn, dim=0).mean(dim=0)
            out["attn_matrix"] = attn
        return out


class CMRETModel(nn.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        n_kernel: int = 20,
        n_atom_basis: int = 128,
        n_interaction: int = 6,
        n_output: int = 2,
        rbf_type: str = "gaussian",
        num_head: int = 4,
        temperature_coeff: float = 2.0,
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
        :param temperature_coeff: temperature coefficient
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
        elif output_mode == "pretrain":
            out = EquivarientScalar(
                n_feature=n_atom_basis,
                n_output=n_output,
                dy=dy,
                return_vector_feature=True,
            )
        else:
            raise NotImplementedError(f"{output_mode} is not defined.")
        self.model = CMRET(
            output=out,
            cutoff=cutoff,
            n_kernel=n_kernel,
            n_atom_basis=n_atom_basis,
            n_interaction=n_interaction,
            rbf_type=rbf_type,
            num_head=num_head,
            temperature_coeff=temperature_coeff,
            dy=dy,
        )

    def forward(
        self,
        mol: Dict[str, Tensor],
        return_attn_matrix: bool = False,
        average_attn_matrix_over_layers: bool = True,
    ) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;      shape: (1, n_a)
            "R": nuclear coordinates tensor;  shape: (1, n_a, 3)
            "batch": batch mask;              shape: (n_b, n_a, 1)
            "Q": total charge tensor;         shape: (n_b, 1) which is optional
            "S": spin state tensor;           shape: (n_b, 1) which is optional
        }
        :param return_attn_matrix: whether to return the attention matrices
        :param average_attn_matrix_over_layers: whether to average the attention matrices over layers
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        return self.model(
            mol,
            return_attn_matrix,
            average_attn_matrix_over_layers,
        )

    def pretrained(self, file: Optional[str]) -> nn.Module:
        """
        Load pre-trained weight.

        :param file: model file name <file>
        :return: model
        """
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
