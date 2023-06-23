# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CMRET nn modules
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor


class RBF1(nn.Module):
    def __init__(self, cell: float = 5.0, n_kernel: int = 20) -> None:
        """
        Bessel RBF kernel for 3D input.

        :param cell: unit cell length
        :param n_kernel: number of kernels
        """
        super().__init__()
        self.register_buffer("cell", torch.tensor([cell]))
        offsets = torch.linspace(1.0, n_kernel, n_kernel)
        self.register_buffer("offsets", offsets[None, None, None, :])

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (1, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (1, n_a, n_a - 1, n_k)
        """
        out = (
            torch.pi * self.offsets * d[:, :, :, None] / self.cell
        ).sin() / d.masked_fill(d == 0, torch.inf)[:, :, :, None]
        return out


class RBF2(nn.Module):
    def __init__(self, cell: float = 5.0, n_kernel: int = 20) -> None:
        """
        Gaussian RBF kernel for 3D input.

        :param cell: unit cell length
        :param n_kernel: number of kernels
        """
        super().__init__()
        self.register_buffer("cell", torch.tensor([cell]))
        self.register_buffer("n_kernel", torch.tensor([n_kernel]))
        offsets = torch.linspace((-self.cell).exp().item(), 1, n_kernel)[
            None, None, None, :
        ]
        coeff = ((1 - (-self.cell).exp()) / n_kernel).pow(-2) * torch.ones_like(offsets)
        self.offsets = nn.Parameter(offsets, requires_grad=True)
        self.coeff = nn.Parameter(coeff / 4, requires_grad=True)

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (1, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (1, n_a, n_a - 1, n_k)
        """
        out = (-self.coeff * ((-d[:, :, :, None]).exp() - self.offsets).pow(2)).exp()
        return out


class CosinCutOff(nn.Module):
    def __init__(self, cutoff: float = 5.0) -> None:
        """
        Compute cosin-cutoff mask.

        :param cutoff: cutoff radius
        """
        super().__init__()
        self.register_buffer("cutoff", Tensor([cutoff]))

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: pair-wise distances;     shape: (1, n_a, n_a - 1)
        :return: cutoff mask;              shape: (1, n_a, n_a - 1)
        """
        cutoff = 0.5 * (torch.pi * d / self.cutoff).cos() + 0.5
        cutoff *= (d <= self.cutoff).float()
        return cutoff


class Distance(nn.Module):
    def __init__(self) -> None:
        """
        Compute pair-wise distances and dist_vec.
        """
        super().__init__()

    def forward(
        self, r: Tensor, batch_mask: Tensor, loop_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        :param r: nuclear coordinates;  shape: (1, n_a, 3)
        :param batch_mask: batch mask;  shape: (1, n_a, n_a, 1)
        :param loop_mask: loop mask;    shape: (1, n_a, n_a)
        :return: d, d_vec;              shape: (1, n_a, n_a - 1), (1, n_a, n_a - 1, 3)
        """
        n_b, n_a, _ = r.shape
        d_vec = r[:, :, None, :] - r[:, None, :, :]
        d_vec = d_vec * batch_mask
        # remove 0 vectors
        d_vec = d_vec[loop_mask].view(n_b, n_a, n_a - 1, 3)
        d = torch.linalg.norm(d_vec, 2, -1)
        return d, d_vec


class ResML(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        """
        Residual layer.

        :param dim: input dimension
        """
        super().__init__()
        self.res = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
        )
        nn.init.normal_(
            self.res[0].weight, 0, 0.5 * (6 / dim) ** 0.5
        )  # weight init: N[0.0, (3/(2 dim))^1/2]

    def forward(self, x: Tensor) -> Tensor:
        return self.res(x) + x


class CFConv(nn.Module):
    def __init__(self, n_kernel: int = 20, n_feature: int = 128) -> None:
        """
        Improved Contiunous-filter convolution block.

        :param n_kernel: number of RBF kernels
        :param n_feature: number of feature dimensions
        """
        super().__init__()
        self.n_feature = n_feature
        self.w1 = nn.Sequential(
            nn.Linear(in_features=n_kernel, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.w2 = nn.Sequential(
            nn.Linear(in_features=n_kernel, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.o = nn.Linear(
            in_features=n_feature * 2, out_features=n_feature * 3, bias=True
        )
        nn.init.uniform_(
            self.w1[0].weight, -((6 / n_kernel) ** 0.5), (6 / n_kernel) ** 0.5
        )  # weight init: U[-(6/in_dim)^1/2, (6/in_dim)^1/2]
        nn.init.uniform_(
            self.w2[0].weight, -((6 / n_kernel) ** 0.5), (6 / n_kernel) ** 0.5
        )  # weight init: U[-(6/in_dim)^1/2, (6/in_dim)^1/2]
        nn.init.normal_(
            self.phi[0].weight, 0, 0.5 * (6 / n_feature) ** 0.5
        )  # weight init: N[0.0, (3/(2 dim))^1/2]

    def forward(
        self, x: Tensor, e: Tensor, mask: Tensor, loop_mask: Tensor
    ) -> Tuple[Tensor]:
        """
        :param x: input info;              shape: (1, n_a, n_f)
        :param e: extended tensor;         shape: (1, n_a, n_a - 1, n_k)
        :param mask: neighbour mask;       shape: (1, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;  shape: (1, n_a, n_a)
        :return: convoluted scalar info;   shape: (1, n_a, n_a - 1, n_f) * 3
        """
        w1, w2 = self.w1(e), self.w2(e)
        x = self.phi(x)
        _, n_a, f = x.shape
        x_nbs = x[:, None, :, :].repeat(1, n_a, 1, 1)
        x_nbs = x_nbs[loop_mask].view(1, n_a, n_a - 1, f)
        v1 = x[:, :, None, :] * w1 * mask
        v2 = x_nbs * w2 * mask
        v = self.o(torch.cat([v1, v2], dim=-1)) * mask
        s1, s2, s3 = torch.split(
            v,
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        return s1, s2, s3


class NonLoacalInteraction(nn.Module):
    def __init__(
        self, n_feature: int = 128, num_head: int = 4, temperature_coeff: float = 2.0
    ) -> None:
        """
        NonLoacalInteraction block (single/multi-head self-attention).

        :param n_feature: number of feature dimension
        :param num_head: number of attention head
        :param temperature_coeff: temperature coefficient
        """
        super().__init__()
        assert (
            num_head > 0 and n_feature % num_head == 0
        ), f"Cannot split {num_head} attention heads when feature is {n_feature}."
        self.d = n_feature // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (temperature_coeff * self.d) ** 0.5  # attention temperature
        self.multi = num_head > 1
        self.q = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.k = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.v = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.activate = nn.Softmax(dim=-1)

    def forward(
        self, x: Tensor, batch_mask: Tensor, return_attn_matrix: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: input tensor;            shape: (1, n_a, n_f)
        :param batch_mask: batch mask;     shape: (n_a, n_a)
        :param return_attn_matrix: whether to return the attenton matrix
        :return: attention-scored output;  shape: (1, n_a, n_f)
                 attention matrix;         shape: (1, n_a, n_a)
        """
        _, n_a, n_f = x.shape
        q = self.q(x).view(1, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(1, n_a, self.nh, self.d).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(1, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        alpha = self.activate(a.masked_fill(batch_mask, -torch.inf))
        out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(1, n_a, n_f)
        if return_attn_matrix:
            return out, alpha.mean(dim=0)
        return out, None


class Interaction(nn.Module):
    def __init__(
        self,
        n_feature: int = 128,
        n_kernel: int = 50,
        num_head: int = 4,
        temperature_coeff: float = 2.0,
    ) -> None:
        """
        Interaction block.

        :param n_feature: number of feature dimension
        :param n_kernel: number of RBF kernels
        :param num_head: number of attention head
        :param temperature_coeff: temperature coefficient
        """
        super().__init__()
        self.n_feature = n_feature
        self.cfconv = CFConv(n_kernel=n_kernel, n_feature=n_feature)
        self.nonloacl = NonLoacalInteraction(
            n_feature=n_feature, num_head=num_head, temperature_coeff=temperature_coeff
        )
        self.u = nn.Linear(
            in_features=n_feature, out_features=n_feature * 3, bias=False
        )
        self.o = nn.Linear(in_features=n_feature, out_features=n_feature * 3, bias=True)
        self.res = ResML(dim=n_feature)

    def forward(
        self,
        s: Tensor,
        o: Tensor,
        v: Tensor,
        e: Tensor,
        d_vec_norm: Tensor,
        mask: Tensor,
        loop_mask: Tensor,
        batch_mask: Tensor,
        return_attn_matrix: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        :param s: scale info;                 shape: (1, n_a, n_f)
        :param o: scale from pervious layer;  shape: (1, n_a, n_f)
        :param v: vector info;                shape: (1, n_a, 3, n_f)
        :param e: rbf extended distances;     shape: (1, n_a, n_a - 1, n_k)
        :param d_vec_norm: normalised d_vec;  shape: (1, n_a, n_a - 1, 3, 1)
        :param mask: neighbour mask;          shape: (1, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;     shape: (1, n_a, n_a)
        :param batch_mask: batch mask;        shape: (n_a, n_a)
        :param return_attn_matrix: whether to return the attenton matrix
        :return: new scale & output scale & vector info & attention matrix
        """
        s1, s2, s3 = self.cfconv(s, e, mask, loop_mask)
        v1, v2, v3 = torch.split(
            self.u(v),
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        s1_sum = s1.sum(dim=-2) if s1.dim() == 4 else s1.sum(dim=[-3, -2])
        s_nonlocal, attn_matrix = self.nonloacl(s, batch_mask, return_attn_matrix)
        s_n1, s_n2, s_n3 = torch.split(
            self.o(s + s_nonlocal + s1_sum),
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        s_m = s_n1 + s_n2 * (v1 * v2).sum(dim=-2)
        s_out = self.res(s_m) + o
        v = v[:, :, None, :, :]
        v_m = s_n3[:, :, None, :] * v3 + (
            s2[:, :, :, None, :] * v + s3[:, :, :, None, :] * d_vec_norm
        ).sum(dim=-3)
        return s_m, s_out, v_m, attn_matrix


class GatedEquivariant(nn.Module):
    def __init__(self, n_feature: int = 128) -> None:
        """
        Gated equivariant block.

        :param n_feature: number of feature dimension
        """
        super().__init__()
        self.n_feature = n_feature
        self.u = nn.Linear(in_features=n_feature, out_features=n_feature, bias=False)
        self.v = nn.Linear(in_features=n_feature, out_features=n_feature, bias=False)
        self.a = nn.Sequential(
            nn.Linear(in_features=n_feature * 2, out_features=n_feature, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=n_feature, out_features=n_feature * 2, bias=True),
        )

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor]:
        """
        :param s: scale info;   shape: (1, n_a, n_f)
        :param v: vector info;  shape: (1, n_a, 3, n_f)
        :return: updated s, updated v
        """
        v1, v2 = self.u(v), self.v(v)
        # add 1e-8 to avoid nan in the gradient of gradient in some extreme cases
        s0 = self.a(torch.cat([s, torch.linalg.norm(v2 + 1e-8, 2, -2)], dim=-1))
        sg, ss = torch.split(s0, split_size_or_sections=self.n_feature, dim=-1)
        vg = v1 * ss[:, :, None, :]
        return sg, vg


if __name__ == "__main__":
    ...
