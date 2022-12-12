# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CMRET nn modules
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor


def softmax2d(x: Tensor) -> Tensor:
    """
    Applies Softmax over last two dimensions.

    :param x: input tensor;   shape: (n_b, H, W)
    :return: output tensor;   shape: (n_b, H, W)
    """
    assert x.dim() == 3, "input tensor must be 3D"
    n_b, dim1, dim2 = x.shape
    x = x.view(n_b, dim1 * dim2)
    x = nn.functional.softmax(x, dim=-1)
    return x.view(n_b, dim1, dim2)


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

    def forward(self, **karg) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (n_b, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (n_b, n_a, n_a - 1, n_k)
        """
        d = karg["d"]
        out = (
            torch.pi * self.offsets * d.unsqueeze(dim=-1) / self.cell
        ).sin() / d.unsqueeze(dim=-1)
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

    def forward(self, **karg) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (n_b, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (n_b, n_a, n_a - 1, n_k)
        """
        d = karg["d"]
        out = (-self.coeff * ((-d.unsqueeze(dim=-1)).exp() - self.offsets).pow(2)).exp()
        return out


class RBF3(nn.Module):
    def __init__(self, cell: float = 5.0, n_kernel: int = 20) -> None:
        """
        Spherical RBF kernel for 3D input.

        :param cell: unit cell length
        :param n_kernel: number of kernels
        """
        super().__init__()
        self.bessel = RBF1(cell=cell, n_kernel=n_kernel)
        self.sphere = Sphere()
        offsets = torch.linspace(1.0, 3.0, 3)
        self.register_buffer("offsets", offsets[None, None, None, :])

    def forward(self, **karg) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (n_b, n_a, n_a - 1)
        :param d_vec: pair-wise vector;   shape: (n_b, n_a, n_a - 1, 3)
        :return: RBF-extanded distances;  shape: (n_b, n_a, n_a - 1, 3, n_k)
        """
        d, d_vec = karg["d"], karg["d_vec"]
        d_e = self.bessel(d=d).unsqueeze(dim=-2)
        theta = self.sphere(d_vec)
        theta = theta.unsqueeze(dim=-1)
        harmonic = (
            ((2 * self.offsets + 1) / (4 * torch.pi)).sqrt()
            * (-theta.sin().pow(2)).pow(self.offsets)
            * (-theta.sin() * theta.cos()).pow(self.offsets)
        )
        return d_e * harmonic.unsqueeze(dim=-1)


class CosinCutOff(nn.Module):
    def __init__(self, cutoff: float = 5.0) -> None:
        """
        Compute cosin-cutoff mask.

        :param cutoff: cutoff radius
        """
        super().__init__()
        self.register_buffer("cutoff", Tensor([cutoff]))

    def forward(self, d: Tensor) -> Tuple[Tensor]:
        """
        :param d: pair-wise distances;     shape: (n_b, n_a, n_a - 1)
        :return: cutoff & neighbour mask;  shape: (n_b, n_a, n_a - 1)
        """
        cutoff = 0.5 * (torch.pi * d / self.cutoff).cos() + 0.5
        mask = (d <= self.cutoff).float()
        cutoff *= mask
        return cutoff, mask


class Distance(nn.Module):
    def __init__(self) -> None:
        """
        Compute pair-wise distances and dist_vec.
        """
        super().__init__()

    def forward(self, r: Tensor) -> Tuple[Tensor]:
        """
        :param r: nuclear coordinates;  shape: (n_b, n_a, 3)
        :return: d, d_vec;              shape: (n_b, n_a, n_a - 1), (n_b, n_a, n_a - 1, 3)
        """
        n_b, n_a, _ = r.shape
        d_vec = r.unsqueeze(dim=-2) - r.unsqueeze(dim=-3)
        # remove 0 vectors
        d_vec = d_vec[torch.linalg.norm(d_vec, 2, -1) != 0].view(n_b, n_a, n_a - 1, 3)
        d = torch.linalg.norm(d_vec, 2, -1)
        return d, d_vec


class Sphere(nn.Module):
    def __init__(self) -> None:
        """
        Compute spherical angles.
        """
        super().__init__()

    def forward(self, d_vec: Tensor) -> Tensor:
        """
        :param d_vec: vectors;  shape: (n_b, n_a, n_a - 1, 3)
        :return: theta values;  shape: (n_b, n_a, n_a - 1)
        """
        theta = torch.atan2(d_vec[..., 1], d_vec[..., 0])
        # phi = torch.acos(d_vec[..., 2] / d)
        return theta


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
        Contiunous-filter convolution block.

        :param n_kernel: number of RBF kernels
        :param n_feature: number of feature dimensions
        """
        super().__init__()
        self.n_feature = n_feature
        self.w = nn.Sequential(
            nn.Linear(in_features=n_kernel, out_features=n_feature * 3, bias=True),
            nn.SiLU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=n_feature, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=n_feature, out_features=n_feature * 3, bias=True),
        )
        nn.init.uniform_(
            self.w[0].weight, -((6 / n_kernel) ** 0.5), (6 / n_kernel) ** 0.5
        )  # weight init: U[-(6/in_dim)^1/2, (6/in_dim)^1/2]
        nn.init.normal_(
            self.phi[0].weight, 0, 0.5 * (6 / n_feature) ** 0.5
        )  # weight init: N[0.0, (3/(2 dim))^1/2]

    def forward(
        self, x: Tensor, e: Tensor, mask: Tensor, loop_mask: Optional[Tensor]
    ) -> Tuple[Tensor]:
        """
        :param x: input info;              shape: (n_b, n_a, n_f)
        :param e: extended tensor;         shape: (n_b, n_a, n_a - 1, n_k)
                                               or (n_b, n_a, n_a - 1, 3, n_k)
        :param mask: neighbour mask;       shape: (n_b, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;  shape: (n_b, n_a, n_a)
        :return: convoluted info;          shape: (n_b, n_a, n_a - 1, n_f) * 3
                                               or (n_b, n_a, n_a - 1, 3, n_f) * 3
        """
        w = self.w(e)
        x = self.phi(x)
        if loop_mask is None:
            # simplified CFConv
            if w.dim() == 4:
                v = x.unsqueeze(dim=-2) * w * mask
            else:
                v = x[:, :, None, None, :] * w * mask.unsqueeze(dim=-2)
        else:
            # CFConv: loop_mask helps to remove self-loop
            n_b, n_a, f = x.shape
            x_nbs = x.unsqueeze(dim=-3).repeat(1, n_a, 1, 1)
            x_nbs = x_nbs[loop_mask == 0].view(n_b, n_a, n_a - 1, f)
            if w.dim() == 4:
                v = x_nbs * w * mask
            else:  # higher order L = 3
                v = x_nbs.unsqueeze(dim=-2) * w * mask.unsqueeze(dim=-2)
        s1, s2, s3 = torch.split(
            v,
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        return s1, s2, s3


class NonLoacalInteraction(nn.Module):
    def __init__(
        self, n_feature: int = 128, num_head: int = 1, activation: str = "softmax"
    ) -> None:
        """
        NonLoacalInteraction block (single/multi-head self-attention).

        :param n_feature: number of feature dimension
        :param num_head: number of attention head
        :param activation: activation function type
        """
        super().__init__()
        assert (
            num_head > 0 and n_feature % num_head == 0
        ), f"Cannot split {num_head} attention heads when feature is {n_feature}."
        activate_funs = {
            "swish": nn.SiLU(),
            "softplus": nn.Softplus(),
            "softmax": nn.Softmax(dim=-1),
            "softmax2d": softmax2d,
        }
        assert activation.lower() in activate_funs.keys()
        self.temp = (
            (2 * n_feature) ** 0.5
            if "softmax" in activation.lower()
            else n_feature**0.5
        )  # define attention temperature
        self.multi = num_head > 1
        self.q = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.k = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.v = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        if self.multi:
            self.activate = nn.MultiheadAttention(
                embed_dim=128, num_heads=num_head, batch_first=True
            )
        else:
            self.activate = activate_funs[activation.lower()]

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor;            shape: (n_b, n_a, n_f)
        :return: attention-scored output;  shape: (n_b, n_a, n_f)
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        if self.multi:
            out, alpha = self.activate(query, key, value)
        else:
            alpha = self.activate(query @ key.transpose(-2, -1) / self.temp)
            out = alpha @ value
        return out


class Interaction(nn.Module):
    def __init__(
        self,
        n_feature: int = 128,
        n_kernel: int = 20,
        num_head: int = 1,
        attention_activation: str = "softmax",
    ) -> None:
        """
        Interaction block.

        :param n_feature: number of feature dimension
        :param n_kernel: number of RBF kernels
        :param num_head: number of attention head
        :param attention_activation: attention activation function type
        """
        super().__init__()
        self.n_feature = n_feature
        self.cfconv = CFConv(n_kernel=n_kernel, n_feature=n_feature)
        self.nonloacl = (
            NonLoacalInteraction(
                n_feature=n_feature, num_head=num_head, activation=attention_activation
            )
            if num_head > 0
            else lambda _: 0
        )
        self.u = nn.Linear(
            in_features=n_feature, out_features=n_feature * 3, bias=False
        )
        self.o = nn.Linear(in_features=n_feature, out_features=n_feature * 3, bias=True)
        self.res = ResML(dim=n_feature)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        e: Tensor,
        d_vec_norm: Tensor,
        mask: Tensor,
        loop_mask: Optional[Tensor],
    ) -> Tuple[Tensor]:
        """
        :param s: scale info;                 shape: (n_b, n_a, n_f)
        :param v: vector info;                shape: (n_b, n_a, 3, n_f)
        :param e: rbf extended distances;     shape: (n_b, n_a, n_a - 1, n_k)
        :param d_vec_norm: normalised d_vec;  shape: (n_b, n_a, n_a - 1, 3, 1)
        :param mask: neighbour mask;          shape: (n_b, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;     shape: (n_b, n_a, n_a)
        :return: new scale & output scale & vector info
        """
        s1, s2, s3 = self.cfconv(s, e, mask, loop_mask)
        v1, v2, v3 = torch.split(
            self.u(v),
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        s1_sum = s1.sum(dim=-2) if s1.dim() == 4 else s1.sum(dim=[-3, -2])
        s_n1, s_n2, s_n3 = torch.split(
            self.o(s + self.nonloacl(s) + s1_sum),
            split_size_or_sections=self.n_feature,
            dim=-1,
        )
        s_m = s_n1 + s_n2 * (v1 * v2).sum(dim=-2)
        s_out = self.res(s_m)
        if loop_mask is None:
            v = v.unsqueeze(dim=-3)
        else:
            n_b, n_a, _, f = v.shape
            v = v.unsqueeze(dim=-4)
            v = v.repeat(1, n_a, 1, 1, 1)
            v = v[loop_mask == 0].view(n_b, n_a, n_a - 1, 3, f)
        if s1.dim() == 4:
            v_m = s_n3.unsqueeze(dim=-2) * v3 + (
                s2.unsqueeze(dim=-2) * v + s3.unsqueeze(dim=-2) * d_vec_norm
            ).sum(dim=-3)
        else:  # higher order L = 3
            v_m = s_n3.unsqueeze(dim=-2) * v3 + (s2 * v + s3 * d_vec_norm).sum(dim=-3)
        return s_m, s_out, v_m


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
        :param s: scale info;   shape: (n_b, n_a, n_f)
        :param v: vector info;  shape: (n_b, n_a, 3, n_f)
        :return: updated s, updated v
        """
        v1, v2 = self.u(v), self.v(v)
        s0 = self.a(torch.cat([s, torch.linalg.norm(v2, 2, -2)], dim=-1))
        sg, ss = torch.split(s0, split_size_or_sections=self.n_feature, dim=-1)
        vg = v1 * ss.unsqueeze(dim=-2)
        return sg, vg


if __name__ == "__main__":
    ...
