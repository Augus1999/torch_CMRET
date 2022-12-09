# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Molecular dynamics model using Velocity Verlet
"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from ase import Atoms, io


k_B = 1.380649e-23  #         Boltzmann contant in J/K
E_h = 4.3597447222071e-18  #  Hartree energy in J
E_eV = 1.602177e-19  #        eV energy in J
E_kcal_mol = 6.9477e-21  #    kcal/mol energy in J
N_A = 6.02214e23  #           Avogadro constant in mol^-1

unit_factor = {r"Hartree": E_h, r"eV": E_eV, r"kcal/mol": E_kcal_mol}


class Molecule:
    """
    Define a molecule that can undergo molecular dynamics process. \n
    Using Velocity Verlet method.
    """

    def __init__(self, **kargv) -> None:
        """
        :param kargv: it should be "Z": nuclear charges tensor;      shape: (1, n_a)
                                   "R": nuclear coordinates tensor;  shape: (1, n_a, 3)
                                   "Q": total charge tensor;         shape: (1, 1)
                                   "S": spin state tensor;           shape: (1, 1)
        """
        self.device = torch.device("cpu")
        self.calculator: Optional[nn.Module] = None
        self.energy, self.molecule = [], []
        self.Z, self.R, self.Q = None, None, None
        if kargv:
            self.Z: Tensor = kargv["Z"]
            self.R: Tensor = kargv["R"]
            if "Q" in kargv:
                self.Q: Tensor = kargv["Q"]
            if "S" in kargv:
                self.Q: Tensor = kargv["S"]

    @staticmethod
    def _init_velocity(M: Tensor, T: float = 298) -> Tensor:
        """
        Initialise the velocitis.

        :param M: atomic masses;        shape: (1, n_a)
        :param T: temperature, unit in K
        :return: a list of velocities;  shape: (1, n_a, 3), unit in ångström/s
        """
        E = 1.5 * T * k_B
        _, n_atom = M.shape
        v_atoms = (2 * E / M).sqrt() * 1e10  # convert to ångström/s
        rand_v = torch.rand(size=[1, n_atom, 3], dtype=torch.float32, device=M.device)
        coeff = torch.linalg.norm(rand_v, 2, -1)
        velocities = (v_atoms / coeff).unsqueeze(dim=-1) * rand_v
        return velocities

    def from_ase(self, atoms: Atoms) -> None:
        """
        Obtain molecular info from ASE Atoms object.

        :param atoms: ase.Atoms object
        :return: None
        """
        self.Z = torch.tensor(atoms.numbers, dtype=torch.long)[None, :]
        self.R = torch.tensor(atoms.positions, dtype=torch.float32)[None, :, :]

    def from_file(self, file: str, idx: int = 0) -> None:
        """
        Obtain molecular info from a file supported by ASE.

        :param file: file name <file>
        :param idx: index of the molecule in the file
        :return: None
        """
        assert isinstance(idx, int), "only single molecule input is supported."
        atoms = io.read(file, idx=idx)
        self.from_ase(atoms=atoms)

    def run(
        self,
        delta_t: float = 1e-18,
        step: int = 10000,
        temperature: float = 298,
    ) -> None:
        """
        :param delta_t: time step, unit in second
        :param steps: MD loop steps
        :param temperature: temperature, unit in K;
                            if mode is "relaxation" this argument will be ignored
        :return: None
        """
        assert self.calculator != None, "You must specify a force-field calculator."
        assert self.Z != None, "You forgot to specify nuclear charges."
        assert self.R != None, "You forgot to provide the atomic positions."
        factor = unit_factor[self.calculator.unit]
        self.energy, self.molecule = [], []  # clear list
        self.calculator = self.calculator.to(device=self.device).eval()
        mol = {"Z": self.Z.to(device=self.device), "R": self.R.to(device=self.device)}
        if self.Q:
            mol["Q"] = self.Q.to(device=self.device)
        atoms = Atoms(numbers=self.Z[0].numpy(), positions=mol["R"][0].detach().numpy())
        M = torch.tensor(atoms.get_masses() * 1e-3 / N_A, device=self.device)
        M = M[None, :]
        out = self.calculator(mol)
        energy, forces = out["energy"], out["force"]
        self.energy.append(energy.item())
        self.molecule.append(atoms)
        velocity = self._init_velocity(M=M, T=temperature)
        acc = forces / M.unsqueeze(dim=-1) * factor * 1e10
        for _ in range(step):
            velocity += 0.5 * acc * delta_t * 1e10
            mol["R"].requires_grad = False
            mol["R"] += velocity * delta_t
            out = self.calculator(mol)
            energy, forces = out["energy"], out["force"]
            acc = forces / M.unsqueeze(dim=-1) * factor * 1e10
            velocity = velocity + 0.5 * acc * delta_t * 1e10
            atoms = Atoms(
                numbers=self.Z[0].numpy(), positions=mol["R"][0].detach().numpy()
            )
            self.energy.append(energy.item())
            self.molecule.append(atoms)


if __name__ == "__main__":
    ...
