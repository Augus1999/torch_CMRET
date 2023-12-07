# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Define ASE calculator.
"""
from typing import List, Union
import torch
import torch.nn as nn
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

k_B = 1.380649e-23  #         Boltzmann contant in J/K
E_h = 4.3597447222071e-18  #  Hartree energy in J
E_eV = 1.602177e-19  #        eV energy in J
E_kcal_mol = 6.9477e-21  #    kcal/mol energy in J

unit_factor = {r"Hartree": E_h, r"eV": E_eV, r"kcal/mol": E_kcal_mol}


class CMRETCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model: nn.Module, device: Union[str, torch.device] = "cpu"
    ) -> None:
        """
        ASE calculator class wrapping CMRET model.

        :param model: a trained CMRET model
        :param device: target device
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        if model.unit in unit_factor:
            self.scale = unit_factor[model.unit] / E_eV
        else:
            self.scale = 1.0

    def calculate(
        self,
        atoms: Atoms,
        properties: List[str] = ["energy", "forces"],
        system_changes: List[str] = all_changes,
    ) -> None:
        """
        Calculate the properties.

        :param atoms: ase.Atoms object
        :param properties: implemented properties (not in use)
        :param system_changes: list of changes for ASE (not in use)
        :return: properties
        """
        atoms_ = atoms.copy()
        Z = torch.tensor(atoms_.numbers, dtype=torch.long)[None, :]
        R = torch.tensor(atoms_.positions, dtype=torch.float32)[None, :, :]
        mol = {"Z": Z.to(device=self.device), "R": R.to(device=self.device)}
        mol["batch"] = torch.ones_like(mol["Z"], dtype=torch.float32).unsqueeze(dim=-1)
        lattice = torch.tensor(atoms_.cell.array, dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(atoms_.pbc, dtype=torch.float32)
            lattice *= pbc[:, None]  # mask the non-periodic direction(s)
            mol["lattice"] = lattice[None, :, :].to(self.device)
        mol_info = atoms_.info
        if "S" in mol_info:
            spin = mol_info["S"]
            mol["S"] = torch.tensor([[spin]], dtype=torch.float32, device=self.device)
        if "Q" in mol_info:
            charge = mol_info["Q"]
            mol["Q"] = torch.tensor([[charge]], dtype=torch.float32, device=self.device)
        res = self.model(mol)
        results = {}
        results["energy"] = res["scalar"].detach().cpu().item() * self.scale
        if "vector" in res:
            results["forces"] = res["vector"][0].detach().cpu().numpy() * self.scale
        self.results = results


if __name__ == "__main__":
    ...
