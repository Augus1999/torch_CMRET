# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
DataSets classes.

The item returned from these classes is dict instance: {"mol": mol_dict, "label": label_dict},
where mol_dict is a dictionary as {
                                    "Z": nuclear charges (Tensor),
                                    "R": atomic positions (Tensor),
                                    "Q": molecular net charge (Tensor) which is optional,
                                    "S": spin state (Tensor) which is optional,
                                    "lattice": non-unit lattice vectors (Tensor) which is optional,
                                    }
and label_dict is dictionary as {
                                  "scalar": scalar property (Tensor),
                                  "vector": atomic forces (Tensor) which is optional,
                                }.
"""
from typing import Optional, Dict, Union, Callable
from ase import Atoms
from ase.io import read
from ase.db import connect, row
import torch
from torch import Tensor
from torch.utils import data


class ASEData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        idx_file: Optional[str] = None,
        first_idx: int = 0,
        task: Optional[str] = None,
        use_pbc: bool = False,
    ) -> None:
        """
        Dataset stored in sql via ASE.

        :param file: dataset file name <file>
        :param limit: item limit
        :param idx_file: index file name <file>
        :param first_idx: the integer (0 or 1) represents the first index
        :param task: task name
        :param use_pbc: whether to apply PBC in a cell
        """
        super().__init__()
        with connect(file) as db:
            data = db.select(limit=limit)
        self.data = list(data)
        self.task = task
        self.use_pbc = use_pbc
        self.idx = []
        if idx_file:
            with open(idx_file, "r") as f:
                idx = f.readlines()
            self.idx = [int(i) - first_idx for i in idx]
        self.mapping = self._mapping

    def __len__(self):
        length = len(self.data)
        if self.idx:
            idx_length = len(self.idx)
            length = min((length, idx_length))
        return length

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.idx:
            idx = self.idx[idx]
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        lattice = torch.tensor(d.cell.tolist(), dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(d.pbc, dtype=torch.float32)
            if pbc.sum() > 0 and self.use_pbc:
                # mask the non-periodic direction(s)
                mol["lattice"] = lattice * pbc[:, None]  # shape: (3, 3)
        if "S" in d.data:
            mol["S"] = torch.tensor(d.data.S, dtype=torch.float32)
        if "Q" in d.data:
            mol["Q"] = torch.tensor(d.data.Q, dtype=torch.float32)
        if self.task:
            label = {"scalar": torch.tensor(d.data[self.task], dtype=torch.float32)}
        else:
            label = self.mapping(d)
        return {"mol": mol, "label": label}

    def map(self, mapping: Callable[[row.AtomsRow], Dict[str, Tensor]]) -> None:
        """
        Pass a customised mapping function to transform the data entities to tensors of label.

        e.g.
        ```python
        import torch


        def encode(atoms):
            energy = torch.tensor([atoms["total_energy"]], dtype=torch.float32)
            # or
            # energy = torch.tensor(atoms.data.E, dtype=torch.float32)
            # etc.
            forces = torch.tensor(atoms.data["atomic_forces"], dtype=torch.float32)
            return {"scalar": energy, "vector": forces}

        dataset = ASEData(...)
        dataset.map(encode)
        ```

        :param mapping: customised mapping function
        """
        self.mapping = mapping

    @staticmethod
    def _mapping(atoms: row.AtomsRow) -> Dict[str, Tensor]:
        energy = torch.tensor(atoms.data.E, dtype=torch.float32)
        label = {"scalar": energy}
        if "F" in atoms.data:
            label["vector"] = torch.tensor(atoms.data.F, dtype=torch.float32)
        return label


class XYZData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        idx_file: Optional[str] = None,
        first_idx: int = 0,
        use_pbc: bool = False,
    ) -> None:
        """
        Dataset stored in extend xyz file.

        :param file: dataset file name <file>
        :param limit: item limit
        :param idx_file: index file name <file>
        :param first_idx: the integer (0 or 1) represents the first index
        :param use_pbc: whether to apply PBC in a cell
        """
        super().__init__()
        self.data = read(file, index=f":{limit if limit else ''}")
        self.use_pbc = use_pbc
        self.idx = []
        if idx_file:
            with open(idx_file, "r") as f:
                idx = f.readlines()
            self.idx = [int(i) - first_idx for i in idx]
        self.mapping = self._mapping

    def __len__(self):
        length = len(self.data)
        if self.idx:
            idx_length = len(self.idx)
            length = min((length, idx_length))
        return length

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.idx:
            idx = self.idx[idx]
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        lattice = torch.tensor(d.cell.tolist(), dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(d.pbc, dtype=torch.float32)
            if pbc.sum() > 0 and self.use_pbc:
                # mask the non-periodic direction(s)
                mol["lattice"] = lattice * pbc[:, None]
        if "S" in d.info:
            mol["S"] = torch.tensor([d.info["S"]], dtype=torch.float32)
        if "Q" in d.info:
            mol["Q"] = torch.tensor([d.info["Q"]], dtype=torch.float32)
        label = self.mapping(d)
        assert "scalar" in label or "vector" in label
        return {"mol": mol, "label": label}

    def map(self, mapping: Callable[[Atoms], Dict[str, Tensor]]) -> None:
        """
        Pass a customised mapping function to transform the data entities to tensors of label.

        e.g.
        ```python
        import torch


        def encode(atoms):
            energy = torch.tensor([atoms.get_total_energy()], dtype=torch.float32)
            # or
            # energy = torch.tensor([atoms.info["internal_energy"]], dtype=torch.float32)
            # etc.
            label = {"scalar": energy}  # shape: (1,)
            if "forces" in atoms._calc.results:
                forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
                label["vector"] = forces  # shape: (n_a, 3)
            return label

        dataset = XYZData(...)
        dataset.map(encode)
        ```

        :param mapping: customised mapping function
        """
        self.mapping = mapping

    @staticmethod
    def _mapping(atoms: Atoms) -> Dict[str, Tensor]:
        energy = torch.tensor([atoms.get_total_energy()], dtype=torch.float32)
        label = {"scalar": energy}  # shape: (1,)
        if "forces" in atoms._calc.results:
            forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
            label["vector"] = forces  # shape: (n_a, 3)
        return label


if __name__ == "__main__":
    ...
