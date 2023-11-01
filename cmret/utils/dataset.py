# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
DataSets classes.

The item returned from these classes is a python 
dictionary: {"mol": mol_dict, "label": label_dict},
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
from typing import Optional, Dict, Union
from ase.io import read
from ase.db import connect
import torch
from torch import Tensor
from torch.utils import data


class ASEData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        task: Optional[str] = None,
        use_pbc: bool = False,
    ) -> None:
        """
        Dataset stored in sql via ASE.

        :param file: dataset file name <file>
        :param limit: item limit
        :param task: task name
        :param use_pbc: whether to apply PBC in a cell
        """
        super().__init__()
        with connect(file) as db:
            data = db.select(limit=limit)
        self.data = list(data)
        self.task = task
        self.use_pbc = use_pbc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        lattice = torch.tensor(d.cell, dtype=torch.float32)
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
            energy = torch.tensor(d.data.E, dtype=torch.float32)
            label = {"scalar": energy}
            if "F" in d.data:
                label["vector"] = torch.tensor(d.data.F, dtype=torch.float32)
        return {"mol": mol, "label": label}


class XYZData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        use_pbc: bool = False,
    ) -> None:
        """
        Dataset stored in extend xyz file.

        :param file: dataset file name <file>
        :param limit: item limit
        :param use_pbc: whether to apply PBC in a cell
        """
        super().__init__()
        self.data = read(file, index=f":{limit if limit else ''}")
        self.use_pbc = use_pbc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        lattice = torch.tensor(d.cell, dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(d.pbc, dtype=torch.float32)
            if pbc.sum() > 0 and self.use_pbc:
                # mask the non-periodic direction(s)
                mol["lattice"] = lattice * pbc[:, None]
        energy = torch.tensor([d.get_total_energy()], dtype=torch.float32)
        label = {"scalar": energy}  # shape: (1,)
        try:
            forces = torch.tensor(d.get_forces(), dtype=torch.float32)
            label["vector"] = forces  # shape: (n_a, 3)
        except AttributeError:
            pass
        return {"mol": mol, "label": label}


if __name__ == "__main__":
    ...
