# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
DataSets classes
The item returned from these classes is a python 
dictionary: {"mol": mol_dict, "label": label_dict},
where mol_dict is a dictionary as {
                                    "Z": nuclear charges (Tensor), 
                                    "R": atomic positions (Tensor), 
                                    "Q": molecular net charge (Tensor) which is optional,
                                    "S": spin state (Tensor) which is optional,
                                    }
and label_dict is dictionary as {"E": energy (Tensor), "F": atomic forces (Tensor)}.
"""
import os
import glob
from pathlib import Path
from typing import Optional, Dict, Union, Generator
from ase.io import read
from ase.db import connect
import requests
from clint.textui import progress
import torch
from torch import Tensor
from torch.utils import data


valid_dataset_name = (
    "QM.CH2",
    "rMD17.benzene",
    "rMD17.ethanol",
    "rMD17.uracil",
    "rMD17.naphthalene",
    "rMD17.aspirin",
    "rMD17.salicylic",
    "rMD17.malonaldehyde",
    "rMD17.toluene",
    "rMD17.paracetamol",
    "rMD17.azobenzene",
)


class ASEData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
    ) -> None:
        """
        Dataset stored in sql via ASE.

        :param file: dataset file name <file>
        :param limit: item limit
        """
        super().__init__()
        with connect(file) as db:
            data = db.select(limit=limit)
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        forces = torch.tensor(d.data.F, dtype=torch.float32)
        energy = torch.tensor(d.data.E, dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        if "S" in d.data:
            spin = torch.tensor(d.data.S, dtype=torch.float32)
            mol["S"] = spin
        if "Q" in d.data:
            charge = torch.tensor(d.data.Q, dtype=torch.float32)
            mol["Q"] = charge
        label = {"E": energy, "F": forces}
        return {"mol": mol, "label": label}


class XYZData(data.Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
    ) -> None:
        """
        Dataset stored extend xyz file.

        :param file: dataset file name <file>
        :param limit: item limit
        """
        super().__init__()
        self.data = read(file, index=":")
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.data[idx]
        charges = torch.tensor(d.numbers, dtype=torch.long)
        positions = torch.tensor(d.positions, dtype=torch.float32)
        forces = torch.tensor(d.get_forces(), dtype=torch.float32)
        energy = torch.tensor([d.get_total_energy()], dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        label = {"E": energy, "F": forces}
        return {"mol": mol, "label": label}


class DataSet:
    """
    Import, download dataset\n
    Recently support downloading rMD17 and QM datasets.
    """

    def __init__(
        self,
        name: str,
        dir_: str,
        mode: str = "train",
        limit: Optional[int] = None,
    ) -> None:
        """
        Check the state of dataset, download if not exist.

        :param name: dataset name, choose from "rMD17.xxx", "QM.CH2"
        :param dir: where the dataset locates, or where to store it after downloading <path>
        :param mode: mode; "train" or "test"
        :param limit: item limit
        """
        assert (
            name in valid_dataset_name
        ), f"keyword 'name' should be one of {valid_dataset_name}"
        self.name = name.split(".")[0]
        assert mode in ("train", "test"), "invalid mode..."
        self.dir = Path(dir_)
        self.base_name = name.split(".")[-1]
        self.mode = mode
        self.limit = limit
        if not self._check:
            self._download()

    @property
    def _dataset(self) -> data.Dataset:
        """
        Return the required dataset class.
        """
        dataset = {"QM": ASEData, "rMD17": ASEData}
        return dataset[self.name]

    @property
    def unit(self) -> str:
        """
        Return the dataset energy unit.
        """
        units = {"QM": "eV", "rMD17": "eV"}
        return units[self.name]

    @property
    def _check(self) -> bool:
        """
        Check whether all required files exist.
        """
        if not os.path.exists(self.dir):
            return False
        if self.name == "rMD17":
            d_files = list(glob.glob(str(self.dir / r"*.db")))
            d_files = set([os.path.basename(i) for i in d_files])
            valid_files = {
                self.base_name + "_train.db",
                self.base_name + "_test.db",
            }
            if valid_files & d_files == valid_files:
                return True
        if self.name == "QM":
            d_files = list(glob.glob(str(self.dir / r"*.db")))
            d_files = set([os.path.basename(i) for i in d_files])
            valid_files = {
                self.base_name.lower() + "_2000_train.db",
                self.base_name.lower() + "_2000_test.db",
            }
            if valid_files & d_files == valid_files:
                return True
        return False

    def _download(self) -> None:
        """
        Download and prepare the dataset.
        """
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.name == "rMD17":
            urls = {
                f"{self.base_name}_train.db": f"https://github.com/Augus1999/torch_CMRET/blob/main/dataset/rmd17/{self.base_name}_train.db?raw=true",
                f"{self.base_name}_test.db": f"https://github.com/Augus1999/torch_CMRET/blob/main/dataset/rmd17/{self.base_name}_test.db?raw=true",
            }
            for name in list(urls.keys()):
                print(f"downloading {name}...")
                temp = requests.get(urls[name], stream=True)
                total_length = int(temp.headers.get("content-length"))
                file_name = self.dir / name
                with open(file_name, mode="wb") as f:
                    for chunk in progress.bar(
                        temp.iter_content(chunk_size=1024),
                        expected_size=(total_length / 1024) + 1,
                        width=50,
                    ):
                        if chunk:
                            f.write(chunk)
                print("finished downloading")
        if self.name == "QM":
            urls = {
                f"{self.base_name.lower()}_2000_train.db": "https://github.com/Augus1999/torch_CMRET/blob/main/dataset/ch2_2000_train.db?raw=true",
                f"{self.base_name.lower()}_2000_test.db": "https://github.com/Augus1999/torch_CMRET/blob/main/dataset/ch2_2000_test.db?raw=true",
            }
            for name in list(urls.keys()):
                print(f"downloading {name}...")
                temp = requests.get(urls[name], stream=True)
                total_length = int(temp.headers.get("content-length"))
                file_name = self.dir / name
                with open(file_name, mode="wb") as f:
                    for chunk in progress.bar(
                        temp.iter_content(chunk_size=1024),
                        expected_size=(total_length / 1024) + 1,
                        width=50,
                    ):
                        if chunk:
                            f.write(chunk)
                print("finished downloading")

    @property
    def data(self) -> Generator:
        """
        Return data from dataset.
        """
        if self.name == "rMD17":
            file_names = {
                "train": self.base_name + "_train.db",
                "test": self.base_name + "_test.db",
            }
            file_name = file_names[self.mode]
        if self.name == "QM":
            file_names = {
                "train": self.base_name.lower() + "_2000_train.db",
                "test": self.base_name.lower() + "_2000_test.db",
            }
            file_name = file_names[self.mode]
        dataset = self._dataset(
            file=self.dir / file_name,
            limit=self.limit,
        )
        return dataset


if __name__ == "__main__":
    ...
