# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on ISO17 dataset
"""
import argparse
from pathlib import Path
from typing import Optional, Dict, Union
from ase.db import connect
import torch
from torch import Tensor
from torch.utils import data
from cmret.utils import train, test, find_recent_checkpoint
from cmret.representation import CMRETModel


root = Path(__file__).parent

# you can set 'num_workers' varibale in torch.utils.data.DataLoader
# by un-commentting the following:
# import os
# os.environ["NUM_WORKER"] = "x"  # here 'x' is the number of workers you want 


class ASEData(data.Dataset):
    def __init__(
        self,
        file: str,
        idx_file: Optional[str] = None,
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
        self.idx = []
        if idx_file:
            with open(idx_file, "r") as f:
                idx = f.readlines()
            self.idx = [int(i) - 1 for i in idx]

    def __len__(self):
        length = len(self.data)
        if self.idx:
            length_ = len(self.idx)
            length = min((length, length_))
        return length

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.idx:
            idx = self.idx[idx]
        d = self.data[idx]
        charges = torch.tensor(d.toatoms().numbers, dtype=torch.long)
        positions = torch.tensor(d.toatoms().positions, dtype=torch.float32)
        forces = torch.tensor(d.data["atomic_forces"], dtype=torch.float32)
        energy = torch.tensor([d["total_energy"]], dtype=torch.float32)
        mol = {"Z": charges, "R": positions}
        label = {"scalar": energy, "vector": forces}
        return {"mol": mol, "label": label}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayer", default=6, type=int, help="number of layers")
    parser.add_argument("--nh", default=4, type=int, help="number of attention head")
    parser.add_argument("--batchsize", default=15, type=int, help="batchsize")
    parser.add_argument("--nepoch", default=400, type=int, help="number epochs")
    parser.add_argument("--rbf", default="gaussian", type=str, help="RBF type")
    parser.add_argument("--folder", default=".", type=str, help="the dataset folder")
    args = parser.parse_args()

    workdir = root / "chkpts/iso17"
    log_dir = root / "logs/iso17.log"
    load = find_recent_checkpoint(workdir=workdir)

    dataset = ASEData(f"{args.folder}/reference.db", f"{args.folder}/train_ids.txt")
    model = CMRETModel(n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh)

    train(
        model=model,
        dataset=dataset,
        batch_size=args.batchsize,
        max_n_epochs=args.nepoch,
        unit="eV",
        load=load,
        save_every=10,
        work_dir=workdir,
        log_dir=log_dir,
    )

    test_within = ASEData(f"{args.folder}/test_within.db")
    test_other = ASEData(f"{args.folder}/test_other.db")
    info_within = test(model=model, dataset=test_within, load=workdir / "trained.pt")
    print("ISO17: within", info_within)
    info_other = test(model=model, dataset=test_other, load=workdir / "trained.pt")
    print("ISO17: other", info_other)


if __name__ == "__main__":
    main()
