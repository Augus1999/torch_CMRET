# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tools to test the model, and split dataset stored in xyz files.
"""
from pathlib import Path
from typing import List, Dict, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


def loss_calc(
    out: Dict[str, Tensor],
    label: Dict[str, Tensor],
    loss_f: Callable[[Tensor, Tensor], Tensor],
) -> Dict[str, Tensor]:
    """
    Calculate the scalar/vector/pretrain loss based on key(s) in label.

    :param out: output of model
    :param label: training set label
    :param loss_f: loss function
    :return: calculated loss
    """
    loss: Dict[str, Tensor] = {}
    if "scalar" in label:
        loss["scalar"] = loss_f(out["scalar"], label["scalar"])
    if "vector" in label:
        loss["vector"] = loss_f(out["vector"], label["vector"])
    if "R" in label:
        loss["vector"] = loss_f(out["R"], label["R"])
    return loss


def collate(batch: List) -> Dict[str, Tensor]:
    """
    Collate different molecules into one batch.

    :param batch: a list of data (one batch)
    :return: batched {"mol": mol, "label": label}
                      mol == {
                                "Z": atomic numbers
                                "R": coordinates
                                "Q": molecular net charges (optional)
                                "S": net spin state (optional)
                                "lattice": lattice vectors (optional)
                                "batch": batch mask (for instance molecule A has 4 atoms
                                                     and molecule B has 3 atoms then the
                                                     batched indices is [[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]])
                      }
    """
    mol = [i["mol"] for i in batch]
    label = [i["label"] for i in batch]
    charges, positions, mask = [], [], []
    scalars, vectors, coords = [], [], []
    charge, spin, lattice = [], [], []
    for key, item in enumerate(mol):
        charges.append(item["Z"])
        positions.append(item["R"])
        if "scalar" in label[key]:
            scalars.append(label[key]["scalar"].unsqueeze(dim=0))
        if "vector" in label[key]:
            vectors.append(label[key]["vector"])
        if "R" in label[key]:
            coords.append(label[key]["R"])
        if "Q" in item:
            charge.append(item["Q"].unsqueeze(dim=0))
        if "S" in item:
            spin.append(item["S"].unsqueeze(dim=0))
        if "lattice" in item:
            lattice.append(item["lattice"].unsqueeze(dim=0))
    charges = torch.cat(charges, dim=0).unsqueeze(dim=0)
    positions = torch.cat(positions, dim=0).unsqueeze(dim=0)
    n_total = charges.shape[1]
    i = 0
    for item in mol:
        n = item["Z"].shape[0]
        batch = torch.ones(1, n)
        p1, p2 = torch.zeros(1, i), torch.zeros(1, n_total - n - i)
        mask.append(torch.cat([p1, batch, p2], dim=-1))
        i += n
    mask = torch.cat(mask, dim=0).unsqueeze(dim=-1)
    mol = {"Z": charges, "R": positions, "batch": mask}
    if charge:
        mol["Q"] = torch.cat(charge, dim=0)
    if spin:
        mol["S"] = torch.cat(spin, dim=0)
    if lattice:
        mol["lattice"] = torch.cat(lattice, dim=0)
    label = {}
    if scalars:
        label["scalar"] = torch.cat(scalars, dim=0)
    if vectors:
        label["vector"] = torch.cat(vectors, dim=0).unsqueeze(dim=0)
    if coords:
        label["R"] = torch.cat(coords, dim=0).unsqueeze(dim=0)
    return {"mol": mol, "label": label}


def test(model: nn.Module, testdata: DataLoader) -> Dict[str, float]:
    """
    Test the trained network.

    :param model: a `~cmret.model.representation.CMRETModel` to be tested
    :param testdata: a `~torch.utils.data.DataLoader` instance containing test data
    :return: test metrics
    """
    scalar_count, vector_count = 0, 0
    scalar_mae, scalar_rmse = 0, 0
    vector_mae, vector_rmse = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device).eval()
    results = {}
    for d in testdata:
        mol, label = d["mol"], d["label"]
        for i in mol:
            mol[i] = mol[i].to(device)
        for j in label:
            label[j] = label[j].to(device)
        out = model(mol)
        if "scalar" in label:
            ds = (out["scalar"] - label["scalar"]).flatten()
            scalar_count += len(ds)
            scalar_mae += ds.abs().sum().item()
            scalar_rmse += ds.pow(2).sum().item()
        if "vector" in label:
            dv = (out["vector"] - label["vector"]).flatten()
            vector_count += len(dv)
            vector_mae += dv.abs().sum().item()
            vector_rmse += dv.pow(2).sum().item()
    if scalar_count != 0:
        results["scalar_MAE"] = scalar_mae / scalar_count
        results["scalar_RMSE"] = (scalar_rmse / scalar_count) ** 0.5
    if vector_count != 0:
        results["vector_MAE"] = vector_mae / vector_count
        results["vector_RMSE"] = (vector_rmse / vector_count) ** 0.5
    return results


def split_data(
    file_name: Union[str, Path], train_split_idx: List, test_split_idx: List
) -> None:
    """
    Split a dataset to file_name-train.xyz and file_name-test.xyz files.

    :param file_name: file to be splited <file>
    :param train_split_idx: indeces of training data
    :param test_split_idx: indeces of testing data
    :return: None
    """
    assert str(file_name).endswith(".xyz"), "Unsupported format..."
    with open(file_name, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    line_indexes = []
    lines_group, lines_train, lines_test = [], [], []
    for idx, line in enumerate(lines):
        try:
            int(line)
            line_indexes.append(idx)
        except ValueError:
            pass
    l_i_len = len(line_indexes)
    for key, idx in enumerate(line_indexes):
        mol_info = lines[idx : line_indexes[key + 1] if (key + 1) < l_i_len else None]
        try:
            # convert to the format supported by ASE
            energy = float(mol_info[1])
            mol_info[1] = f"Properties=species:S:1:pos:R:3:forces:R:3 energy={energy}\n"
        except ValueError:
            pass
        lines_group.append(mol_info)
    for key, i in enumerate(lines_group):
        if key in train_split_idx:
            for j in i:
                lines_train.append(j)
        if key in test_split_idx:
            for j in i:
                lines_test.append(j)
    with open(
        str(file_name).replace(".xyz", "-train.xyz"), mode="w", encoding="utf-8"
    ) as sf:
        sf.writelines(lines_train)
    with open(
        str(file_name).replace(".xyz", "-test.xyz"), mode="w", encoding="utf-8"
    ) as sf:
        sf.writelines(lines_test)


if __name__ == "__main__":
    ...
