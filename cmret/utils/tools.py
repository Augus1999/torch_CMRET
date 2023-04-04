# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Useful tools (hopefully :)
"""
import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Generator, Callable
import torch
import torch.nn as nn
import torch.optim as op
from torch import Tensor
from torch.utils.data import DataLoader


class _TwoCycleLR:
    def __init__(self, optimizer, total_steps: int, step_size_up: int = 100000) -> None:
        self.total_steps = total_steps
        self.step_size_up = step_size_up
        self.scheduler = op.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=1e-6,
            max_lr=1e-4,
            step_size_up=step_size_up,
            step_size_down=total_steps // 2 - step_size_up,
            cycle_momentum=False,
        )

    def step(self, step_idx: int) -> None:
        self.scheduler.step()
        if step_idx == self.total_steps // 2 - 1:
            self.scheduler = op.lr_scheduler.CyclicLR(
                optimizer=self.scheduler.optimizer,
                base_lr=1e-8,
                max_lr=1e-6,
                step_size_up=self.step_size_up,
                step_size_down=self.total_steps // 2 - self.step_size_up - 2,
                cycle_momentum=False,
            )
            self.scheduler.last_epoch = -1


def energy_force_loss(
    out: Dict[str, Tensor],
    label: Dict[str, Tensor],
    loss_f: Callable[[Tensor, Tensor], Tensor],
    raw: bool = False,
) -> Union[Tensor, Dict]:
    """
    Claculate the energy-force loss

    :param out: output of model
    :param label: training set label
    :param loss_f: loss function
    :param raw: whether output raw losses
    :return: energy-force loss
    """
    rho = 0.2
    energy, forces = label["E"], label["F"]
    out_en, out_forces = out["energy"], out["force"]
    loss1 = loss_f(out_en, energy)
    loss2 = loss_f(out_forces, forces)
    if raw:
        return {"energy": loss1, "force": loss2}
    loss = loss1 * rho + loss2 * (1 - rho)
    return loss


def energy_loss(
    out: Dict[str, Tensor],
    label: Dict[str, Tensor],
    loss_f: Callable[[Tensor, Tensor], Tensor],
    raw: bool = False,
) -> Union[Tensor, Dict]:
    """
    Claculate the energy loss

    :param out: output of model
    :param label: training set label
    :param loss_f: loss function
    :param raw: whether output raw loss
    :return: energy loss
    """
    energy = label["E"]
    out_en = out["energy"]
    loss = loss_f(out_en, energy)
    if raw:
        return {"energy": loss}
    return loss


def pretrain_loss(
    out: Dict[str, Tensor],
    label: Dict[str, Tensor],
    loss_f: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """
    Claculate the pretrain loss

    :param out: output of model
    :param label: training set label
    :param loss_f: loss function
    :return: pre-train loss
    """
    geometry = label["R"].flatten()
    out_geometry = out["v"].sum(dim=-1).flatten()
    return loss_f(out_geometry, geometry)


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
                                "batch": batch mask (for instance molecule A has 4 atoms
                                                     and molecule B has 3 atoms then the
                                                     batched indices is [[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]])
                      }
    """
    mol = [i["mol"] for i in batch]
    label = [i["label"] for i in batch]
    charges, positions, mask = [], [], []
    energies, forces, coords = [], [], []
    charge, spin = [], []
    for key, item in enumerate(mol):
        charges.append(item["Z"])
        positions.append(item["R"])
        if "E" in label[key]:
            energies.append(label[key]["E"].unsqueeze(dim=0))
        if "F" in label[key]:
            forces.append(label[key]["F"])
        if "R" in label[key]:
            coords.append(label[key]["R"])
        if "Q" in item:
            charge.append(item["Q"].unsqueeze(dim=0))
        if "S" in item:
            spin.append(item["S"].unsqueeze(dim=0))
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
    label = {}
    if energies:
        label["E"] = torch.cat(energies, dim=0)
    if forces:
        label["F"] = torch.cat(forces, dim=0).unsqueeze(dim=0)
    if coords:
        label["R"] = torch.cat(coords, dim=0).unsqueeze(dim=0)
    return {"mol": mol, "label": label}


def train(
    model: nn.Module,
    dataset: Generator,
    batch_size: int = 5,
    max_n_epochs: int = 20000,
    loss_calculator=energy_force_loss,
    unit: str = "eV",
    load: Optional[str] = None,
    log_dir: Optional[str] = None,
    work_dir: str = ".",
    save_every: Optional[int] = None,
) -> None:
    """
    Trian the network.

    :param model: model for training
    :param dataset: training set
    :param batch_size: mini-batch size
    :param max_n_epochs: max training epochs size
    :param loss_calculator: loss calculator
    :param unit: dataset energy unit
    :param load: load from an existence state file <file>
    :param log_dir: where to store log file <file>
    :param work_dir: where to store model state_dict <path>
    :param save_every: store checkpoint every 'save_every' epochs
    :return: None
    """
    if os.path.exists(str(log_dir)):
        with open(log_dir, mode="r+", encoding="utf-8") as f:
            f.truncate()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logging.basicConfig(
        filename=log_dir,
        format=" %(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    start_epoch: int = 0
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate, shuffle=True
    )
    train_size = len(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device).train()
    optimizer = op.Adam(model.parameters(), lr=1e-8)
    scheduler = _TwoCycleLR(optimizer=optimizer, total_steps=max_n_epochs * train_size)
    logging.info(f"using hardware {str(device).upper()}")
    logging.debug(f"{model.check_parameter_number} trainable parameters")
    if load:
        with open(load, mode="rb") as sf:
            state_dic = torch.load(sf, map_location=device)
        keys = {"nn", "opt", "epoch", "scheduler"}
        if keys & set(state_dic.keys()) == keys:
            model.load_state_dict(state_dict=state_dic["nn"])
            scheduler.scheduler.load_state_dict(state_dict=state_dic["scheduler"])
            start_epoch: int = state_dic["epoch"]
            optimizer.load_state_dict(state_dict=state_dic["opt"])
        else:
            model.load_state_dict(state_dict=state_dic["nn"])
        logging.info(f'loaded state from "{load}"')
    train_loss = nn.MSELoss()
    logging.info("start training")
    for epoch in range(start_epoch, max_n_epochs):
        running_loss = 0.0
        for key, d in enumerate(loader):
            mol, label = d["mol"], d["label"]
            for i in mol:
                mol[i] = mol[i].to(device)
            for j in label:
                label[j] = label[j].to(device)
            out = model(mol)
            loss = loss_calculator(out, label, train_loss)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            # update scheduler after optimised
            scheduler.step(step_idx=key + epoch * train_size)
        logging.info(f"epoch: {epoch + 1} loss: {running_loss / train_size}")
        if save_every:
            if (epoch + 1) % save_every == 0:
                state = {
                    "nn": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scheduler": scheduler.scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "unit": unit,
                }
                chkpt_idx = str(epoch + 1).zfill(len(str(max_n_epochs)))
                torch.save(state, Path(work_dir) / f"state-{chkpt_idx}.pkl")
                logging.info(f"saved checkpoint state-{chkpt_idx}.pkl")
    torch.save(
        {"nn": model.state_dict(), "unit": unit},
        Path(work_dir) / r"trained.pt",
    )
    logging.info("saved state!")
    logging.info("finished")


def test(
    model: nn.Module,
    dataset: Generator,
    load: Optional[str] = None,
    loss_calculator=energy_force_loss,
    metric_type: str = "MAE",
) -> Dict[str, float]:
    """
    Test the trained network.

    :param model: model for testing
    :param dataset: dataset class
    :param load: load from an existence state file <file>
    :param loss_calculator: loss calculator
    :param metric_type: chosen from 'MAE' and 'RMSE'
    :return: test metrics
    """
    loader = DataLoader(dataset=dataset, batch_size=10, collate_fn=collate)
    size = len(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.pretrained(file=load).to(device=device).eval()
    if metric_type == "MAE":
        Loss = nn.L1Loss()
    elif metric_type == "RMSE":
        Loss = nn.MSELoss()
    else:
        raise NotImplementedError
    results = {}
    for d in loader:
        mol, label = d["mol"], d["label"]
        for i in mol:
            mol[i] = mol[i].to(device)
        for j in label:
            label[j] = label[j].to(device)
        out = model(mol)
        result = loss_calculator(out, label, Loss, True)
        for key in result:
            if key not in results:
                results[key] = result[key].item()
            else:
                results[key] += result[key].item()
    for key in results:
        results[key] = results[key] / size
        if metric_type == "RMSE":
            results[key] = results[key] ** 0.5
    return results


def find_recent_checkpoint(workdir: str) -> Optional[str]:
    """
    Find the most recent checkpoint file in the work dir. \n
    The name of checkpoint file is like state-abcdef.pkl

    :param workdir: the directory where the checkpoint files stored <path>
    :return: the file name
    """
    load: Optional[str] = None
    if os.path.exists(workdir):
        cps = list(glob.glob(str(Path(workdir) / r"*.pkl")))
        if cps:
            cps.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("-")[1]))
            load = cps[-1]
    return load


def extract_log_info(log_name: str = "training.log") -> Dict[str, List]:
    """
    Extract training loss from training log file.

    :param log_name: log file name <file>
    :return: dict["epoch": epochs, "loss": loss]
    """
    info = {"epoch": [], "loss": []}
    with open(log_name, mode="r", encoding="utf-8") as f:
        lines = f.read()
    loss_info = re.findall(r"epoch: \d+ loss: \d+(.\d+(e|E)?(-|\+)?\d+)?", lines)
    if loss_info:
        for i in loss_info:
            epoch = int(i.split(" ")[1])
            loss = float(i.split(" ")[-1])
            info["epoch"].append(epoch)
            info["loss"].append(loss)
    return info


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
