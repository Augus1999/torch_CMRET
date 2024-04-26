# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on ISO17 dataset
"""
import argparse
from pathlib import Path
from typing import Dict, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from cmret.utils import test, ASEDataBaseClass, collate
from cmret import CMRETModel, CMRET4Training


root = Path(__file__).parent
lightning_model_hparam = {
    "model_unit": "eV",
    "lr_scheduler_factor": 0.9,
    "lr_scheduler_patience": 50,
    "lr_scheduler_interval": "step",
    "lr_scheduler_frequency": 10000,
    "lr_warmup_step": 10000,
    "max_lr": 1e-3,
    "ema_alpha": 0.05,
}


class ASEData(ASEDataBaseClass):
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
    log_dir = root / "logs"

    trainset = ASEData(
        f"{args.folder}/reference.db",
        idx_file=f"{args.folder}/train_ids.txt",
        first_idx=1,
    )
    valset = ASEData(
        f"{args.folder}/reference.db",
        idx_file=f"{args.folder}/validation_ids.txt",
        first_idx=1,
    )
    traindata = DataLoader(trainset, args.batchsize, True, collate_fn=collate)
    valdata = DataLoader(valset, args.batchsize, collate_fn=collate)

    model = CMRETModel(n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh)
    lightning_model = CMRET4Training(model, lightning_model_hparam)

    ckpt_callback = ModelCheckpoint(dirpath=workdir, monitor="val_loss")
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=60)
    trainer = Trainer(
        max_epochs=args.nepoch,
        log_every_n_steps=1000,
        default_root_dir=log_dir,
        callbacks=[ckpt_callback, earlystop_callback],
        gradient_clip_val=5.0,
        gradient_clip_algorithm="value",
    )

    trainer.fit(lightning_model, traindata, valdata)
    lightning_model = CMRET4Training.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, lightning_model_hparam
    )
    lightning_model.export_model(workdir)
    model = lightning_model.cmret

    test_within = ASEData(f"{args.folder}/test_within.db")
    test_within = DataLoader(test_within, 20, collate_fn=collate)
    test_other = ASEData(f"{args.folder}/test_other.db")
    test_other = DataLoader(test_other, 20, collate_fn=collate)
    info_within = test(model=model, dataset=test_within)
    print("ISO17: within", info_within)
    info_other = test(model=model, dataset=test_other)
    print("ISO17: other", info_other)


if __name__ == "__main__":
    main()
