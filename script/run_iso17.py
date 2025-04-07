# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on ISO17 dataset
"""
import datetime
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from cmret.utils import test, ASEData, collate
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


def encode(atoms):
    energy = torch.tensor([atoms["total_energy"]], dtype=torch.float32)
    forces = torch.tensor(atoms.data["atomic_forces"], dtype=torch.float32)
    return {"scalar": energy, "vector": forces}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayer", default=6, type=int, help="number of layers")
    parser.add_argument("--nh", default=4, type=int, help="number of attention head")
    parser.add_argument("--batchsize", default=15, type=int, help="batchsize")
    parser.add_argument("--nepoch", default=400, type=int, help="number epochs")
    parser.add_argument("--rbf", default="gaussian", type=str, help="RBF type")
    parser.add_argument("--folder", default=".", type=str, help="the dataset folder")
    parser.add_argument("--wandb", default=False, type=bool, help="enable W&B or not")
    args = parser.parse_args()

    workdir = root / "chkpts/iso17"
    log_dir = root / "logs"

    trainset = ASEData(
        f"{args.folder}/reference.db",
        idx_file=f"{args.folder}/train_ids.txt",
        first_idx=1,
    )
    trainset.map(encode)
    valset = ASEData(
        f"{args.folder}/reference.db",
        idx_file=f"{args.folder}/validation_ids.txt",
        first_idx=1,
    )
    valset.map(encode)
    traindata = DataLoader(trainset, args.batchsize, True, collate_fn=collate)
    valdata = DataLoader(valset, args.batchsize, collate_fn=collate)

    model = CMRETModel(n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh)
    lightning_model = CMRET4Training(model, lightning_model_hparam)

    if args.wandb:
        logger = loggers.WandbLogger(
            f"run_iso17_{args.nlayer}_{args.nh}_{args.rbf}",
            log_dir,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            project="CMRET",
            job_type="train",
        )
    else:
        logger = loggers.TensorBoardLogger(
            log_dir,
            f"run_iso17_{args.nlayer}_{args.nh}_{args.rbf}",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        )
    ckpt_callback = ModelCheckpoint(dirpath=workdir, monitor="val_loss")
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=60)
    trainer = Trainer(
        max_epochs=args.nepoch,
        logger=logger,
        log_every_n_steps=1000,
        callbacks=[ckpt_callback, earlystop_callback],
        gradient_clip_val=5.0,
        gradient_clip_algorithm="value",
    )

    trainer.fit(lightning_model, traindata, valdata)
    lightning_model = CMRET4Training.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, cmret=model
    )
    lightning_model.export_model(workdir)
    model = lightning_model.cmret

    test_within = ASEData(f"{args.folder}/test_within.db")
    test_within.map(encode)
    test_within = DataLoader(test_within, 20, collate_fn=collate)
    test_other = ASEData(f"{args.folder}/test_other.db")
    test_other.map(encode)
    test_other = DataLoader(test_other, 20, collate_fn=collate)
    info_within = test(model=model, testdata=test_within)
    print("ISO17: within", info_within)
    info_other = test(model=model, testdata=test_other)
    print("ISO17: other", info_other)


if __name__ == "__main__":
    main()
