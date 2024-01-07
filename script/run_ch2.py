# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on singlet/triplet CH2 dataset
"""
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from cmret.utils import test, ASEData, collate
from cmret import CMRETModel, CMRET4Training


root = Path(__file__).parent
lightning_model_hparam = {
    "model_unit": "eV",
    "lr_scheduler_factor": 0.9,
    "lr_scheduler_patience": 50,
    "lr_scheduler_interval": "epoch",
    "lr_scheduler_frequency": 1,
    "lr_warmup_step": 1000,
    "max_lr": 1e-3,
    "ema_alpha": 0.05,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayer", default=5, type=int, help="number of layers")
    parser.add_argument("--nh", default=1, type=int, help="number of attention head")
    parser.add_argument("--batchsize", default=10, type=int, help="batchsize")
    parser.add_argument("--nepoch", default=10000, type=int, help="number epochs")
    parser.add_argument("--rbf", default="gaussian", type=str, help="RBF type")
    parser.add_argument("--folder", default=".", type=str, help="the dataset folder")
    args = parser.parse_args()

    workdir = root / "chkpts/ch2"
    log_dir = root / "logs"

    train_set = ASEData(f"{args.folder}/ch2_2000_train.db")
    traindata = DataLoader(
        train_set,
        args.batchsize,
        True,
        num_workers=4,
        collate_fn=collate,
        persistent_workers=True,
    )
    test_set = ASEData(f"{args.folder}/ch2_2000_test.db")
    testdata = DataLoader(test_set, 20, collate_fn=collate)

    model = CMRETModel(
        n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh, n_kernel=20
    )
    lightning_model = CMRET4Training(model, lightning_model_hparam)

    ckpt_callback = ModelCheckpoint(dirpath=workdir, monitor="val_loss")
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=300)
    trainer = Trainer(
        max_epochs=args.nepoch,
        log_every_n_steps=20,
        default_root_dir=log_dir,
        callbacks=[ckpt_callback, earlystop_callback],
        gradient_clip_val=5.0,
        gradient_clip_algorithm="value",
    )

    trainer.fit(lightning_model, traindata, testdata)
    lightning_model.export_model(workdir)

    test_info = test(model=model, testdata=testdata)
    print("CH2:", test_info)


if __name__ == "__main__":
    main()
