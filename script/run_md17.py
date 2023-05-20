# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on MD17 dataset
"""
import argparse
from pathlib import Path
from cmret.utils import train, test, DataSet, find_recent_checkpoint
from cmret.representation import CMRETModel

root = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="ethanol",
        type=str,
        help="name of the MD17 dataset (ethanol, benzene, ...)",
    )
    parser.add_argument("--nlayer", default=6, type=int, help="number of layers")
    parser.add_argument("--nh", default=4, type=int, help="number of attention heads")
    parser.add_argument("--batchsize", default=5, type=int, help="batchsize")
    parser.add_argument("--nepoch", default=20000, type=int, help="number epochs")
    parser.add_argument("--rbf", default="gaussian", type=str, help="RBF type")
    parser.add_argument("--folder", default=".", type=str, help="the dataset folder")
    args = parser.parse_args()

    workdir = root / f"chkpts/{args.name}"
    log_dir = root / f"logs/{args.name}.log"
    load = find_recent_checkpoint(workdir=workdir)

    train_set = DataSet(f"MD17.{args.name}", args.folder, mode="train")
    model = CMRETModel(n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh)

    train(
        model=model,
        dataset=train_set.data,
        batch_size=args.batchsize,
        max_n_epochs=args.nepoch,
        unit=train_set.unit,
        load=load,
        save_every=500,
        work_dir=workdir,
        log_dir=log_dir,
    )

    test_set = DataSet(f"MD17.{args.name}", args.folder, mode="test").data
    info = test(model=model, dataset=test_set, load=workdir / "trained.pt")
    print(f"MD17.{args.name}", info)


if __name__ == "__main__":
    main()
