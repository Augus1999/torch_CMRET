# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
train and test on singlet/triplet CH2 dataset
"""
import argparse
from pathlib import Path
from cmret.utils import train, test, ASEData
from cmret import CMRETModel


root = Path(__file__).parent

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
    log_dir = root / "logs/ch2.log"

    dataset = ASEData(f"{args.folder}/ch2_2000_train.db")
    model = CMRETModel(n_interaction=args.nlayer, rbf_type=args.rbf, num_head=args.nh, n_kernel=20)

    train(
        model=model,
        datasets=[dataset],
        batch_sizes=[args.batchsize],
        max_n_epochs=args.nepoch,
        work_dir=workdir,
        log_dir=log_dir,
    )

    test_set = ASEData(f"{args.folder}/ch2_2000_test.db")
    test_info = test(model=model, dataset=test_set, load=workdir / "trained.pt")
    print("CH2:", test_info)


if __name__ == "__main__":
    main()
