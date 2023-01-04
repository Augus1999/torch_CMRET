import os
import random
import pickle
from multiprocessing import Process
import numpy as np
from ase import Atoms
from ase.db import connect
from pyscf import dft
from pyscf import gto


def run(spin: str, point: int = 10) -> None:
    data = []

    mol = gto.Mole()

    angle = np.linspace(90, 140, point)
    length = np.linspace(0.95, 1.20, point)

    if spin == "singlet":
        S = 0
    if spin == "triplet":
        S = 1
    S_2 = 2 * S

    for ang in angle:
        for len1 in length:
            for len2 in length:
                r = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [len1, 0.0, 0.0],
                        [
                            len2 * np.cos(ang / 180 * np.pi),
                            len2 * np.sin(ang / 180 * np.pi),
                            0.0,
                        ],
                    ]
                )  # base coordinate
                s_a = random.randint(1, 179) / 180 * np.pi
                s = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(s_a), -np.sin(s_a)],
                        [0, np.sin(s_a), np.cos(s_a)],
                    ]
                )  # random rotate
                t1, t2, t3 = random.sample(range(1, 100), 3)
                t = np.array(
                    [
                        [1, 0, 0, t1 / 7],
                        [0, 1, 0, t2 / 7],
                        [0, 0, 1, t3 / 7],
                    ]
                )  # random transform
                r = (t @ np.vstack((s @ r.T, np.array([1, 1, 1])))).T
                mol_data = {}
                mol_data["Z"] = np.array([6, 1, 1])
                mol_data["R"] = r
                mol_data["S"] = np.array([float(S)])
                mol.atom = [["C", r[0]], ["H", r[1]], ["H", r[2]]]
                mol.basis = "cc-pVDZ"
                mol.spin = S_2
                mol.unit = "ANG"
                mol.build()
                mf_B3LYP = dft.RKS(mol)
                mf_B3LYP.xc = "B3LYP"
                energy = mf_B3LYP.kernel()
                mol_data["E"] = np.array([energy * 27.211386245988])  # convert to eV
                f = [[], [], []]
                for j in range(len(r)):
                    for k in range(3):
                        r_ = np.copy(r)
                        r_[j][k] += 1e-8
                        mol.atom = [["C", r_[0]], ["H", r_[1]], ["H", r_[2]]]
                        mol.basis = "cc-pVDZ"
                        mol.spin = S_2
                        mol.unit = "ANG"
                        mol.build()
                        mf_B3LYP = dft.RKS(mol)
                        mf_B3LYP.xc = "B3LYP"
                        energy2 = mf_B3LYP.kernel()
                        d_en = (energy2 - energy) * 27.211386245988
                        f_ = -d_en / 1e-8
                        f[j].append(f_)
                mol_data["F"] = np.array(f)
                data.append(mol_data)

    with open(os.path.join(os.getcwd(), f"CH2_1000_{spin}.pkl"), "wb") as fh:
        pickle.dump(data, fh)


if __name__ == "__main__":
    p1 = Process(
        target=run,
        args=("singlet", 10),
    )
    p2 = Process(
        target=run,
        args=("triplet", 10),
    )
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    with open(os.path.join(os.getcwd(), f"CH2_1000_singlet.pkl"), "rb") as f1:
        s_data = pickle.load(f1)
    with open(os.path.join(os.getcwd(), f"CH2_1000_triplet.pkl"), "rb") as f2:
        t_data = pickle.load(f2)
    random.shuffle(s_data)
    random.shuffle(t_data)
    train_data = s_data[:750] + t_data[:750]
    test_data = s_data[750:] + t_data[750:]
    with connect(os.path.join(os.getcwd(), f"ch2_2000_train.db")) as db1:
        for m in train_data:
            mol = Atoms(positions=m["R"], numbers=m["Z"])
            db1.write(mol, data={"S": m["S"], "E": m["E"], "F": m["F"]})
    with connect(os.path.join(os.getcwd(), f"ch2_2000_test.db")) as db2:
        for m in test_data:
            mol = Atoms(positions=m["R"], numbers=m["Z"])
            db2.write(mol, data={"S": m["S"], "E": m["E"], "F": m["F"]})
