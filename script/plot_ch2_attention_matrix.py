# -*- coding: utf-8 -*-
"""
Plot the attention matrix.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from cmret.models import trained_model

model = trained_model(name="carbene")

alphas2 = []
label_map = {0: "106", 1: "180"}
r3s = [[-0.3032, 1.0573, 0.0], [-1.1, 0, 0]]
Z = torch.tensor([[6, 1, 1]], dtype=torch.long)
labels = ["C", "H", "H"]

for i in range(len(r3s)):
    R = torch.tensor([[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], r3s[i]]])
    mol = {"Z": Z, "R": R}
    out = model(mol, return_attn_matrix=True, average_attn_matrix_over_layers=False)
    alphas = out["attn_matrix"]
    alphas2.append(alphas)
    print(out["energy"].item(), "kcal/mol")

vmax = np.vstack([i.detach().numpy() for i in alphas2[0] + alphas2[1]]).max()
vmin = np.vstack([i.detach().numpy() for i in alphas2[0] + alphas2[1]]).min()

for key, alphas in enumerate(alphas2):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    plt.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for i in range(3):
        ax = axes[0][i]
        a = alphas[i][0].detach().numpy()
        im = ax.imshow(a, cmap=plt.get_cmap("RdBu_r"), vmax=vmax, vmin=vmin)
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.set_title(f"T = {i + 1}")
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                t = str("%e" % a[i, j]).split("e")
                t = f"{round(float(t[0]), 2)}{'e' if t[-1] != '' else ''}{t[-1]}"
                ax.text(
                    j,
                    i,
                    t,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="w"
                    if (
                        float(t) >= (3 * vmax - vmin) / 4
                        or float(t) <= (vmax - 3 * vmin) / 4
                    )
                    else "black",
                )
    for i in range(3, 6):
        ax = axes[1][i - 3]
        if i >= len(alphas):
            ax.axis("off")
            continue
        a = alphas[i][0].detach().numpy()
        im = ax.imshow(a, cmap=plt.get_cmap("RdBu_r"), vmax=vmax, vmin=vmin)
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.set_title(f"T = {i + 1}")
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                t = str("%e" % a[i, j]).split("e")
                t = f"{round(float(t[0]), 2)}{'e' if t[-1] != '' else ''}{t[-1]}"
                ax.text(
                    j,
                    i,
                    t,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="w"
                    if (
                        float(t) >= (3 * vmax - vmin) / 4
                        or float(t) <= (vmax - 3 * vmin) / 4
                    )
                    else "black",
                )

    plt.suptitle(
        f"Attention matrix of Carbene (H-C-H angle of {label_map[key]}°) in each Interaction block."
    )
    plt.tight_layout()
    plt.savefig(
        f"attention_matrix_CH2_{key+1}.png", dpi=250
    )  # you can change the file names here
    plt.show()  # ご注意：comment this line if running this script on supercomputer
    #        because it has no graphic interface to display images.
