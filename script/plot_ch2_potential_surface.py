import numpy as np
import matplotlib.pyplot as plt


N = 100
d = np.linspace(0.9, 1.4, N)
x = np.linspace(90, 180, N)
cmret1 = np.load("ch2_potential_surface/cmret.npz")
cmret2 = np.load("ch2_potential_surface/cmret_t1d.npz")
cmret3 = np.load("ch2_potential_surface/cmret_swish.npz")
painn = np.load("ch2_potential_surface/painn.npz")
reference = np.load("ch2_potential_surface/reference.npz")
s_r, t_r = reference["s"], reference["t"]
s_m1, t_m1 = cmret1["s"], cmret1["t"]
s_m2, t_m2 = cmret2["s"], cmret2["t"]
s_m3, t_m3 = cmret3["s"], cmret3["t"]
s_p, t_p = painn["s"], painn["t"]
data = [[s_m1, s_m2, s_m3, s_p, s_r], [t_m1, t_m2, t_m3, t_p, t_r]]
vmax = np.array([s_r, t_r]).max()
vmin = np.array([s_r, t_r]).min()
title1 = ["Singlet", "Triplet"]
title2 = [
    "softmax $\\tau=\\sqrt{2d}$",
    "softmax $\\tau=\\sqrt{d}$",
    "Swish",
    "PAINN",
    "Reference",
]

fig, axes = plt.subplots(2, 5, figsize=[15, 6])
for i in range(2):
    ax = axes[i]
    levels = (
        -np.array([1064, 1063.5, 1063, 1062.5, 1062, 1061.5, 1061, 1060.5])
        if i == 0
        else -np.array([1064.4, 1064, 1063.6, 1063.2, 1062.8, 1062.4, 1062, 1061.6])
    )
    for j in range(5):
        ax_ = ax[j]
        ax_.set_aspect(0.5 / 90)
        cp = ax_.contourf(
            d, x, data[i][j], 100, cmap=plt.get_cmap("magma"), vmax=vmax, vmin=vmin
        )
        ax_.plot(
            np.array([0.95, 0.95, 1.20, 1.20]),
            np.array([90, 140, 140, 90]),
            color="grey",
            linestyle="--",
        )
        ct = ax_.contour(
            d,
            x,
            data[i][j],
            levels,
            colors="#f8f8f2",
            linestyles="solid",
            linewidths=1.0,
        )
        plt.clabel(ct, fontsize=9.5, colors="#f8f8f2")
        ax_.set_title(f"{title2[j]}, {title1[i]}")
plt.tight_layout()
plt.savefig("CH2_2d.pdf", dpi=250)
plt.show()
