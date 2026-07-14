"""Water partial RDFs (g_OO, g_OH, g_HH) from the MACE-OFF movie trajectory,
split into inter- and intramolecular contributions.

Neutron diffraction reports the intermolecular partials; the raw pair
histograms of the flexible molecules additionally contain the intramolecular
O-H bond (~0.98 A) and same-molecule H-H (~1.53 A) correlations, which are
kept here as separate columns/curves. Reference landmarks: the 25 C curves
in Fig. 4 of Tromp, Postorino, Neilson, Ricci & Soper, J. Chem. Phys. 101,
6210 (1994) for g_OH/g_HH, and Skinner et al., J. Chem. Phys. 138, 074506
(2013) x-ray values for g_OO.

Needs `ovito` (see requirements.txt) and validation/water_mace_movie.gsd
from water_movie_gsd.jl.

Run:  python py_water_partials.py
Outputs: validation/water_partial_rdfs.{csv,png}
"""
import os

import numpy as np
from ovito.io import import_file
import matplotlib.pyplot as plt

import pd_style

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "validation")

pd_style.apply()
pipe = import_file(os.path.join(VAL, "water_mace_movie.gsd"))
nfr = pipe.source.num_frames
d0 = pipe.compute(0)
types = np.array([d0.particles.particle_types.type_by_id(t).name
                  for t in d0.particles.particle_types[...]])
L = d0.cell[0, 0]
Nt = len(types)
assert Nt % 3 == 0 and np.all(types[0::3] == "O") \
    and np.all(types[1::3] == "H") and np.all(types[2::3] == "H"), \
    "expected O,H,H per-molecule ordering"
mol = np.arange(Nt) // 3
NO, NH = int((types == "O").sum()), int((types == "H").sum())
print(f"{nfr} frames, {NO} O + {NH} H, L = {L:.3f} A")

iu, ju = np.triu_indices(Nt, 1)
ti, tj = types[iu], types[ju]
same = mol[iu] == mol[ju]
m_oo = (ti == "O") & (tj == "O")
m_oh = ((ti == "O") & (tj == "H")) | ((ti == "H") & (tj == "O"))
m_hh = (ti == "H") & (tj == "H")

NB = 124
edges = np.linspace(0, L / 2, NB + 1)
mids = 0.5 * (edges[:-1] + edges[1:])
vshell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
V = L ** 3

masks = {"oo": m_oo, "oh_inter": m_oh & ~same, "oh_intra": m_oh & same,
         "hh_inter": m_hh & ~same, "hh_intra": m_hh & same}
counts = {k: np.zeros(NB) for k in masks}
for i in range(nfr):
    pos = np.asarray(pipe.compute(i).particles.positions)
    dif = pos[iu] - pos[ju]
    dif -= L * np.round(dif / L)
    dist = np.sqrt((dif ** 2).sum(-1))
    for k, m in masks.items():
        h, _ = np.histogram(dist[m], bins=edges)
        counts[k] += h

npairs = {"oo": NO * (NO - 1) / 2, "oh_inter": NO * NH, "oh_intra": NO * NH,
          "hh_inter": NH * (NH - 1) / 2, "hh_intra": NH * (NH - 1) / 2}
g = {k: counts[k] * V / (npairs[k] * vshell * nfr) for k in masks}

tail = (mids > 5.4) & (mids < 6.0)
print("tails (should be ~1):",
      " ".join(f"{k}={g[k][tail].mean():.3f}" for k in ("oo", "oh_inter", "hh_inter")))

def peak(gg, lo, hi):
    w = (mids >= lo) & (mids <= hi)
    i = np.argmax(gg[w])
    return mids[w][i], gg[w][i]

def valley(gg, lo, hi):
    w = (mids >= lo) & (mids <= hi)
    i = np.argmin(gg[w])
    return mids[w][i], gg[w][i]

p_oo = peak(g["oo"], 2.4, 3.2)
p_oh = peak(g["oh_inter"], 1.4, 2.6)
v_oh = valley(g["oh_inter"], 2.1, 2.9)
p_oh2 = peak(g["oh_inter"], 2.9, 3.8)
p_hh = peak(g["hh_inter"], 1.9, 3.0)
b_oh = peak(g["oh_intra"], 0.7, 1.3)
b_hh = peak(g["hh_intra"], 1.2, 1.8)
print(f"gOO  peak {p_oo[1]:.2f} @ {p_oo[0]:.2f} A   (x-ray 2.57 @ 2.80)")
print(f"gOH  H-bond peak {p_oh[1]:.2f} @ {p_oh[0]:.2f} A   (neutron ~1.9)")
print(f"gOH  first min {v_oh[1]:.2f} @ {v_oh[0]:.2f} A ; 2nd peak {p_oh2[1]:.2f} @ {p_oh2[0]:.2f}")
print(f"gHH  peak {p_hh[1]:.2f} @ {p_hh[0]:.2f} A   (neutron ~2.3)")
print(f"intra: O-H bond {b_oh[1]:.1f} @ {b_oh[0]:.3f} A ; H-H {b_hh[1]:.1f} @ {b_hh[0]:.3f} A")

hdr = "r_A,g_OO,g_OH_inter,g_HH_inter,g_OH_intra,g_HH_intra"
np.savetxt(os.path.join(VAL, "water_partial_rdfs.csv"),
           np.column_stack([mids, g["oo"], g["oh_inter"], g["hh_inter"],
                            g["oh_intra"], g["hh_intra"]]),
           delimiter=",", header=hdr, comments="")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4))

ax = axes[0]
ax.plot(mids, g["oo"], color=pd_style.NAVY, label="MACE-OFF (this work)")
ax.plot([2.80, 3.45, 4.50], [2.57, 0.84, 1.12], "s", color=pd_style.RED, ms=12,
        zorder=5, label="x-ray, Skinner et al. (2013)")
ax.set_xlim(2.0, 6.0); ax.set_ylim(0, 3.2)
ax.set_title(r"g$_{\mathrm{OO}}$(r)", pad=10)
pd_style.callout(ax, f"{p_oo[0]:.2f} Å\nvs 2.80 Å", xy=(p_oo[0] + 0.06, p_oo[1]),
                 xytext=(4.55, 1.85), fontsize=15)

ax = axes[1]
ax.plot(mids, g["oh_inter"] + g["oh_intra"], color="#b0b0b0", lw=1.8, ls="--",
        label="total (incl. intramolecular)")
ax.plot(mids, g["oh_inter"], color=pd_style.NAVY, label="intermolecular")
ax.plot([1.82, 2.50, 3.34], [1.00, 0.29, 1.40], "s", color=pd_style.RED, ms=12,
        zorder=5, label="neutron 25 °C, Tromp et al. (1994)")
ax.set_xlim(0.5, 6.0); ax.set_ylim(0, 2.4)
ax.set_title(r"g$_{\mathrm{OH}}$(r)", pad=10)
ax.text(b_oh[0] + 0.14, 1.66, f"O–H bond\n{b_oh[0]:.2f} Å\n(g ≈ {b_oh[1]:.0f}, off scale)",
        fontsize=13, fontweight="bold", color=pd_style.GRAY_SUB, va="top")
pd_style.callout(ax, f"H-bond\n{p_oh[0]:.2f} Å vs ~1.9 Å", xy=(p_oh[0] + 0.05, p_oh[1] + 0.03),
                 xytext=(4.2, 0.45), fontsize=15)

ax = axes[2]
ax.plot(mids, g["hh_inter"] + g["hh_intra"], color="#b0b0b0", lw=1.8, ls="--",
        label="total (incl. intramolecular)")
ax.plot(mids, g["hh_inter"], color=pd_style.NAVY, label="intermolecular")
ax.plot([2.31, 2.94, 3.72], [1.20, 0.82, 1.18], "s", color=pd_style.RED, ms=12,
        zorder=5, label="neutron 25 °C, Tromp et al. (1994)")
ax.set_xlim(0.5, 6.0); ax.set_ylim(0, 2.4)
ax.set_title(r"g$_{\mathrm{HH}}$(r)", pad=10)
ax.text(b_hh[0] - 0.02, 1.82, f"H–H same\nmolecule {b_hh[0]:.2f} Å\n(g ≈ {b_hh[1]:.0f}, off scale)",
        fontsize=13, fontweight="bold", color=pd_style.GRAY_SUB, va="top", ha="left")
pd_style.callout(ax, f"{p_hh[0]:.2f} Å vs ~2.3 Å", xy=(p_hh[0] + 0.06, p_hh[1] + 0.02),
                 xytext=(4.7, 0.45), fontsize=15)

for ax in axes:
    ax.axhline(1, color="gray", lw=0.8)
    ax.grid(True, axis="y")
    ax.set_xlabel("r (Å)")
    ax.legend(loc="upper right", fontsize=12)
axes[0].set_ylabel("g(r)")

pd_style.title(fig, "Liquid-water partial structure: foundation MLIP vs diffraction",
               subtitle="ParticleDynamics.jl + MACE-OFF23(S) — 64 H₂O, 300 K, fully flexible molecules, no classical force field", y=0.995)
fig.text(0.5, 0.034,
         "Neutron (25 °C curve, Fig. 4): Tromp, Postorino, Neilson, Ricci & Soper, J. Chem. Phys. 101, 6210 (1994) · "
         "X-ray: Skinner et al., J. Chem. Phys. 138, 074506 (2013)",
         ha="center", fontsize=11, color=pd_style.GRAY_SUB)
fig.text(0.5, 0.010,
         "Experiments report intermolecular partials; the intramolecular contributions (dashed gray) come out of the flexible-molecule dynamics directly.",
         ha="center", fontsize=11, color=pd_style.GRAY_SUB)
fig.tight_layout(rect=(0, 0.055, 1, 0.90))
fig.savefig(os.path.join(VAL, "water_partial_rdfs.png"))
print("wrote water_partial_rdfs.png and water_partial_rdfs.csv")
