"""Render validation plots from the CSV outputs.

- validation/v2_energy.png : total energy vs time (Si216 NVE, drift inset)
- validation/water_rdf.png : O-O RDF vs experimental landmarks

Run:  python py_make_plots.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        cols = list(zip(*[[float(x) for x in row] for row in r]))
    return header, cols


# --- V2 energy conservation ---
p = os.path.join(OUT, "v2_energy.csv")
if os.path.exists(p):
    _, (steps, E) = read_csv(p)
    t_ps = [s * 1e-3 for s in steps]  # 1 fs steps -> ps
    E0 = E[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t_ps, [(e - E0) / abs(E0) for e in E], lw=0.8, color="#1f6feb")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("relative energy deviation  (E(t) − E₀)/|E₀|")
    ax.set_title("NVE energy conservation — Si₂₁₆, MACE-MP-0 (small), float64, dt = 1 fs")
    ax.axhline(0, color="gray", lw=0.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v2_energy.png"), dpi=160)
    print("wrote v2_energy.png")

# --- water RDF ---
p = os.path.join(OUT, "water_rdf.csv")
if os.path.exists(p):
    _, (r, g) = read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r, g, lw=1.6, color="#1f6feb",
            label="ParticleDynamics.jl + MACE-OFF (64 H₂O, 10 ps NVE)")
    # Experimental x-ray landmarks: Skinner et al., J. Chem. Phys. 138,
    # 074506 (2013), ambient water
    ax.plot([2.80], [2.57], "o", color="#d1242f", ms=7,
            label="exp. first peak (Skinner 2013): g=2.57 @ 2.80 Å")
    ax.plot([3.45, 4.5], [0.84, 1.12], "s", color="#bf8700", ms=6,
            label="exp. first min / second peak")
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g$_{OO}$(r)")
    ax.set_title("Liquid water structure from a foundation MLIP")
    ax.set_xlim(2, 6.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "water_rdf.png"), dpi=160)
    print("wrote water_rdf.png")
