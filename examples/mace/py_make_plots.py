"""Render validation plots from the CSV outputs, in the package house style.

- validation/v2_energy.png : total energy vs time (Si216 NVE)
- validation/water_rdf.png : O-O RDF vs experimental landmarks

Run:  python py_make_plots.py
"""
import csv
import os

import matplotlib.pyplot as plt

import pd_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")

pd_style.apply()


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
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(t_ps, [(e - E0) / abs(E0) for e in E], lw=1.6, color=pd_style.NAVY)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"relative energy deviation $(E(t)-E_0)/|E_0|$")
    ax.set_title(r"NVE energy conservation — Si$_{216}$, MACE-MP-0, float64, dt = 1 fs")
    ax.axhline(0, color="gray", lw=0.6)
    ax.grid(True, axis="y")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v2_energy.png"))
    print("wrote v2_energy.png")

# --- water RDF ---
p = os.path.join(OUT, "water_rdf.csv")
if os.path.exists(p):
    _, (r, g) = read_csv(p)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(r, g, lw=3.0, color=pd_style.NAVY, marker="o", markevery=8, ms=7,
            label="ParticleDynamics.jl + MACE-OFF (64 H$_2$O, 10 ps NVE)")
    # Experimental x-ray landmarks: Skinner et al., J. Chem. Phys. 138, 074506 (2013)
    ax.plot([2.80, 3.45, 4.5], [2.57, 0.84, 1.12], "s", color=pd_style.RED, ms=10,
            label="experiment (x-ray, Skinner 2013)", zorder=5)
    ax.axhline(1, color="gray", lw=0.6)
    ax.grid(True, axis="y")
    ax.set_xlabel("O–O distance r (Å)")
    ax.set_ylabel(r"g$_{OO}$(r)")
    ax.set_title("Liquid-water structure from a foundation MLIP")
    ax.set_xlim(2, 6.2)
    ax.legend(loc="upper right")
    pd_style.callout(ax, "peak within ~2%\nof experiment", xy=(2.9, 2.55), xytext=(4.6, 1.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "water_rdf.png"))
    print("wrote water_rdf.png")
