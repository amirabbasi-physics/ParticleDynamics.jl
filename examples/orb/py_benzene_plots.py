"""Figures for the benzene crystal MACE-OFF vs Orb-v3 head-to-head.

Reads the artefacts written by the Julia scripts in validation/ and produces:

  benzene_ev_scan.png  -- E(V) per molecule for each model, experimental cell marked
  benzene_vdos.png     -- vibrational DOS vs experimental Raman frequencies
  benzene_nve.png      -- NVE energy conservation: conservative vs direct forces

Run: python examples/orb/py_benzene_plots.py
"""

import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "validation")
sys.path.insert(0, os.path.join(HERE, "..", "mace"))
import pd_style as S  # noqa: E402

S.apply()
import matplotlib.pyplot as plt  # noqa: E402

C_CM_PER_S = 2.99792458e10
EV_TO_KJMOL = 96.48533212331

# Experimental Raman frequencies of benzene (cm^-1): ring breathing, CC
# stretch, and aromatic C-H stretch.
RAMAN = [(992, "ring breathing"), (1586, "C–C stretch"), (3062, "C–H stretch")]

MODEL_ORDER = ["MACE-OFF23-small", "Orb-v3-cons-inf-omat", "Orb-v3-cons-omol",
               "Orb-v3-direct-inf-omat"]
KEY_LABEL = {"mace-off": "MACE-OFF23-small",
             "orb-cons": "Orb-v3-cons-inf-omat",
             "orb-omol": "Orb-v3-cons-omol",
             "orb-direct": "Orb-v3-direct-inf-omat"}


def load_scoreboard():
    """model label -> lattice energy (kJ/mol) from the scan summary."""
    out = {}
    p = os.path.join(VAL, "benzene_lattice_energy.txt")
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if "E_latt" in f:
                out[f[0]] = float(f[f.index("E_latt") + 2])
    return out


def style_for(label):
    """Fixed colour/marker per model, never cycled on filtering."""
    idx = MODEL_ORDER.index(label) if label in MODEL_ORDER else len(MODEL_ORDER) - 1
    return S.COLORS[idx], S.MARKERS[idx]


# ---------------------------------------------------------------- E(V) scan
def plot_ev():
    path = os.path.join(VAL, "benzene_ev_scan.csv")
    if not os.path.exists(path):
        print("skip E(V): no benzene_ev_scan.csv")
        return
    rows = [l.strip().split(",") for l in open(path).readlines()[1:] if l.strip()]
    if not rows:
        print("skip E(V): csv empty")
        return
    models = []
    for r in rows:
        if r[0] not in models:
            models.append(r[0])

    # Lattice energies from the summary file, so the legend is self-contained.
    elatt = load_scoreboard()

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    # Experimental cells: 462 Å^3 at the lowest temperatures up to 494 Å^3 for
    # the 150 K entry the scan starts from (see REFERENCES.md).
    ax.axvspan(462.0, 494.3, color="black", alpha=0.09,
               label="experiment, 4–150 K")
    for m in models:
        sel = [r for r in rows if r[0] == m]
        V = np.array([float(r[2]) for r in sel])
        E = np.array([float(r[4]) for r in sel]) * EV_TO_KJMOL
        E = E - E.min()
        col, mk = style_for(m)
        lab = m if m not in elatt else f"{m}   ($E_{{latt}}$ = {elatt[m]:.1f})"
        ax.plot(V, E, marker=mk, color=col, label=lab, markersize=7)
        k = np.argsort(E)[:3]
        if len(k) == 3 and len(set(V[k])) == 3:
            c = np.polyfit(V[k], E[k], 2)
            if c[0] > 0:
                ax.axvline(-c[1] / (2 * c[0]), color=col, lw=1.3, ls=":", alpha=0.85)

    ax.set_xlabel("unit-cell volume  (Å$^3$)")
    ax.set_ylabel("energy per molecule  (kJ/mol)")
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1, title="$E_{latt}$ in kJ/mol; "
              "experiment $-55.3 \\pm 2.2$")
    ax.get_legend().get_title().set_fontsize(11)
    S.title(fig, "Benzene crystal I: energy–volume curves",
            "same engine, same protocol — atomic positions relaxed at fixed cell "
            "by overdamped Langevin dynamics")
    fig.tight_layout()
    out = os.path.join(VAL, "benzene_ev_scan.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------------------------- VDOS
def vdos(vel, masses, dt_fs):
    """Mass-weighted velocity power spectrum -> (wavenumber cm^-1, intensity)."""
    n = vel.shape[0]
    w = np.hanning(n)[:, None, None]
    F = np.fft.rfft(vel * w, axis=0)
    p = (np.abs(F) ** 2).sum(axis=2)          # sum over x,y,z
    g = (p * masses[None, :]).sum(axis=1)     # mass-weighted sum over atoms
    f = np.fft.rfftfreq(n, d=dt_fs * 1e-15)   # Hz
    return f / C_CM_PER_S, g


def plot_vdos():
    files = sorted(glob.glob(os.path.join(VAL, "benzene_md_*.npz")))
    if not files:
        print("skip VDOS: no benzene_md_*.npz")
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    plotted = 0
    for fp in files:
        key = os.path.basename(fp)[len("benzene_md_"):-len(".npz")]
        if key == "orb-direct":
            # Excluded on purpose: this run's energy was not conserved, so its
            # velocities sample a drifting temperature (150 K -> 246 K) rather
            # than the 150 K the other spectra are measured at.
            continue
        d = np.load(fp, allow_pickle=True)
        label = KEY_LABEL.get(key, key)
        k, g = vdos(d["velocities"], d["masses"], float(d["dt_fs"]))
        sel = k < 3600
        g = g[sel] / g[sel].max()
        col, _ = style_for(label)
        ax.plot(k[sel], g, color=col, label=label, lw=2.2)
        plotted += 1
    if plotted == 0:
        print("skip VDOS: nothing to plot")
        plt.close(fig)
        return

    for nu, name in RAMAN:
        ax.axvline(nu, color="black", lw=1.6, ls="--", alpha=0.75)
        ax.text(nu, 1.02, f"{nu}\n{name}", fontsize=10, ha="center",
                color=S.GRAY_SUB)

    ax.set_xlim(0, 3600)
    ax.set_ylim(0, 1.14)
    ax.set_xlabel("wavenumber  (cm$^{-1}$)")
    ax.set_ylabel("vibrational DOS  (norm.)")
    ax.grid(True)
    # The 1700-3000 cm^-1 window is empty for benzene, so the legend goes there
    # rather than over the C-H stretch label.
    ax.legend(loc="upper left", bbox_to_anchor=(0.485, 0.93))
    S.title(fig, "Benzene crystal I at 150 K: vibrational spectrum",
            "5 ps NVE, mass-weighted velocity power spectrum; dashed lines are "
            "experimental Raman frequencies")
    fig.tight_layout()
    out = os.path.join(VAL, "benzene_vdos.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------- NVE energy conservation
def plot_nve():
    files = sorted(glob.glob(os.path.join(VAL, "benzene_md_*.npz")))
    if not files:
        print("skip NVE: no benzene_md_*.npz")
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    plotted = 0
    for fp in files:
        key = os.path.basename(fp)[len("benzene_md_"):-len(".npz")]
        d = np.load(fp, allow_pickle=True)
        label = KEY_LABEL.get(key, key)
        t = d["etimes"]
        e = d["etot"]
        n_at = len(d["numbers"])
        dev = (e - e[0]) / n_at * 1000.0       # meV per atom
        col, _ = style_for(label)
        ax.plot(t, dev, color=col, label=label, lw=2.2)
        plotted += 1
        if key == "orb-direct":
            # Spell out what the drift means physically: this NVE run was
            # started at 150 K and the non-conservative forces heated it.
            tmean = float(np.mean(d["temps"]))
            S.callout(ax,
                      f"+{dev[-1]:.0f} meV/atom in {t[-1]:.0f} ps\n"
                      f"NVE self-heats 150 K $\\rightarrow$ {tmean:.0f} K",
                      xy=(t[-1] * 0.93, dev[-1] * 0.97),
                      xytext=(t[-1] * 0.52, dev[-1] * 0.60), fontsize=13)
    if plotted == 0:
        plt.close(fig)
        return
    ax.axhline(0.0, color="black", lw=1.2, ls=":")
    ax.set_xlabel("time  (ps)")
    ax.set_ylabel("total-energy drift  (meV/atom)")
    ax.grid(True)
    ax.legend(loc="best")
    S.title(fig, "NVE energy conservation: conservative vs direct forces",
            "identical integrator and timestep; direct-force models predict "
            "forces that are not the gradient of any energy")
    fig.tight_layout()
    out = os.path.join(VAL, "benzene_nve.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_scorecard():
    """Speed vs accuracy: the one-figure summary of the head-to-head."""
    tp = os.path.join(VAL, "benzene_throughput.txt")
    if not os.path.exists(tp):
        print("skip scorecard: no benzene_throughput.txt")
        return
    # label -> steps/s at the matched precision used for production
    rates = {}
    for line in open(tp):
        f = line.split()
        if "steps/s" not in line:
            continue
        label, prec = f[0], f[1]
        rate = float(f[f.index("steps/s") - 1])
        # keep the float32 row for each model (matched-precision comparison)
        if prec.startswith("float32"):
            rates[label] = rate
    elatt = load_scoreboard()
    drift = {}
    for fp in sorted(glob.glob(os.path.join(VAL, "benzene_md_*.npz"))):
        key = os.path.basename(fp)[len("benzene_md_"):-len(".npz")]
        d = np.load(fp, allow_pickle=True)
        drift[KEY_LABEL.get(key, key)] = float(d["drift_rel"])

    pts = [(lab, rates[lab], abs(elatt[lab] - (-55.3)))
           for lab in rates if lab in elatt]
    if not pts:
        print("skip scorecard: no overlapping speed/accuracy data")
        return

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    xmax = max(list(rates.values())) * 1.18
    ymax = max(e for _, _, e in pts) * 1.22
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    for lab, rate, err in pts:
        col, mk = style_for(lab)
        ax.scatter([rate], [err], s=340, color=col, marker=mk, zorder=5,
                   edgecolor="white", linewidth=2)
        note = lab
        if lab in drift:
            note += f"\nNVE |dE/E| = {drift[lab]:.1e}"
        # Flip the label to the inside when the point sits in the right half,
        # so long model names cannot run off the axes.
        right = rate > 0.62 * xmax
        ax.annotate(note, (rate, err),
                    xytext=(-14 if right else 14, 0), textcoords="offset points",
                    fontsize=12, fontweight="bold", color=col,
                    va="center", ha="right" if right else "left")

    # direct-force model: fastest, but unusable in NVE -- mark it separately
    dlab = "Orb-v3-direct-inf-omat"
    if dlab in rates and dlab in drift:
        col, _ = style_for(dlab)
        ax.axvline(rates[dlab], color=col, lw=2.0, ls=":")
        ax.text(rates[dlab] - 0.12, ymax * 0.60,
                f"{dlab}\nfastest, but NVE |dE/E| = {drift[dlab]:.1e}\n"
                f"— no conserved energy",
                fontsize=11.5, fontweight="bold", color=col, va="center",
                ha="right")

    ax.set_xlabel("throughput  (steps/s, 384 atoms, float32)")
    ax.set_ylabel("|lattice-energy error|  (kJ/mol)")
    ax.grid(True)
    S.title(fig, "Benzene crystal: speed vs accuracy",
            "lower and further right is better; error is against the "
            "experimental $-55.3 \\pm 2.2$ kJ/mol")
    fig.tight_layout()
    out = os.path.join(VAL, "benzene_scorecard.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def report_peaks():
    """Print located VDOS peaks near each experimental Raman line."""
    files = sorted(glob.glob(os.path.join(VAL, "benzene_md_*.npz")))
    for fp in files:
        key = os.path.basename(fp)[len("benzene_md_"):-len(".npz")]
        d = np.load(fp, allow_pickle=True)
        k, g = vdos(d["velocities"], d["masses"], float(d["dt_fs"]))
        print(f"\n{KEY_LABEL.get(key, key)}:")
        for nu, name in RAMAN:
            win = (k > nu - 150) & (k < nu + 150)
            if win.sum() == 0:
                continue
            peak = k[win][np.argmax(g[win])]
            print(f"  {name:16s} exp {nu:5d}  ->  model {peak:7.1f} cm^-1"
                  f"  (dev {peak - nu:+6.1f})")


if __name__ == "__main__":
    plot_ev()
    plot_vdos()
    plot_nve()
    plot_scorecard()
    report_peaks()
