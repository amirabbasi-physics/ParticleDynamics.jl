#!/usr/bin/env python3
"""Water NPT analysis: speed and accuracy of the head-to-head runs.

Reads RUNS_DIR/<N>/<engine>/ (run_all.sh) and writes summary.json, summary.md and
figures (house style, examples/mace/pd_style.py):
  speed      ms/step of the timed loop (each engine's own loop timer), ns/day, wall time of the
             whole process (setup included), mean GPU utilization over the loop window
  density    final frame (all engines; one-frame spread sqrt(kT kappa / V)), and the in-memory
             average over the run where the engine keeps one (GROMACS energy averages,
             LAMMPS fix ave/time)
  structure  partial RDFs g_OO, g_OH, g_HH of the final frame: RMS difference to GROMACS over
             0.2-1.2 nm, divided by the combined counting noise of the two histograms (z ~ 1 means
             statistically indistinguishable), and the first g_OO peak
Usage: analyze.py RUNS_DIR OUT_DIR
"""
import json, math, os, re, sys, tomllib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "mace"))
import pd_style
import matplotlib.pyplot as plt

RUNS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SIZES = [10000, 100000, 500000]
STEPS = 10000
DT_PS = 0.002
# engine key -> (label, color index, marker index, linestyle, filled, device)
ENGINES = {
    "pd_pme": ("ParticleDynamics PME", 0, 0, "-", True, "GPU"),
    "pd_esp": ("ParticleDynamics ESP", 0, 0, "--", True, "GPU"),
    "openmm": ("OpenMM PME", 1, 1, "-", True, "GPU"),
    "gromacs": ("GROMACS PME", 2, 2, "-", True, "GPU"),
    "gromacs_esp": ("GROMACS-ESP (16 CPU cores)", 2, 2, "--", False, "CPU"),
    "lammps": ("LAMMPS PPPM (KOKKOS)", 3, 3, "-", True, "GPU"),
    "lammps_esp": ("LAMMPS-ESP (16 CPU cores)", 3, 3, "--", False, "CPU"),
}
KB = 1.380649e-23; KAPPA = 4.5e-10; T = 298.0
MW = (15.9994 + 2 * 1.008) / 6.02214076e23 * 1e-3   # kg per water


def read(path):
    try:
        return open(path).read()
    except OSError:
        return None


def loop_seconds(d, e):
    """Timed loop of the engine's own timer (s), or None."""
    if e in ("pd_pme", "pd_esp", "openmm"):
        r = read(os.path.join(d, "record.json"))
        return json.loads(r)["loop_s"] if r else None, (STEPS - 1 if e.startswith("pd") else STEPS)
    if e in ("gromacs", "gromacs_esp"):
        t = read(os.path.join(d, "run", "npt.log"))
        m = t and re.search(r"Time:\s+([0-9.]+)\s+([0-9.]+)", t)
        return (float(m.group(2)), STEPS) if m else (None, STEPS)
    t = read(os.path.join(d, "run", "log.lammps"))
    m = t and re.search(r"Loop time of ([0-9.eE+-]+) on", t)
    return (float(m.group(1)), STEPS) if m else (None, STEPS)


def averaged_density(d, e, nw):
    """In-memory run average (kg/m^3) where the engine keeps one."""
    if e in ("gromacs", "gromacs_esp"):
        t = read(os.path.join(d, "run", "energy.log"))
        m = t and re.search(r"^Density\s+([0-9.]+)", t, re.M)
        return float(m.group(1)) if m else None
    if e in ("lammps", "lammps_esp"):
        t = read(os.path.join(d, "run", "volume_average.txt"))
        if not t:
            return None
        rows = [l.split() for l in t.splitlines() if l.strip() and not l.startswith("#")]
        return nw * MW / (float(rows[-1][1]) * 1e-30) if rows else None
    return None


def gpu_utilization(d, loop_s):
    """Mean utilization over the last `loop_s` seconds before the last active sample."""
    t = read(os.path.join(d, "gpu_util.csv"))
    if not t or not loop_s:
        return None
    ts, us = [], []
    for l in t.splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) < 2:
            continue
        try:
            h, mi, s = p[0].split(" ")[1].split(":")
            ts.append(int(h) * 3600 + int(mi) * 60 + float(s)); us.append(float(p[1].split()[0]))
        except (ValueError, IndexError):
            continue
    act = [i for i, u in enumerate(us) if u >= 1]
    if not act:
        return 0.0
    t1 = ts[act[-1]]; t0 = t1 - loop_s
    w = [u for tt, u in zip(ts, us) if t0 <= tt <= t1]
    return sum(w) / len(w) if len(w) >= 3 else None


def rdf(d):
    t = os.path.join(d, "rdf.toml")
    return tomllib.load(open(t, "rb")) if os.path.exists(t) else None


def counting_sigma(r, g, nw, box, dr, pair):
    """Poisson spread of g(r) from the histogram counts of one frame."""
    V = box[0] * box[1] * box[2]
    nO, nH = nw, 2 * nw
    pairs = {"g_OO": nO * (nO - 1) / 2, "g_OH": nO * nH, "g_HH": nH * (nH - 1) / 2}[pair]
    shell = 4 * math.pi / 3 * ((r + dr / 2) ** 3 - (r - dr / 2) ** 3)
    ideal = pairs * shell / V
    return np.sqrt(np.maximum(g, 1e-12) / ideal)


res = {}
for N in SIZES:
    for e in ENGINES:
        d = os.path.join(RUNS, str(N), e)
        if not os.path.exists(os.path.join(d, "exit.txt")):
            continue
        ok = read(os.path.join(d, "exit.txt")).strip() == "0"
        loop, nsteps = loop_seconds(d, e) if ok else (None, STEPS)
        R = rdf(d)
        nw = N
        rec = {"ok": ok, "loop_s": loop, "ms_per_step": 1e3 * loop / nsteps if loop else None,
               "ns_per_day": nsteps * DT_PS * 1e-3 / (loop / 86400) if loop else None,
               "wall_s": float(read(os.path.join(d, "wall_s.txt")) or "nan"),
               "gpu_util": gpu_utilization(d, loop) if ENGINES[e][5] == "GPU" else None,
               "density_final": R["density_kg_m3"] if R else None,
               "density_average": averaged_density(d, e, nw)}
        if R:
            V = R["box"][0] * R["box"][1] * R["box"][2] * 1e-27
            rec["density_frame_sigma"] = rec["density_final"] * math.sqrt(KB * T * KAPPA / V)
            r = np.array(R["r"]); gOO = np.array(R["g_OO"]); k = int(np.argmax(gOO))
            rec["gOO_peak"] = [float(r[k]), float(gOO[k])]
        res[(N, e)] = (rec, R)

# structure: differences to GROMACS (same size), in units of the combined counting noise
for N in SIZES:
    ref = res.get((N, "gromacs"), (None, None))[1]
    if ref is None:
        continue
    r = np.array(ref["r"]); sel = (r >= 0.2) & (r <= 1.2)
    for e in ENGINES:
        if (N, e) not in res or res[(N, e)][1] is None or e == "gromacs":
            continue
        rec, R = res[(N, e)]
        for pair in ("g_OO", "g_OH", "g_HH"):
            g1 = np.array(R[pair]); g0 = np.array(ref[pair])
            s1 = counting_sigma(r, g1, N, R["box"], R["dr"], pair)
            s0 = counting_sigma(r, g0, N, ref["box"], ref["dr"], pair)
            diff = (g1 - g0)[sel]; sig = np.sqrt(s1 ** 2 + s0 ** 2)[sel]
            rec["rdf_rms_diff_" + pair] = float(np.sqrt(np.mean(diff ** 2)))
            rec["rdf_z_" + pair] = float(np.sqrt(np.mean((diff / sig) ** 2)))

summary = {f"{N}/{e}": rec for (N, e), (rec, _) in res.items()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ---------------- markdown -----------------
lines = ["# Rigid SPC/E water NPT, 298 K / 1 atm, 10,000 steps", ""]
for N in SIZES:
    lines += [f"## {N:,} waters ({3 * N:,} atoms)", "",
              "| Engine | ms/step | ns/day | wall s | GPU util % | ρ final (kg/m³) | ρ run avg | z(g_OO, g_OH, g_HH) vs GROMACS |",
              "|---|---|---|---|---|---|---|---|"]
    for e, (label, *_rest) in ENGINES.items():
        if (N, e) not in res:
            continue
        rec = res[(N, e)][0]
        f = lambda x, fmt: (fmt % x) if isinstance(x, (int, float)) and x is not None and not (isinstance(x, float) and math.isnan(x)) else "—"
        z = ", ".join(f(rec.get("rdf_z_" + p), "%.2f") for p in ("g_OO", "g_OH", "g_HH")) if e != "gromacs" else "reference"
        dens = f(rec["density_final"], "%.2f") + (" ± " + f(rec.get("density_frame_sigma"), "%.2f") if rec.get("density_frame_sigma") else "")
        lines.append(f"| {label} | {f(rec['ms_per_step'], '%.3f')} | {f(rec['ns_per_day'], '%.1f')} | {f(rec['wall_s'], '%.1f')} | "
                     f"{f(rec['gpu_util'], '%.0f')} | {dens} | {f(rec['density_average'], '%.2f')} | {z} |")
    lines.append("")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))

# ---------------- figures -----------------
pd_style.apply()
C, MK = pd_style.COLORS, pd_style.MARKERS
# speed: ms/step vs atoms (one axis, log-log)
fig, ax = plt.subplots(figsize=(8.5, 6))
for e, (label, ci, mi, ls, filled, dev) in ENGINES.items():
    pts = [(3 * N, res[(N, e)][0]["ms_per_step"]) for N in SIZES if (N, e) in res and res[(N, e)][0]["ms_per_step"]]
    if not pts:
        continue
    x, y = zip(*pts)
    ax.plot(x, y, ls=ls, color=C[ci], marker=MK[mi], ms=9, lw=2,
            mfc=C[ci] if filled else "white", mec=C[ci], mew=2, label=label)
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major")
ax.set_xlabel("Atoms"); ax.set_ylabel("Time per step (ms)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
pd_style.title(fig, "Rigid SPC/E water NPT: time per step",
               "298 K, 1 atm, 10,000 steps, single precision, matched Coulomb force error (GPU unless noted)")
fig.savefig(os.path.join(OUT, "water_speed.png"), bbox_inches="tight"); plt.close(fig)

# structure: g(r) of the largest size with all engines, and the differences to GROMACS
Nbig = max([N for N in SIZES if (N, "gromacs") in res and res[(N, "gromacs")][1]] or [None])
if Nbig:
    ref = res[(Nbig, "gromacs")][1]; r = np.array(ref["r"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for j, pair in enumerate(("g_OO", "g_OH", "g_HH")):
        for e, (label, ci, mi, ls, filled, dev) in ENGINES.items():
            if (Nbig, e) not in res or res[(Nbig, e)][1] is None:
                continue
            g = np.array(res[(Nbig, e)][1][pair])
            axes[0, j].plot(r, g, ls=ls, color=C[ci], lw=1.6, label=label)
            if e != "gromacs":
                axes[1, j].plot(r, g - np.array(ref[pair]), ls=ls, color=C[ci], lw=1.2)
        s = counting_sigma(r, np.array(ref[pair]), Nbig, ref["box"], ref["dr"], pair) * math.sqrt(2)
        axes[1, j].fill_between(r, -2 * s, 2 * s, color="#c9c9c9", alpha=0.6, lw=0, label="±2σ counting noise")
        axes[0, j].set_title(pair.replace("g_", "g$_{") + "}$(r)"); axes[0, j].grid(True)
        axes[1, j].grid(True); axes[1, j].set_xlabel("r (nm)")
        axes[1, j].set_xlim(0.15, 1.2)
    axes[0, 0].set_ylabel("g(r)"); axes[1, 0].set_ylabel("Δg vs GROMACS")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    pd_style.title(fig, f"Partial RDFs of the final frame, {Nbig:,} waters",
                   "Top: g(r); bottom: difference to GROMACS with the ±2σ counting noise of one frame")
    fig.savefig(os.path.join(OUT, "water_rdf.png"), bbox_inches="tight"); plt.close(fig)

# density: dot plot per size with the one-frame spread (and run averages where kept)
fig, axes = plt.subplots(1, len(SIZES), figsize=(16, 5.5), sharey=False)
for k, N in enumerate(SIZES):
    ax = axes[k]; ys = []
    for i, (e, (label, ci, mi, ls, filled, dev)) in enumerate(ENGINES.items()):
        if (N, e) not in res or res[(N, e)][0]["density_final"] is None:
            continue
        rec = res[(N, e)][0]
        ax.errorbar(rec["density_final"], i, xerr=rec["density_frame_sigma"], fmt=MK[mi], color=C[ci], ms=9,
                    mfc=C[ci] if filled else "white", mew=2, capsize=4, lw=1.6)
        if rec["density_average"]:
            ax.plot(rec["density_average"], i, "|", color=C[ci], ms=18, mew=2.5)
        ys.append((i, label))
    ax.set_yticks([i for i, _ in ys]); ax.set_yticklabels([l for _, l in ys] if k == 0 else [])
    ax.set_title(f"{N:,} waters"); ax.grid(True, axis="x"); ax.set_xlabel("Density (kg/m³)")
    ax.invert_yaxis()
pd_style.title(fig, "Water density after 10,000 NPT steps",
               "Final frame ± one-frame spread √(kTκ/V); bar marks: run average (GROMACS, LAMMPS)")
fig.savefig(os.path.join(OUT, "water_density.png"), bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
