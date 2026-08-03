#!/usr/bin/env python3
"""Side-by-side movie of the benzene-crystal heating ramp under two foundation
models, scored against experimental data.

Left panel: MACE-OFF23-small. Right panel: Orb-v3-conservative-inf-omat. Both
trajectories come from the identical engine, protocol, timestep and precision,
so everything that differs on screen is the model.

The bottom-left panel is the point of the figure: every experimentally
measurable quantity for benzene crystal I — lattice energy, equilibrium cell
volume, both bond lengths, and three Raman frequencies — with each model's value
beside the measured one. Neither model wins across the board, which is the
result worth showing. The bottom-right panel tracks the mean C–C and C–H bond
lengths live against their experimental values as the crystal is heated.

An earlier version of this figure plotted the thermostat trace and a
mean-squared-displacement order parameter. Both were dropped: at fixed cell the
two models disorder at indistinguishable temperatures (301/300 K and
469/467 K), so those panels carried no comparison, only decoration.

Run benzene_melt_movie_gsd.jl for each model first, then:

    micromamba run -p ~/.venvs/pd-fresnel python examples/orb/py_benzene_movie.py --preview
    micromamba run -p ~/.venvs/pd-fresnel python examples/orb/py_benzene_movie.py

Bonds are derived per frame from interatomic distances rather than read from the
trajectory: external potentials require bond-free states, so the GSD carries no
bond topology, and recomputing per frame keeps the rendering honest once the
crystal starts to disorder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import gsd.hoomd
import fresnel
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
VAL = HERE / "validation"
DATA = VAL  # overridden by --data-dir
sys.path.insert(0, str(HERE.parent / "mace"))
import pd_style as S  # noqa: E402

S.apply()

# The two rendered panels are the DOMAIN-MATCHED pair: both models are trained
# on molecules, so the comparison is fair. The materials-trained `omat` variant
# appears in the scorecard as a labelled off-domain control (it is not rendered;
# comparing an organic specialist against a materials model would be a strawman).
# Colours match the other figures in this directory.
MODELS = [
    ("mace-off", "MACE-OFF23-small", S.NAVY),
    ("orb-omol", "Orb-v3-cons-omol", S.PURPLE),
]
CONTROL_COLOR = S.RED

C_COLOR = np.array([0.16, 0.22, 0.42])
H_COLOR = np.array([0.86, 0.88, 0.93])


# Experimental reference values for benzene crystal I. See REFERENCES.md for
# sources and uncertainties; these are what the model columns are judged against.
CC_EXP = 1.379          # Å, X-ray at 150 K
CH_EXP = 1.08           # Å, neutron (X-ray gives 0.93 Å from foreshortening)

# Scorecard: (label, experiment, MACE-OFF, Orb-omol, Orb-omat, index of closest).
# Energies and volumes from benzene_lattice_energy.jl; frequencies from the 150 K
# NVE spectra in benzene_md_vdos.jl. See REFERENCES.md for sources.
#
# The volume row carries its ensemble mismatch in the label on purpose: a 0 K
# relaxed volume is not a finite-temperature, constant-pressure measurement, and
# without a barostat that is as close as this engine can get.
SCORE_ROWS = [
    ("lattice energy  (kJ/mol)", "−55.3 ± 2.2", "−46.2", "−59.6", "−20.8", 1),
    ("cell volume, 0 K  (Å³)", "462–494", "466", "437", "583", 0),
    ("C–C bond  (Å)", "1.379", "1.389", "1.390", "1.396", 0),
    ("C–H bond  (Å)", "1.08", "1.081", "1.084", "1.093", 0),
    ("ring breathing  (cm⁻¹)", "992", "1021", "1021", "1007", 2),
    ("C–C stretch  (cm⁻¹)", "1586", "1508", "1508", "1594", 2),
    ("C–H stretch  (cm⁻¹)", "3062", "3209", "3202", "3122", 2),
]


def _pair_indices(pos, idx_a, idx_b, lengths, cutoff, same=False):
    """Vectorised minimum-image neighbour pairs between two index sets."""
    d = pos[idx_a][:, None, :] - pos[idx_b][None, :, :]
    d -= lengths * np.round(d / lengths)
    r = np.linalg.norm(d, axis=2)
    if same:
        mask = np.triu(r < cutoff, k=1)
    else:
        mask = r < cutoff
    ia, ib = np.nonzero(mask)
    return idx_a[ia], idx_b[ib], r[ia, ib]


def bond_lists(pos, numbers, lengths):
    """C–C and C–H bonds as (index pairs, lengths). Vectorised per frame.

    Cutoffs are generous on purpose: by ~800 K a C–H bond (1.09 Å at rest)
    vibrates well past 1.3 Å, and a tighter cutoff makes hydrogens flicker in
    and out of their bonds and render as floating specks.
    """
    idx_c = np.flatnonzero(numbers == 6)
    idx_h = np.flatnonzero(numbers != 6)
    ca, cb, rcc = _pair_indices(pos, idx_c, idx_c, lengths, 1.85, same=True)
    ha, hb, rch = _pair_indices(pos, idx_c, idx_h, lengths, 1.45)
    pairs = np.concatenate([np.stack([ca, cb], axis=1),
                            np.stack([ha, hb], axis=1)], axis=0)
    return pairs, rcc, rch


def _smooth(y, w=9):
    """Centred moving average, with the ends left unsmoothed."""
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    out = y.copy()
    out[w // 2:len(y) - w // 2] = np.convolve(y, k, mode="valid")
    return out


def bond_points(pos, pairs, lengths):
    """Minimum-image cylinder endpoints for the given bond pairs."""
    if len(pairs) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    first = pos[pairs[:, 0]]
    delta = pos[pairs[:, 1]] - first
    delta -= lengths * np.round(delta / lengths)
    mid = first + 0.5 * delta
    return np.stack((mid - 0.5 * delta, mid + 0.5 * delta),
                    axis=1).astype(np.float32)


class Panel:
    """One fresnel scene for one model's trajectory."""

    def __init__(self, gsd_path, numbers, box_color, size, tracer, threads,
                 samples, light_samples):
        self.traj = gsd.hoomd.open(str(gsd_path), mode="r")
        self.numbers = numbers
        self.size = size
        self.tracer = tracer
        self.samples = samples
        self.light_samples = light_samples

        frame0 = self.traj[0]
        box = np.asarray(frame0.configuration.box, dtype=np.float32)
        self.lengths = box[:3].copy()

        self.device = fresnel.Device(mode="cpu", n=threads)
        self.scene = fresnel.Scene(device=self.device)
        self.scene.background_color = fresnel.color.linear([0.965, 0.975, 0.990])
        self.scene.background_alpha = 1.0
        self.scene.lights = fresnel.light.lightbox()

        n = int(frame0.particles.N)
        radii = np.where(numbers == 6, 0.34, 0.21).astype(np.float32)
        colors = np.where(numbers[:, None] == 6, C_COLOR, H_COLOR)
        self.atoms = fresnel.geometry.Sphere(self.scene, N=n)
        self.atoms.radius[:] = radii
        self.atoms.color[:] = fresnel.color.linear(colors)
        self.atoms.material = fresnel.material.Material(
            roughness=0.70, specular=0.36, primitive_color_mix=1.0)
        self.atoms.outline_width = 0.020

        # Cylinder geometry is resized per frame, so allocate generously once.
        self.max_bonds = 4 * n
        self.bonds = fresnel.geometry.Cylinder(self.scene, N=self.max_bonds)
        self.bonds.material = fresnel.material.Material(
            roughness=0.62, specular=0.28, primitive_color_mix=1.0)
        self.bonds.outline_width = 0.012
        self.bonds.radius[:] = 0.085
        c_lin = np.asarray(fresnel.color.linear(C_COLOR))
        self.bonds.color[:] = np.tile(c_lin, (self.max_bonds, 2, 1))

        self.frame_box = fresnel.geometry.Box(self.scene, box, box_radius=0.035)
        self.frame_box.box_color = fresnel.color.linear(
            matplotlib.colors.to_rgb(box_color))

        longest = float(np.max(self.lengths))
        self.scene.camera = fresnel.camera.Orthographic(
            position=(2.25 * longest, 1.95 * longest, 1.75 * longest),
            look_at=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
            height=1.32 * longest)

    def __len__(self):
        return len(self.traj)

    def bond_history(self):
        """Mean C–C and C–H bond length for every frame (experiment-comparable)."""
        cc = np.empty(len(self.traj))
        ch = np.empty(len(self.traj))
        for k in range(len(self.traj)):
            pos = np.asarray(self.traj[k].particles.position, dtype=np.float32)
            _, rcc, rch = bond_lists(pos, self.numbers, self.lengths)
            cc[k] = rcc.mean() if len(rcc) else np.nan
            ch[k] = rch.mean() if len(rch) else np.nan
        return cc, ch

    def render(self, k):
        frame = self.traj[min(k, len(self.traj) - 1)]
        pos = np.asarray(frame.particles.position, dtype=np.float32)
        self.atoms.position[:] = pos
        pairs, _, _ = bond_lists(pos, self.numbers, self.lengths)
        pts = bond_points(pos, pairs, self.lengths)
        nb = min(len(pts), self.max_bonds)
        # Park unused cylinders at a degenerate point so they do not render.
        parked = np.zeros((self.max_bonds, 2, 3), dtype=np.float32)
        parked[:nb] = pts[:nb]
        self.bonds.points[:] = parked
        r = np.full(self.max_bonds, 0.085, dtype=np.float32)
        r[nb:] = 0.0
        self.bonds.radius[:] = r

        w, h = self.size
        if self.tracer == "preview":
            img = fresnel.preview(self.scene, w=w, h=h, anti_alias=True)
        else:
            img = fresnel.pathtrace(self.scene, w=w, h=h,
                                    samples=self.samples,
                                    light_samples=self.light_samples)
        return np.asarray(img[:], dtype=np.uint8)


def load_diag(key):
    p = DATA / f"benzene_melt_{key}_diag.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}; run benzene_melt_movie_gsd.jl {key} first")
    return np.load(p, allow_pickle=True)


def load_scoreboard():
    """E_latt per model from the lattice-energy summary, for the accuracy badge."""
    out = {}
    p = DATA / "benzene_lattice_energy.txt"
    if p.exists():
        for line in open(p):
            f = line.split()
            if "E_latt" in f:
                out[f[0]] = float(f[f.index("E_latt") + 2])
    return out


def build_frame(fig, gs, panels, diags, bonds, k, nframes):
    fig.clf()
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_m = fig.add_subplot(gs[1, 1])

    for ax, (key, label, col), panel in zip((ax_l, ax_r), MODELS, panels):
        ax.imshow(panel.render(k), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.0)
        d = diags[key]
        rate = float(np.mean(d["rates"][max(0, k - 5):k + 1])) if k > 0 else float(d["rates"][0])
        ax.set_title(label, color=col, fontsize=19, fontweight="bold", pad=6)
        ax.text(0.015, 0.985, f"{rate:.1f} MD steps/s",
                transform=ax.transAxes, va="top", ha="left", fontsize=14,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.6))

    d0 = diags[MODELS[0][0]]

    # --- scorecard: every experimentally measurable quantity, side by side ---
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(0.0, 0.975, "Measured against experiment", fontsize=16,
              fontweight="bold", color=S.NAVY)
    xc = (0.400, 0.573, 0.745, 0.925)
    heads = (("experiment", "black"), ("MACE-OFF", MODELS[0][2]),
             ("Orb-omol", MODELS[1][2]), ("Orb-omat*", CONTROL_COLOR))
    for x, (txt, col) in zip(xc, heads):
        ax_t.text(x, 0.895, txt, fontsize=12, fontweight="bold", color=col,
                  ha="center")
    ax_t.plot([0.0, 1.0], [0.862, 0.862], color="black", lw=1.4)
    cols = ("black", MODELS[0][2], MODELS[1][2], CONTROL_COLOR)
    for r, row in enumerate(SCORE_ROWS):
        lab, vals, best = row[0], row[1:5], row[5]
        y = 0.785 - r * 0.109
        ax_t.text(0.0, y, lab, fontsize=12, va="center", color="black")
        for c, (val, col) in enumerate(zip(vals, cols)):
            win = (best == c - 1)   # c = 0 is the experiment column
            ax_t.text(xc[c], y, val, fontsize=13 if win else 12,
                      va="center", ha="center", color=col,
                      fontweight="bold" if win else "normal")
    ax_t.text(0.0, -0.045,
              "bold = closest to experiment   ·   *materials-trained "
              "off-domain control (not rendered above)",
              fontsize=10, color=S.GRAY_SUB, style="italic")

    # --- live bond lengths, plotted as deviation from the measured values so
    # that zero *is* experiment and both bonds share one scale ---
    for (key, label, col), (cc, ch) in zip(MODELS, bonds):
        T = diags[key]["targets"]
        n = min(k + 1, len(cc), len(T))
        # 9-frame moving average: the per-frame mean over 192 bonds is noisy
        # enough to hide the offsets between models, which are the point.
        ax_m.plot(T[:n], _smooth(cc[:n] - CC_EXP), color=col, lw=2.4,
                  label=f"{label}  C–C")
        ax_m.plot(T[:n], _smooth(ch[:n] - CH_EXP), color=col, lw=2.4, ls="--",
                  label=f"{label}  C–H")
    ax_m.axhline(0.0, color="black", lw=2.2)
    ax_m.text(0.985, 0.0012, "experiment", transform=ax_m.get_yaxis_transform(),
              ha="right", va="bottom", fontsize=12, color="black",
              fontweight="bold")
    ax_m.set_xlim(float(d0["targets"][0]), float(d0["targets"][-1]))
    ax_m.set_ylim(-0.004, 0.024)
    ax_m.set_xlabel("temperature  (K)")
    # Plain Ångström, not milli-Ångström: the zero line already says
    # "experiment", so the axis only has to name the quantity and its unit.
    ax_m.set_ylabel("bond length error  (Å)", fontsize=13)
    ax_m.grid(True)
    ax_m.legend(loc="upper left", fontsize=10.5, ncol=2)

    tnow = float(d0["times"][min(k, len(d0["times"]) - 1)])
    Tnow = float(d0["targets"][min(k, len(d0["targets"]) - 1)])
    fig.suptitle("Two foundation models, one MD engine, judged against experiment",
                 fontsize=23, fontweight="bold", color=S.NAVY, y=0.982)
    fig.text(0.5, 0.928,
             f"benzene crystal I · same integrator, timestep and precision "
             f"(float32) · t = {tnow:5.2f} ps,  T = {Tnow:5.0f} K",
             ha="center", fontsize=13, color=S.GRAY_SUB)
    fig.text(0.5, 0.012,
             "Heating at fixed cell (an external potential supplies no virial, so "
             "no barostat): the crystal is deliberately driven past its stability "
             "limit as a stress test — this is not a melting-point prediction.",
             ha="center", fontsize=11, color=S.GRAY_SUB, style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=VAL / "benzene_head_to_head.mp4")
    ap.add_argument("--preview", action="store_true",
                    help="render a single frame to PNG and exit")
    ap.add_argument("--preview-frame", type=int, default=-1)
    ap.add_argument("--preview-output", type=Path,
                    default=VAL / "benzene_movie_preview.png")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--tracer", choices=("preview", "path"), default="preview")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--light-samples", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--panel", type=int, nargs=2, default=(1000, 630))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="read trajectories/diagnostics from here (testing)")
    args = ap.parse_args()

    global DATA
    if args.data_dir is not None:
        DATA = args.data_dir

    diags = {key: load_diag(key) for key, _, _ in MODELS}
    numbers = np.asarray(diags[MODELS[0][0]]["numbers"], dtype=np.int64)

    panels = []
    for (key, label, col) in MODELS:
        gsd_path = DATA / f"benzene_melt_{key}.gsd"
        if not gsd_path.exists():
            raise SystemExit(f"missing {gsd_path}; run benzene_melt_movie_gsd.jl {key}")
        panels.append(Panel(gsd_path, numbers, col, tuple(args.panel),
                            args.tracer, args.threads, args.samples,
                            args.light_samples))

    print("precomputing bond-length histories ...", flush=True)
    bonds = [p.bond_history() for p in panels]

    nframes = min(len(p) for p in panels)
    nframes = min(nframes, *(len(diags[k]["times"]) for k, _, _ in MODELS))
    print(f"{nframes} frames available; tracer={args.tracer}")

    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.62, 1.0],
                  left=0.062, right=0.985, top=0.875, bottom=0.085,
                  wspace=0.11, hspace=0.34)

    if args.preview:
        k = args.preview_frame % nframes
        t0 = time.time()
        build_frame(fig, gs, panels, diags, bonds, k, nframes)
        fig.savefig(args.preview_output)
        print(f"wrote {args.preview_output} (frame {k}, {time.time() - t0:.1f} s)")
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to encode the movie")
    tmp = DATA / "_movie_frames"
    tmp.mkdir(exist_ok=True)
    for old in tmp.glob("frame_*.png"):
        old.unlink()

    t0 = time.time()
    written = 0
    for k in range(0, nframes, args.stride):
        build_frame(fig, gs, panels, diags, bonds, k, nframes)
        fig.savefig(tmp / f"frame_{written:05d}.png")
        written += 1
        if written % 20 == 0:
            el = time.time() - t0
            print(f"  {written} frames  ({el / written:.2f} s/frame)", flush=True)
    plt.close(fig)

    cmd = [ffmpeg, "-y", "-framerate", str(args.fps),
           "-i", str(tmp / "frame_%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"wrote {args.output} ({written} frames, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
