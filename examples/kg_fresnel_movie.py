#!/usr/bin/env python3
"""Render the Kremer--Grest melt trajectory as a Fresnel dashboard movie.

The left panel path traces the GSD trajectory with rough, blue-toned beads and
bonds. The right panels reveal two diagnostics live with the trajectory:

* ``Ree^2`` is compared live against Table I of Kremer and Grest (1990).
* ``Rg^2`` is compared live against the same reported melt reference.

Run ``3D_KG_melt_showcase.jl`` first, then create the conda environment in
``kg_fresnel_environment.yml``. Use ``--preview`` to render one PNG before
committing to the full movie.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fresnel
import gsd.hoomd
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import (  # noqa: E402
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.patches import FancyBboxPatch  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "kg_out"
sys.path.insert(0, str(HERE / "mace"))
import pd_style  # noqa: E402

pd_style.apply()

BG = "#f7f9fc"
PANEL = "#ffffff"
PLOT_BG = "#ffffff"
TEXT = "#172554"
MUTED = "#555555"
CYAN = "#43c6e8"
NAVY = pd_style.NAVY
RED = pd_style.RED
TEAL = pd_style.TEAL
GRID = pd_style.GRID

# Six related colors preserve the trajectory's chain typing while staying
# close to the rough-blue Fresnel introduction image.
CHAIN_COLORS = np.asarray(
    [
        [0.12, 0.31, 0.72],
        [0.16, 0.48, 0.91],
        [0.10, 0.62, 0.79],
        [0.31, 0.32, 0.82],
        [0.09, 0.41, 0.64],
        [0.38, 0.56, 0.94],
    ],
    dtype=np.float32,
)

# Kremer & Grest, J. Chem. Phys. 92, 5057 (1990), Table I: the N=25 melt
# reports <Ree²>=37.8 and <Rg²>=6.3. Section II C establishes ideal-chain
# scaling, so the reference for another chain length scales with N - 1.
PAPER_CHAIN_LENGTH = 25
PAPER_REE2 = 37.8
PAPER_RG2 = 6.3


@dataclass(frozen=True)
class Observables:
    time: np.ndarray
    ree2: np.ndarray
    rg2: np.ndarray
    g1: np.ndarray
    g3: np.ndarray

    @property
    def ratio(self) -> np.ndarray:
        return self.ree2 / self.rg2


@dataclass(frozen=True)
class Summary:
    particles: int
    bonds: int
    chains: int
    beads_per_chain: int
    mean_bond: float
    c_infinity: float
    mean_ratio: float


@dataclass(frozen=True)
class PaperReference:
    ree2: float
    rg2: float


def paper_reference(beads_per_chain: int) -> PaperReference:
    """Scale the Table I N=25 chain-size values to this chain length."""
    if beads_per_chain < 2:
        raise ValueError("at least two beads per chain are required")
    scale = (beads_per_chain - 1) / (PAPER_CHAIN_LENGTH - 1)
    return PaperReference(ree2=PAPER_REE2 * scale, rg2=PAPER_RG2 * scale)


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(x) for x in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as err:
        raise argparse.ArgumentTypeError("resolution must look like 1920x1080") from err
    if width < 640 or height < 360 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError(
            "resolution must be at least 640x360 and have even dimensions"
        )
    return width, height


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Path trace the KG melt with live validation plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_DATA / "kg_melt.gsd")
    parser.add_argument(
        "--observables", type=Path, default=DEFAULT_DATA / "kg_observables.csv"
    )
    parser.add_argument("--bonds", type=Path, default=DEFAULT_DATA / "kg_bonds.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA / "kg_showcase.mp4")
    parser.add_argument(
        "--preview-output", type=Path, default=DEFAULT_DATA / "kg_showcase_preview.png"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="render one dashboard frame to PNG and exit",
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=-1,
        help="trajectory frame for --preview (negative indices count from the end)",
    )
    parser.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--duration", type=positive_float, default=12.0)
    parser.add_argument(
        "--final-hold",
        type=positive_float,
        default=2.0,
        help="seconds to hold the final validation summary",
    )
    parser.add_argument(
        "--stride", type=positive_int, default=1, help="consider every Nth GSD frame"
    )
    parser.add_argument("--dt", type=positive_float, default=0.005, help="LJ time per MD step")
    parser.add_argument(
        "--tracer",
        choices=("path", "preview"),
        default="path",
        help="Fresnel path tracing, or fast direct-lighting previews",
    )
    parser.add_argument("--samples", type=positive_int, default=8)
    parser.add_argument("--light-samples", type=positive_int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Fresnel render device (conda-forge binaries provide CPU/Embree)",
    )
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=None,
        help="CPU threads or GPUs passed to fresnel.Device",
    )
    parser.add_argument("--sphere-radius", type=positive_float, default=0.43)
    parser.add_argument("--bond-radius", type=positive_float, default=0.13)
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality; lower is better")
    args = parser.parse_args()
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.final_hold >= args.duration:
        parser.error("--final-hold must be shorter than --duration")
    return args


def load_observables(path: Path) -> Observables:
    required = ("t_LJ", "Ree2", "Rg2", "g1_bead_MSD", "g3_com_MSD")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.size == 0:
        raise ValueError(f"{path} contains no observable samples")
    missing = set(required).difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    columns = [np.atleast_1d(table[name]) for name in required]
    if any(not np.all(np.isfinite(column)) for column in columns):
        raise ValueError(f"{path} contains non-finite observable values")
    if np.any(columns[0] <= 0) or np.any(np.diff(columns[0]) <= 0):
        raise ValueError(f"{path} time values must be positive and strictly increasing")
    if any(np.any(column <= 0) for column in columns[1:]):
        raise ValueError(f"{path} observables must be positive")
    return Observables(*columns)


def load_mean_bond(path: Path) -> float:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = table.dtype.names or ()
    if "bond_length" not in names:
        raise ValueError(f"{path} is missing the bond_length column")
    lengths = np.atleast_1d(table["bond_length"])
    if lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError(f"{path} contains invalid bond lengths")
    return float(np.mean(lengths))


def validate_snapshot(snapshot) -> tuple[np.ndarray, np.ndarray]:
    if snapshot.particles.N <= 0:
        raise ValueError("the trajectory contains no particles")
    if snapshot.bonds.N <= 0:
        raise ValueError("the trajectory contains no bonds")
    box = np.asarray(snapshot.configuration.box, dtype=np.float32)
    if box.shape != (6,) or np.any(box[:3] <= 0) or np.any(np.abs(box[3:]) > 1e-7):
        raise ValueError("the renderer currently requires an orthorhombic 3D GSD box")
    groups = np.asarray(snapshot.bonds.group, dtype=np.int64)
    if groups.shape != (snapshot.bonds.N, 2):
        raise ValueError("the GSD bond topology has an unexpected shape")
    if np.any(groups < 0) or np.any(groups >= snapshot.particles.N):
        raise ValueError("the GSD bond topology contains an invalid particle index")
    return box, groups


def summarize(snapshot, observables: Observables, mean_bond: float) -> Summary:
    particles = int(snapshot.particles.N)
    bonds = int(snapshot.bonds.N)
    # A forest of linear chains has one fewer bond than bead per chain.
    chains = particles - bonds
    if chains <= 0 or particles % chains:
        raise ValueError("cannot infer equal linear chains from N - number_of_bonds")
    beads_per_chain = particles // chains
    expected_bonds = chains * (beads_per_chain - 1)
    if expected_bonds != bonds:
        raise ValueError("the GSD topology is not a collection of equal linear chains")
    c_infinity = float(
        np.mean(observables.ree2) / ((beads_per_chain - 1) * mean_bond**2)
    )
    return Summary(
        particles=particles,
        bonds=bonds,
        chains=chains,
        beads_per_chain=beads_per_chain,
        mean_bond=mean_bond,
        c_infinity=c_infinity,
        mean_ratio=float(np.mean(observables.ratio)),
    )


def minimum_image_bond_points(
    positions: np.ndarray, groups: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Return short bond segments centered in the primary periodic box.

    A wrapped bond can otherwise appear as a cylinder spanning the complete
    box. Centering the minimum-image segment leaves, at worst, a bead-radius
    overhang where a bond crosses a periodic face and avoids those artifacts.
    """
    first = positions[groups[:, 0]]
    delta = positions[groups[:, 1]] - first
    delta -= lengths * np.floor(delta / lengths + 0.5)
    middle = first + 0.5 * delta
    middle -= lengths * np.floor(middle / lengths + 0.5)
    return np.stack((middle - 0.5 * delta, middle + 0.5 * delta), axis=1)


class MeltScene:
    def __init__(
        self,
        snapshot,
        render_size: tuple[int, int],
        *,
        device_mode: str,
        threads: int | None,
        sphere_radius: float,
        bond_radius: float,
        tracer: str,
        samples: int,
        light_samples: int,
    ) -> None:
        box, groups = validate_snapshot(snapshot)
        self.box = box
        self.lengths = box[:3].copy()
        self.groups = groups
        self.render_size = render_size
        self.tracer = tracer
        self.samples = samples
        self.light_samples = light_samples

        self.device = fresnel.Device(mode=device_mode, n=threads)
        self.scene = fresnel.Scene(device=self.device)
        self.scene.background_color = fresnel.color.linear([0.965, 0.975, 0.990])
        self.scene.background_alpha = 1.0
        self.scene.lights = fresnel.light.lightbox()

        n_particles = int(snapshot.particles.N)
        type_ids = np.asarray(snapshot.particles.typeid, dtype=np.int64)
        if np.any(type_ids < 0):
            raise ValueError("negative particle type id in the trajectory")
        colors = CHAIN_COLORS[type_ids % len(CHAIN_COLORS)]
        linear_colors = fresnel.color.linear(colors)

        self.beads = fresnel.geometry.Sphere(
            self.scene, N=n_particles, radius=sphere_radius
        )
        self.beads.material = fresnel.material.Material(
            roughness=0.72, specular=0.34, primitive_color_mix=1.0
        )
        self.beads.outline_width = 0.018
        self.beads.color[:] = linear_colors

        self.bonds = fresnel.geometry.Cylinder(self.scene, N=len(groups))
        self.bonds.material = fresnel.material.Material(
            roughness=0.64, specular=0.28, primitive_color_mix=1.0
        )
        self.bonds.outline_width = 0.012
        self.bonds.radius[:] = bond_radius
        self.bonds.color[:] = np.stack(
            (linear_colors[groups[:, 0]], linear_colors[groups[:, 1]]), axis=1
        )

        self.frame_box = fresnel.geometry.Box(self.scene, box, box_radius=0.035)
        self.frame_box.box_color = fresnel.color.linear([0.10, 0.27, 0.58])

        longest = float(np.max(self.lengths))
        # Fill the left panel vertically while retaining a clean safety margin
        # around the periodic box and its bead-radius overhangs.
        camera_position = (2.25 * longest, 1.95 * longest, 1.75 * longest)
        self.scene.camera = fresnel.camera.Orthographic(
            position=camera_position,
            look_at=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            height=1.65 * longest,
        )
        self.update(snapshot)

    def update(self, snapshot) -> None:
        if snapshot.particles.N != self.beads.position.shape[0]:
            raise ValueError("particle count changed between GSD frames")
        groups = np.asarray(snapshot.bonds.group, dtype=np.int64)
        if groups.shape != self.groups.shape or not np.array_equal(groups, self.groups):
            raise ValueError("bond topology changed between GSD frames")
        positions = np.asarray(snapshot.particles.position, dtype=np.float32)
        self.beads.position[:] = positions
        self.bonds.points[:] = minimum_image_bond_points(
            positions, self.groups, self.lengths
        )

    def render(self) -> np.ndarray:
        width, height = self.render_size
        if self.tracer == "preview":
            image = fresnel.preview(self.scene, w=width, h=height, anti_alias=True)
        else:
            image = fresnel.pathtrace(
                self.scene,
                w=width,
                h=height,
                samples=self.samples,
                light_samples=self.light_samples,
            )
        return np.asarray(image[:], dtype=np.uint8)


def style_axis(axis) -> None:
    """Apply the MACE house typography and strokes on a light movie panel."""
    axis.set_facecolor(PLOT_BG)
    axis.tick_params(
        colors="black", labelsize=16, direction="in", length=8, width=1.4
    )
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.4)
    axis.grid(True, color=GRID, linestyle="--", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.xaxis.label.set_color("black")
    axis.yaxis.label.set_color("black")
    axis.title.set_color(NAVY)


class LiveLegend:
    """A two-row legend whose running average is a distinct red value."""

    def __init__(self, axis, reference: str, average_label: str) -> None:
        self.value = TextArea(
            "",
            textprops={"color": RED, "fontsize": 14, "fontweight": "bold"},
        )
        reference_row = HPacker(
            children=(self._swatch(RED, "--"), self._label(reference)),
            align="center",
            pad=0,
            sep=6,
        )
        average_row = HPacker(
            children=(self._swatch(NAVY, "-"), self._label(average_label), self.value),
            align="center",
            pad=0,
            sep=4,
        )
        contents = VPacker(
            children=(reference_row, average_row), align="left", pad=0, sep=3
        )
        self.artist = AnchoredOffsetbox(
            loc="lower center",
            child=contents,
            frameon=False,
            pad=0.2,
            borderpad=0.2,
            bbox_to_anchor=(0.5, 0.015),
            bbox_transform=axis.transAxes,
        )
        axis.add_artist(self.artist)

    @staticmethod
    def _label(text: str) -> TextArea:
        return TextArea(text, textprops={"color": "black", "fontsize": 13})

    @staticmethod
    def _swatch(color: str, linestyle: str) -> DrawingArea:
        area = DrawingArea(30, 12, 0, 0)
        area.add_artist(
            Line2D((1, 29), (6, 6), color=color, linewidth=2.5, linestyle=linestyle)
        )
        return area

    def set_value(self, value: float) -> None:
        self.value.set_text(rf"$\mathbf{{{value:.2f}}}$")


class Dashboard:
    def __init__(
        self,
        resolution: tuple[int, int],
        observables: Observables,
        summary: Summary,
        reference: PaperReference,
    ) -> None:
        self.observables = observables
        self.summary = summary
        self.reference = reference
        width, height = resolution
        dpi = 100
        self.figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.1, 1.1),
            left=0.03,
            right=0.97,
            bottom=0.095,
            top=0.845,
            wspace=0.015,
            hspace=0.42,
        )
        self.render_axis = self.figure.add_subplot(grid[:, 0])
        self.ree_axis = self.figure.add_subplot(grid[0, 1])
        self.rg_axis = self.figure.add_subplot(grid[1, 1])

        self.figure.text(
            0.5,
            0.955,
            "ParticleDynamics.jl",
            color=TEXT,
            fontsize=26,
            fontweight="bold",
            ha="center",
            va="top",
        )
        self.figure.text(
            0.5,
            0.910,
            "Kremer–Grest melt  ·  structural validation against Kremer & Grest (1990)",
            color=MUTED,
            fontsize=15,
            ha="center",
            va="top",
        )
        self.render_axis.set_facecolor(PLOT_BG)
        self.render_axis.set_axis_off()
        self.render_image = self.render_axis.imshow(
            np.zeros((32, 32, 4), dtype=np.uint8), interpolation="bilinear"
        )
        trajectory_bounds = self.render_axis.get_subplotspec().get_position(self.figure)
        self.trajectory_caption = self.figure.text(
            trajectory_bounds.x0 + 0.5 * trajectory_bounds.width,
            0.047,
            "",
            ha="center",
            va="center",
            color=TEXT,
            fontsize=30,
            fontweight="bold",
        )
        self.validation_box = FancyBboxPatch(
            (0.235, 0.38),
            0.53,
            0.24,
            transform=self.render_axis.transAxes,
            boxstyle="round,pad=0.02",
            facecolor=(1.0, 1.0, 1.0, 0.94),
            edgecolor=TEAL,
            linewidth=1.7,
            visible=False,
            zorder=5,
        )
        self.render_axis.add_patch(self.validation_box)
        self.validation_title = self.render_axis.text(
            0.5,
            0.555,
            "",
            transform=self.render_axis.transAxes,
            ha="center",
            va="center",
            color=NAVY,
            fontsize=20,
            fontweight="bold",
            visible=False,
            zorder=6,
        )
        self.validation_ree = self.render_axis.text(
            0.5,
            0.485,
            "",
            transform=self.render_axis.transAxes,
            ha="center",
            va="center",
            color=NAVY,
            fontsize=17,
            fontweight="bold",
            visible=False,
            zorder=6,
        )
        self.validation_rg = self.render_axis.text(
            0.5,
            0.425,
            "",
            transform=self.render_axis.transAxes,
            ha="center",
            va="center",
            color=NAVY,
            fontsize=17,
            fontweight="bold",
            visible=False,
            zorder=6,
        )
        style_axis(self.ree_axis)
        self.ree_axis.set_title(
            r"End-to-end distance  $\langle R_{\mathrm{e}}^2 \rangle$",
            loc="center",
            fontsize=20,
            fontweight="bold",
            pad=9,
            color=NAVY,
        )
        self.ree_axis.set_xlabel(
            r"production time  $t\,[\tau_{\mathrm{LJ}}]$", fontsize=18
        )
        self.ree_axis.set_ylabel(r"$\langle R_{\mathrm{e}}^2 \rangle$", fontsize=18)
        self.ree_axis.set_xlim(0.0, observables.time[-1])
        lower = min(float(np.min(observables.ree2)), reference.ree2)
        upper = max(float(np.max(observables.ree2)), reference.ree2)
        pad = 0.12 * (upper - lower)
        self.ree_axis.set_ylim(lower - pad, upper + pad)
        self.ree_axis.axhline(
            reference.ree2,
            color=RED,
            linewidth=2.0,
            linestyle="--",
        )
        (self.ree_line,) = self.ree_axis.plot(
            [],
            [],
            color=NAVY,
            linewidth=3.5,
            marker="o",
            markevery=40,
            markersize=4.5,
            label="ParticleDynamics.jl",
        )
        (self.ree_dot,) = self.ree_axis.plot([], [], "o", color=NAVY, markersize=5)
        self.ree_legend = LiveLegend(
            self.ree_axis,
            rf"Kremer–Grest (1990): $\langle R_{{\mathrm{{e}}}}^2 \rangle_{{\mathrm{{ref}}}} = {reference.ree2:.1f}$",
            r"ParticleDynamics.jl: $\overline{R_{\mathrm{e}}^2} =$",
        )

        style_axis(self.rg_axis)
        self.rg_axis.set_title(
            r"Radius of gyration  $\langle R_{\mathrm{g}}^2 \rangle$",
            loc="center",
            fontsize=20,
            fontweight="bold",
            pad=9,
            color=NAVY,
        )
        self.rg_axis.set_xlabel(
            r"production time  $t\,[\tau_{\mathrm{LJ}}]$", fontsize=18
        )
        self.rg_axis.set_ylabel(r"$\langle R_{\mathrm{g}}^2 \rangle$", fontsize=18)
        self.rg_axis.set_xlim(0.0, observables.time[-1])
        lower = min(float(np.min(observables.rg2)), reference.rg2)
        upper = max(float(np.max(observables.rg2)), reference.rg2)
        pad = 0.12 * (upper - lower)
        self.rg_axis.set_ylim(lower - pad, upper + pad)
        self.rg_axis.axhline(
            reference.rg2,
            color=RED,
            linewidth=2.0,
            linestyle="--",
        )
        (self.rg_line,) = self.rg_axis.plot(
            [],
            [],
            color=NAVY,
            linewidth=3.5,
            marker="o",
            markevery=40,
            markersize=4.5,
            label="ParticleDynamics.jl",
        )
        (self.rg_dot,) = self.rg_axis.plot([], [], "o", color=NAVY, markersize=5)
        self.rg_legend = LiveLegend(
            self.rg_axis,
            rf"Kremer–Grest (1990): $\langle R_{{\mathrm{{g}}}}^2 \rangle_{{\mathrm{{ref}}}} = {reference.rg2:.2f}$",
            r"ParticleDynamics.jl: $\overline{R_{\mathrm{g}}^2} =$",
        )

    @property
    def render_size(self) -> tuple[int, int]:
        width, height = self.figure.canvas.get_width_height()
        return max(320, int(width * 0.466)), max(240, int(height * 0.75))

    def update(self, image: np.ndarray, step: int, dt: float) -> None:
        current_time = step * dt
        stop = int(np.searchsorted(self.observables.time, current_time, side="right"))
        stop = max(1, min(stop, len(self.observables.time)))
        sl = slice(0, stop)
        self.render_image.set_data(image)
        self.render_image.set_extent((0, image.shape[1], image.shape[0], 0))
        self.render_axis.set_xlim(0, image.shape[1])
        self.render_axis.set_ylim(image.shape[0], 0)
        self.trajectory_caption.set_text(
            rf"$t = {current_time:,.1f}\,\tau_{{\mathrm{{LJ}}}}$"
        )
        t = self.observables.time[sl]
        ree2 = self.observables.ree2[sl]
        rg2 = self.observables.rg2[sl]
        self.ree_line.set_data(t, ree2)
        self.ree_dot.set_data([t[-1]], [ree2[-1]])
        self.rg_line.set_data(t, rg2)
        self.rg_dot.set_data([t[-1]], [rg2[-1]])
        self.ree_legend.set_value(float(np.mean(ree2)))
        self.rg_legend.set_value(float(np.mean(rg2)))
        if stop == len(self.observables.time):
            ree_mean = float(np.mean(ree2))
            rg_mean = float(np.mean(rg2))
            ree_error = 100 * (ree_mean / self.reference.ree2 - 1)
            rg_error = 100 * (rg_mean / self.reference.rg2 - 1)
            self.validation_title.set_text("Kremer–Grest validation")
            self.validation_ree.set_text(
                rf"$\langle R_{{\mathrm{{e}}}}^2 \rangle = {ree_mean:.2f}"
                rf"\quad\mathrm{{vs.}}\quad {self.reference.ree2:.2f}"
                rf"\quad({ree_error:+.1f}\%)$"
            )
            self.validation_rg.set_text(
                rf"$\langle R_{{\mathrm{{g}}}}^2 \rangle = {rg_mean:.2f}"
                rf"\quad\mathrm{{vs.}}\quad {self.reference.rg2:.2f}"
                rf"\quad({rg_error:+.1f}\%)$"
            )
            self.validation_box.set_visible(True)
            self.validation_title.set_visible(True)
            self.validation_ree.set_visible(True)
            self.validation_rg.set_visible(True)
        else:
            self.validation_box.set_visible(False)
            self.validation_title.set_visible(False)
            self.validation_ree.set_visible(False)
            self.validation_rg.set_visible(False)

    def rgba_bytes(self) -> bytes:
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8).tobytes()

    def save_png(self, path: Path) -> None:
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").convert("RGB").save(path, quality=95)


def ffmpeg_command(
    executable: str,
    output: Path,
    resolution: tuple[int, int],
    fps: int,
    crf: int,
) -> list[str]:
    width, height = resolution
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def render_preview(args: argparse.Namespace, trajectory, frame_index: int) -> None:
    snapshot = trajectory[frame_index]
    observables = load_observables(args.observables)
    mean_bond = load_mean_bond(args.bonds)
    summary = summarize(trajectory[0], observables, mean_bond)
    dashboard = Dashboard(
        args.resolution, observables, summary, paper_reference(summary.beads_per_chain)
    )
    scene = MeltScene(
        snapshot,
        dashboard.render_size,
        device_mode=args.device,
        threads=args.threads,
        sphere_radius=args.sphere_radius,
        bond_radius=args.bond_radius,
        tracer=args.tracer,
        samples=args.samples,
        light_samples=args.light_samples,
    )
    started = time.perf_counter()
    image = scene.render()
    dashboard.update(image, int(snapshot.configuration.step), args.dt)
    dashboard.save_png(args.preview_output)
    elapsed = time.perf_counter() - started
    print(f"wrote {args.preview_output} ({elapsed:.1f} s, {scene.device})")


def render_movie(args: argparse.Namespace, trajectory) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode MP4 output")
    observables = load_observables(args.observables)
    mean_bond = load_mean_bond(args.bonds)
    summary = summarize(trajectory[0], observables, mean_bond)
    frame_indices = list(range(0, len(trajectory), args.stride))
    if frame_indices[-1] != len(trajectory) - 1:
        frame_indices.append(len(trajectory) - 1)
    dashboard = Dashboard(
        args.resolution, observables, summary, paper_reference(summary.beads_per_chain)
    )
    scene = MeltScene(
        trajectory[frame_indices[0]],
        dashboard.render_size,
        device_mode=args.device,
        threads=args.threads,
        sphere_radius=args.sphere_radius,
        bond_radius=args.bond_radius,
        tracer=args.tracer,
        samples=args.samples,
        light_samples=args.light_samples,
    )

    output_frames = max(1, round(args.duration * args.fps))
    hold_frames = min(round(args.final_hold * args.fps), output_frames - 1)
    motion_frames = output_frames - hold_frames
    source_slots = np.rint(
        np.linspace(0, len(frame_indices) - 1, motion_frames)
    ).astype(np.int64)
    if hold_frames:
        source_slots = np.concatenate(
            (source_slots, np.full(hold_frames, len(frame_indices) - 1, dtype=np.int64))
        )
    repeats = np.bincount(source_slots, minlength=len(frame_indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ffmpeg_command(ffmpeg, args.output, args.resolution, args.fps, args.crf)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to open the ffmpeg input pipe")

    started = time.perf_counter()
    rendered = 0
    try:
        for slot, (frame_index, repeat) in enumerate(zip(frame_indices, repeats)):
            if repeat == 0:
                continue
            snapshot = trajectory[frame_index]
            scene.update(snapshot)
            image = scene.render()
            step = int(snapshot.configuration.step)
            dashboard.update(image, step, args.dt)
            raw_frame = dashboard.rgba_bytes()
            for _ in range(int(repeat)):
                process.stdin.write(raw_frame)
            rendered += 1
            elapsed = time.perf_counter() - started
            rate = rendered / elapsed if elapsed else 0.0
            remaining = (np.count_nonzero(repeats) - rendered) / rate if rate else math.inf
            print(
                f"render {rendered:3d}/{np.count_nonzero(repeats)}  "
                f"GSD frame {frame_index:3d}  step {step:7d}  "
                f"ETA {remaining / 60:5.1f} min",
                flush=True,
            )
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    elapsed = time.perf_counter() - started
    print(
        f"wrote {args.output}: {output_frames} frames, "
        f"{args.duration:.1f} s at {args.fps} fps ({elapsed / 60:.1f} min)"
    )


def main() -> None:
    args = arguments()
    inputs = (args.trajectory, args.observables, args.bonds)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing KG output(s): " + ", ".join(missing))
    if getattr(fresnel.version, "version", "") != "0.13.8":
        print(
            f"warning: designed for Fresnel 0.13.8, found {fresnel.version.version}",
            file=sys.stderr,
        )
    with gsd.hoomd.open(name=str(args.trajectory), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"{args.trajectory} contains no frames")
        if args.preview:
            index = args.preview_frame
            if index < 0:
                index += len(trajectory)
            if not 0 <= index < len(trajectory):
                raise IndexError(
                    f"preview frame {args.preview_frame} is outside a "
                    f"{len(trajectory)}-frame trajectory"
                )
            render_preview(args, trajectory, index)
        else:
            render_movie(args, trajectory)


if __name__ == "__main__":
    main()
