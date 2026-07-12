"""ParticleDynamics.jl house plot style.

Matches the published benchmark figures: bold sans-serif navy titles, gray
subtitles, the fixed five-engine categorical palette, light dashed grid,
bottom legends, and navy callout boxes. Import and call `apply()` before
plotting; use `COLORS` in fixed order (never cycled).

Categorical identity is never color-alone: every series also carries a
distinct marker (o, s, D, ^, v) per the fixed order below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (never reassign on filtering)
NAVY = "#1e3a8a"      # ParticleDynamics.jl / primary series
RED = "#cc3b44"       # OpenMM / experiment or reference points
PURPLE = "#7d3cb5"    # GROMACS / tertiary
TEAL = "#2a8f85"      # LAMMPS / quaternary
BROWN = "#8b6b4f"     # kUPS / quinary
COLORS = [NAVY, RED, PURPLE, TEAL, BROWN]
MARKERS = ["o", "s", "D", "^", "v"]

GRAY_SUB = "#555555"  # subtitles / secondary text
GRID = "#c9c9c9"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlecolor": NAVY,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.4,
        "lines.linewidth": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })


def title(fig, text, subtitle=None, y=1.02):
    fig.suptitle(text, fontsize=17, fontweight="bold", color=NAVY, y=y)
    if subtitle:
        fig.text(0.5, y - 0.065, subtitle, ha="center", fontsize=11, color=GRAY_SUB)


def callout(ax, text, xy, xytext, fontsize=14):
    """Navy-bordered rounded callout with a dashed arrow, benchmark-figure style."""
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=fontsize, fontweight="bold", color=NAVY, ha="center",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=NAVY, lw=1.8),
                arrowprops=dict(arrowstyle="->", linestyle="--", color=NAVY, lw=1.6))
