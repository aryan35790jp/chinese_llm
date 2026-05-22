"""
Publication style for matplotlib. Imported by every figure-producing script.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import FIGURES_DIR


def setup_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str, also_pdf: bool = True) -> None:
    """Save fig to figures/ as PNG (300 DPI) and optionally PDF."""
    out = FIGURES_DIR / name
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
