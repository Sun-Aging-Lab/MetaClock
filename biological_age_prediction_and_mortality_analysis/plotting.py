"""Shared Figure 4 typography, labels, and vector-output helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metaclock-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

LABELS = {
    "bio_metaclock": "BioMeta",
    "baseline": "Baseline",
    "dnamphenoage": "DNAm\nPheno",
    "clinical_phenoage": "PhenoAge",
}
COLORS = {
    "bio_metaclock": "#0072B2",
    "baseline": "#4D4D4D",
    "clinical_phenoage": "#CC79A7",
}


def configure_style() -> None:
    """Require Liberation Sans and apply the formal manuscript typography."""
    font_manager.findfont("Liberation Sans", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "ytick.major.width": 0.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_vector_pair(figure: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write matching PDF/SVG vectors and close the figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf, svg = output_dir / f"{stem}.pdf", output_dir / f"{stem}.svg"
    figure.savefig(pdf)
    figure.savefig(svg)
    plt.close(figure)
    return pdf, svg


def clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    """Apply the shared open-axis style."""
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.05)


def ordered_rows(table: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    """Require exactly one row per requested model and return manifest order."""
    if table["model"].duplicated().any() or set(table["model"]) != set(models):
        raise ValueError("Plot source must contain exactly one row per requested model.")
    return table.set_index("model").loc[list(models)].reset_index()
