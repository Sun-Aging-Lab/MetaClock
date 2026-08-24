"""Shared biological-analysis typography, labels, and vector-output helpers."""

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

FONT_DIR = Path(__file__).resolve().parents[1] / "generalization" / "assets" / "fonts"
FONT_FILES = (
    "LiberationSans-Regular.ttf",
    "LiberationSans-Italic.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-BoldItalic.ttf",
)

LABELS = {
    "bio_metaclock": "BioMeta",
    "baseline": "Baseline",
    "dnamphenoage": "DNAm\nPheno",
    "clinical_phenoage": "PhenoAge",
}
COLORS = {
    "bio_metaclock": "#0072B2",
    "baseline": "#4D4D4D",
    "dnamphenoage": "#009E73",
    "clinical_phenoage": "#CC79A7",
}


def configure_style() -> None:
    """Require Liberation Sans and apply the formal manuscript typography."""
    paths = [FONT_DIR / name for name in FONT_FILES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manuscript fonts: {missing}")
    allowed = {path.resolve() for path in paths}
    font_manager.fontManager.ttflist = [
        entry
        for entry in font_manager.fontManager.ttflist
        if entry.name != "Liberation Sans" or Path(entry.fname).resolve() in allowed
    ]
    for path in paths:
        font_manager.fontManager.addfont(path)
    resolved_font = Path(
        font_manager.findfont("Liberation Sans", fallback_to_default=False)
    ).resolve()
    if resolved_font not in allowed:
        raise RuntimeError("Liberation Sans did not resolve to the audited project font.")
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.75,
            "axes.unicode_minus": False,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "ytick.major.width": 0.2,
            "xtick.minor.width": 0.2,
            "ytick.minor.width": 0.2,
            "patch.linewidth": 0.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.fallback": None,
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
