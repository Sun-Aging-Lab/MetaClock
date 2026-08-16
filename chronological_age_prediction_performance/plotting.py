"""Shared, publication-style scatter plotting utilities for age prediction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

DATASETS: tuple[str, ...] = (
    "GSE147221",
    "GSE157131",
    "GSE234461",
    "GSE72680",
)
ARCHITECTURES: tuple[str, ...] = (
    "ElasticNet",
    "Lasso",
    "Ridge",
    "RF",
    "SVR",
    "LightGBM",
    "MLP",
    "Mean",
    "SpectralEnsemble",
)
ARCHITECTURE_LABELS = {
    "ElasticNet": "EN",
    "LightGBM": "LGBM",
    "SVR": "SVM",
    "SpectralEnsemble": "SE",
}
COMPONENT_LABELS = {
    "altumage": "AltumAge",
    "dnamphenoage": "DNAmPhenoAge",
    "encen40": "EN-Cen40",
    "han": "Han",
    "hannum": "Hannum",
    "horvath2013": "Horvath",
    "hrsinchphenoage": "HRSInCHPhenoAge",
    "lin": "Lin",
    "pchannum": "PC Hannum",
    "pchorvath2013": "PC Horvath",
    "pcphenoage": "PC PhenoAge",
    "pcskinandblood": "PC Skin&Blood",
    "skinandblood": "Skin&Blood",
    "zhangblup": "Zhang BLUP",
    "zhangen": "Zhang EN",
    "MetaClock": "MetaClock",
}
META_LABEL = "MetaClock"
FIGURE_SIZE_INCHES = (180 / 25.4, 65 / 25.4)
TAB20 = tuple(plt.get_cmap("tab20").colors)
META_COLOR = TAB20[8]

# These maps retain the colors of the formal scatter plots.  Component colors
# deliberately skip tab20[8], which is reserved for MetaClock/SVM.
ARCHITECTURE_COLORS = {
    label: TAB20[index] for label, index in zip(ARCHITECTURES, range(0, 18, 2), strict=True)
}
_COMPONENT_COLOR_INDICES = tuple(index for index in range(len(TAB20)) if index != 8)[:15]
COMPONENT_COLORS = {
    label: TAB20[index]
    for label, index in zip(
        (label for label in COMPONENT_LABELS if label != META_LABEL),
        _COMPONENT_COLOR_INDICES,
        strict=True,
    )
}
COMPONENT_COLORS[META_LABEL] = META_COLOR


def configure_manuscript_style() -> None:
    """Configure deterministic vector typography and line styling."""
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_dataset_axes() -> tuple[plt.Figure, np.ndarray]:
    """Create four aligned dataset axes with a fixed legend margin."""
    figure, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE_INCHES)
    figure.subplots_adjust(left=0.07, right=0.72, bottom=0.25, top=0.80, wspace=0.60)
    return figure, np.asarray(axes)


def style_dataset_axis(axis: plt.Axes, dataset: str, index: int) -> None:
    """Apply common labels, tick formatting, and physical panel margins."""
    axis.set_title(dataset, pad=5)
    axis.set_ylabel("Pearson r" if index == 0 else "")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.margins(x=0.10, y=0.14)


def add_shared_mae_label(figure: plt.Figure) -> None:
    """Add the shared MAE x-axis label centered below the four data axes."""
    figure.text(0.395, 0.08, "MAE (years)", ha="center", va="center")


def add_legend(
    figure: plt.Figure,
    handles: Iterable[Line2D],
    *,
    title: str | None = None,
) -> None:
    """Place a compact legend in the reserved right-side margin."""
    figure.legend(
        handles=list(handles),
        title=title,
        loc="center left",
        bbox_to_anchor=(0.74, 0.50),
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.2,
    )


def plot_labeled_metrics(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    *,
    label_column: str,
    labels: Sequence[str],
    colors: Mapping[str, object],
    meta_label: str | None = META_LABEL,
    component_alpha: float = 0.90,
) -> None:
    """Draw one complete label set, highlighting the optional MetaClock label."""
    indexed = metrics.set_index(label_column)
    for label in labels:
        is_meta = label == meta_label
        axis.scatter(
            indexed.at[label, "mae"],
            indexed.at[label, "pearson_r"],
            s=86 if is_meta else 22,
            marker="*" if is_meta else "o",
            color=colors[label],
            edgecolor="#202020",
            linewidth=0.45,
            alpha=1.0 if is_meta else component_alpha,
            zorder=4 if is_meta else 3,
        )


def metric_legend_handles(
    labels: Sequence[str],
    *,
    colors: Mapping[str, object],
    display_labels: Mapping[str, str],
    meta_label: str | None = META_LABEL,
    meta_display_label: str | None = None,
    component_alpha: float = 0.90,
) -> list[Line2D]:
    """Return legend handles that exactly match :func:`plot_labeled_metrics`."""
    handles: list[Line2D] = []
    for label in labels:
        is_meta = label == meta_label
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*" if is_meta else "o",
                linestyle="None",
                markersize=7 if is_meta else 4,
                markerfacecolor=colors[label],
                markeredgecolor="#202020",
                markeredgewidth=0.45,
                alpha=1.0 if is_meta else component_alpha,
                label=meta_display_label
                if is_meta and meta_display_label
                else display_labels[label],
            )
        )
    return handles


def save_vector_pair(figure: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Write matching PDF/SVG outputs and return their paths after closing."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    figure.savefig(pdf_path)
    figure.savefig(svg_path)
    plt.close(figure)
    return pdf_path, svg_path
