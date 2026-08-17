"""Exact shared styling and geometry for chronological-age scatter plots."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metaclock-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "metaclock-xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from generalization.fonts import register_fonts

MM_PER_INCH = 25.4
FIGURE_MM = (180.0, 65.0)
FIGURE_SIZE_INCHES = tuple(value / MM_PER_INCH for value in FIGURE_MM)
REFERENCE_WIDTH_MM = 170.0
LAYOUT_SCALE = FIGURE_MM[0] / REFERENCE_WIDTH_MM
PANEL_SIZE_MM = 23.0 * LAYOUT_SCALE
PANEL_LEFTS_MM = tuple(value * LAYOUT_SCALE for value in (13.0, 44.0, 75.0, 106.0))
PANEL_BOTTOM_MM = 18.0 * LAYOUT_SCALE
DATASET_TITLE_GAP_MM = 1.5 * LAYOUT_SCALE
SHARED_XLABEL_Y_MM = 3.0 * LAYOUT_SCALE
SHARED_XLABEL_GAP_FRACTION = 0.2
LEGEND_LEFT_MM = 132.0 * LAYOUT_SCALE

FIGURE_TITLE_FONT_SIZE = 9.0
SUBPLOT_TITLE_FONT_SIZE = 9.5
AXIS_TITLE_FONT_SIZE = 9.75
TICK_FONT_SIZE = 8.0
LEGEND_FONT_SIZE = 7.5
LEGEND_TITLE_FONT_SIZE = 8.5
OUTLINE_WIDTH_PT = 0.2
Y_AXIS_TITLE_PAD_PT = 1.0

DATASETS: tuple[str, ...] = ("GSE147221", "GSE157131", "GSE234461", "GSE72680")
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
TAB20 = tuple(matplotlib.colors.to_hex(color) for color in matplotlib.colormaps["tab20"].colors)
META_COLOR = TAB20[8]
META_MARKER_SIZE = 38.0 * 1.5**2
META_LEGEND_MARKER_SIZE = 6.0 * 1.5
ARCHITECTURE_COLORS = {
    label: TAB20[index] for label, index in zip(ARCHITECTURES, range(0, 18, 2), strict=True)
}
ARCHITECTURE_COLORS["SVR"] = META_COLOR
_COMPONENTS = tuple(sorted(set(COMPONENT_LABELS).difference({META_LABEL})))
_COMPONENT_PALETTE = tuple(color for index, color in enumerate(TAB20) if index != 8)[:15]
COMPONENT_COLORS = dict(zip(_COMPONENTS, _COMPONENT_PALETTE, strict=True))
COMPONENT_COLORS[META_LABEL] = META_COLOR


def configure_manuscript_style() -> None:
    """Register the audited font files and apply the formal plot style."""
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "font.size": FIGURE_TITLE_FONT_SIZE,
            "figure.titlesize": FIGURE_TITLE_FONT_SIZE,
            "axes.titlesize": SUBPLOT_TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
            "axes.linewidth": OUTLINE_WIDTH_PT,
            "xtick.major.width": OUTLINE_WIDTH_PT,
            "ytick.major.width": OUTLINE_WIDTH_PT,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_dataset_axes() -> tuple[plt.Figure, np.ndarray]:
    """Create the four physically square axes used by the formal plots."""
    figure = plt.figure(figsize=FIGURE_SIZE_INCHES)
    axes = np.array(
        [
            figure.add_axes(
                [
                    left_mm / FIGURE_MM[0],
                    PANEL_BOTTOM_MM / FIGURE_MM[1],
                    PANEL_SIZE_MM / FIGURE_MM[0],
                    PANEL_SIZE_MM / FIGURE_MM[1],
                ]
            )
            for left_mm in PANEL_LEFTS_MM
        ]
    )
    return figure, axes


def style_dataset_axis(axis: plt.Axes, dataset: str, index: int) -> None:
    """Apply exact labels, tick locators, gaps, and plot margins."""
    del dataset
    axis.set_ylabel("Pearson r" if index == 0 else "", labelpad=Y_AXIS_TITLE_PAD_PT)
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE, pad=1.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.margins(x=0.10, y=0.14)


def add_dataset_titles(figure: plt.Figure, axes: np.ndarray) -> None:
    """Place dataset titles with the formal physical gap."""
    del figure
    for dataset, axis in zip(DATASETS, axes, strict=True):
        axis.set_title(
            dataset,
            fontsize=SUBPLOT_TITLE_FONT_SIZE,
            loc="center",
            pad=DATASET_TITLE_GAP_MM / MM_PER_INCH * 72.0,
        )


def add_shared_mae_label(figure: plt.Figure, axes: np.ndarray) -> None:
    """Position the shared x label at one fifth of its initial tick-label gap."""
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    label = figure.text(
        (left + right) / 2.0,
        SHARED_XLABEL_Y_MM / FIGURE_MM[1],
        "MAE (years)",
        ha="center",
        va="bottom",
        fontsize=AXIS_TITLE_FONT_SIZE,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    ticks = [
        item.get_window_extent(renderer).transformed(figure.transFigure.inverted())
        for axis in axes
        for item in axis.get_xticklabels()
        if item.get_visible() and item.get_text()
    ]
    if not ticks:
        raise RuntimeError("Cannot position the shared MAE label without x tick labels")
    bound = label.get_window_extent(renderer).transformed(figure.transFigure.inverted())
    initial_gap = min(item.y0 for item in ticks) - bound.y1
    if initial_gap <= 0:
        raise RuntimeError(f"The initial MAE-label gap is not positive: {initial_gap}")
    label.set_y(label.get_position()[1] + initial_gap * (1 - SHARED_XLABEL_GAP_FRACTION))


def add_legend(
    figure: plt.Figure,
    handles: Iterable[Line2D],
    *,
    title: str | None = None,
    ncol: int = 1,
) -> None:
    """Place a compact legend in the exact reserved right-side band."""
    figure.legend(
        handles=list(handles),
        title=title,
        loc="center left",
        bbox_to_anchor=(LEGEND_LEFT_MM / FIGURE_MM[0], 0.5),
        ncol=ncol,
        columnspacing=0.45,
        handlelength=0.8,
        handletextpad=0.18,
        labelspacing=0,
        borderpad=0,
        borderaxespad=0,
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
    """Draw exact formal circles and the optional highlighted MetaClock star."""
    indexed = metrics.set_index(label_column)
    for label in labels:
        is_meta = label == meta_label
        axis.scatter(
            indexed.at[label, "mae"],
            indexed.at[label, "pearson_r"],
            s=META_MARKER_SIZE if is_meta else (20 if meta_label is None else 17),
            marker="*" if is_meta else "o",
            color=colors[label],
            edgecolor="#111111" if is_meta else "#222222",
            linewidth=OUTLINE_WIDTH_PT,
            # The formal architecture panel uses opaque model markers.  The
            # component panels use 0.90 alpha for ordinary clocks.
            alpha=1.0 if is_meta or meta_label is None else component_alpha,
            zorder=5 if is_meta else 3,
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
    """Build exact legend symbols for the plotted series."""
    handles: list[Line2D] = []
    for label in labels:
        is_meta = label == meta_label
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*" if is_meta else "o",
                linestyle="None",
                markersize=(
                    META_LEGEND_MARKER_SIZE
                    if is_meta
                    else (3.8 if meta_label is None else 3.5)
                ),
                markerfacecolor=colors[label],
                markeredgecolor="#111111" if is_meta else "#222222",
                markeredgewidth=OUTLINE_WIDTH_PT,
                alpha=1.0 if is_meta or meta_label is None else component_alpha,
                label=(
                    meta_display_label
                    if is_meta and meta_display_label
                    else display_labels.get(label, label)
                ),
            )
        )
    return handles


def finish_layout(figure: plt.Figure, axes: np.ndarray) -> None:
    """Add the shared dataset titles and x label after all points are drawn."""
    add_dataset_titles(figure, axes)
    add_shared_mae_label(figure, axes)


def save_vector_pair(figure: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Write matching PDF/SVG outputs without bbox resizing."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    figure.savefig(pdf_path)
    figure.savefig(svg_path)
    plt.close(figure)
    return pdf_path, svg_path
