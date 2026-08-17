"""Exact 90 x 60 mm styling for current Figure 3 generalization panels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/metaclock_figure3_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/metaclock_figure3_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

from .aggregation import PLOT_MODELS, SUMMARY_COLUMNS
from .fonts import register_fonts
from .modeling import DatasetSpec, project_path

PANEL_WIDTH_MM = 90.0
PANEL_HEIGHT_MM = 60.0
FIGURE_TITLE_FONT_SIZE_PT = 10.5
SUBPLOT_TITLE_FONT_SIZE_PT = 9.5
AXIS_TITLE_FONT_SIZE_PT = 9.75
TICK_FONT_SIZE_PT = 8.0
LEGEND_FONT_SIZE_PT = 6.0
OUTLINE_WIDTH_PT = 0.2
SUBPLOT_TITLE_PAD_PT = 1.0
GENERALIZATION_AXES_TOP = 0.787
BASELINE_COLOR = "#7A7A7A"
META_COLOR = "#9C89B8"
MEAN_BASELINE_COLOR = "#9EC5E5"
UNIVERSAL_BASELINE_COLOR = "#EFC28B"


def configure_style() -> None:
    """Apply the typography and outline settings used by current formal panels."""
    register_fonts()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "font.size": TICK_FONT_SIZE_PT,
            "figure.titlesize": FIGURE_TITLE_FONT_SIZE_PT,
            "axes.titlesize": SUBPLOT_TITLE_FONT_SIZE_PT,
            "axes.labelsize": AXIS_TITLE_FONT_SIZE_PT,
            "xtick.labelsize": TICK_FONT_SIZE_PT,
            "ytick.labelsize": TICK_FONT_SIZE_PT,
            "legend.fontsize": LEGEND_FONT_SIZE_PT,
            "legend.title_fontsize": LEGEND_FONT_SIZE_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": OUTLINE_WIDTH_PT,
            "xtick.major.width": OUTLINE_WIDTH_PT,
            "xtick.minor.width": OUTLINE_WIDTH_PT,
            "ytick.major.width": OUTLINE_WIDTH_PT,
            "ytick.minor.width": OUTLINE_WIDTH_PT,
            "patch.linewidth": OUTLINE_WIDTH_PT,
            "grid.linewidth": 0.4,
            "legend.borderpad": 0.25,
            "legend.labelspacing": 0.25,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "text.color": "#262626",
            "axes.labelcolor": "#262626",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "axes.edgecolor": "#CCCCCC",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.fallback": None,
            "text.usetex": False,
        }
    )


def read_plot_summary(path: Path, dataset: str) -> dict[str, pd.DataFrame]:
    """Validate and split one aggregated summary by model."""
    frame = pd.read_csv(project_path(path))
    if set(frame.columns) != set(SUMMARY_COLUMNS) or set(frame["dataset"]) != {dataset}:
        raise ValueError(f"Invalid plot summary: {path}")
    expected = set(PLOT_MODELS)
    if set(frame["model"]) != expected:
        raise ValueError(f"{path}: model set differs")
    if frame.duplicated(["model", "training_size"]).any():
        raise ValueError(f"{path}: duplicate model/training-size rows")
    numeric = frame[["training_size", "mean", "q025", "q975"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path}: non-finite plot values")
    return {
        name: frame.loc[frame["model"].eq(name)].sort_values("training_size") for name in expected
    }


def plot_range(
    axis: Axes,
    summary: pd.DataFrame,
    label: str,
    color: str,
    marker: str,
    dodge: float,
    *,
    linestyle: str = "-",
    linewidth: float = 0.98,
    fill_alpha: float = 0.30,
    zorder: int = 3,
) -> None:
    """Draw one mean curve and its 95% resampling interval."""
    x = summary["training_size"] * (1.0 + dodge * 0.05)
    axis.plot(
        x,
        summary["mean"],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=2.8,
        markeredgewidth=OUTLINE_WIDTH_PT,
        alpha=0.9,
        zorder=zorder,
    )
    axis.fill_between(
        x,
        summary["q025"],
        summary["q975"],
        color=color,
        alpha=fill_alpha,
        zorder=zorder,
    )


def finish_axis(axis: Axes, displayed: list[pd.DataFrame], *, show_ylabel: bool) -> None:
    """Apply formal limits, ticks, grid, and legend placement."""
    lower = min(float(frame["q025"].min()) for frame in displayed)
    upper = max(float(frame["q975"].max()) for frame in displayed)
    axis.set_ylim(0.8 * lower, 1.2 * upper)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("# Training Samples")
    axis.set_ylabel("MAE" if show_ylabel else "")
    values = sorted(int(value) for value in displayed[0]["training_size"].unique())
    ticks = [values[index] for index in sorted({0, len(values) // 2, len(values) - 1})]
    axis.set_xscale("log")
    axis.set_xticks(ticks, labels=[str(value) for value in ticks])
    axis.minorticks_off()
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    legend = axis.legend(loc="best", frameon=False, fontsize=LEGEND_FONT_SIZE_PT, handletextpad=0.3)
    legend.set_alignment("left")


def add_title(figure: plt.Figure, title: str, *, italic: bool) -> None:
    """Place the one-line formal panel title at its audited coordinates."""
    if len(title.replace("\\n", "\n").splitlines()) != 1:
        raise ValueError(f"Expected one-line title, found {title!r}")
    figure.text(
        0.575,
        0.890,
        title,
        ha="center",
        va="top",
        fontsize=FIGURE_TITLE_FONT_SIZE_PT,
        fontstyle="italic" if italic else "normal",
        fontweight="bold",
    )


def plot_generalization_panel(
    summary_path: Path,
    spec: DatasetSpec,
    output_stem: Path,
    *,
    include_mean_baseline: bool = True,
    universal_mae: float | None = None,
    italic_title: bool = False,
) -> tuple[Path, Path]:
    """Render one paired SVM/EN panel with the requested baseline layout."""
    if universal_mae is not None and not include_mean_baseline:
        raise ValueError("Universal Clock is only shown in the baseline-comparison layout")
    summaries = read_plot_summary(summary_path, spec.name)
    configure_style()
    figure, (svm_axis, en_axis) = plt.subplots(
        1, 2, figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4)
    )
    for axis, base_label, meta_label in (
        (svm_axis, "SVM", "Meta SVM"),
        (en_axis, "EN", "Meta EN"),
    ):
        mean_label = "Mean SVM" if base_label == "SVM" else "Mean EN"
        displayed: list[pd.DataFrame] = []
        if include_mean_baseline:
            plot_range(
                axis,
                summaries[mean_label],
                "Mean",
                MEAN_BASELINE_COLOR,
                "D",
                0,
                linestyle="--",
                linewidth=0.77,
                fill_alpha=0.08,
                zorder=1,
            )
            displayed.append(summaries[mean_label])
            if universal_mae is not None:
                axis.axhline(
                    universal_mae,
                    label="Universal Clock",
                    color=UNIVERSAL_BASELINE_COLOR,
                    linestyle="--",
                    linewidth=0.84,
                    zorder=2,
                )
                constant = summaries[base_label][["training_size"]].copy()
                constant[["mean", "q025", "q975"]] = universal_mae
                displayed.append(constant)
        plot_range(axis, summaries[base_label], base_label, BASELINE_COLOR, "o", -1, linewidth=0.70)
        plot_range(axis, summaries[meta_label], meta_label, META_COLOR, "s", 1, linewidth=0.98)
        displayed.extend((summaries[base_label], summaries[meta_label]))
        axis.set_title(base_label, fontweight="normal", pad=SUBPLOT_TITLE_PAD_PT)
        finish_axis(axis, displayed, show_ylabel=axis is svm_axis)
    add_title(figure, spec.title, italic=italic_title)
    figure.subplots_adjust(
        left=0.22, right=0.96, bottom=0.23, top=GENERALIZATION_AXES_TOP, wspace=0.48
    )
    stem = project_path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = (stem.with_suffix(".pdf"), stem.with_suffix(".svg"))
    for output in outputs:
        figure.savefig(output, dpi=300)
    plt.close(figure)
    return outputs


def plot_group_main(
    specs: Mapping[str, DatasetSpec],
    default_summaries: Path,
    default_figures: Path,
    description: str,
) -> int:
    """Shared array/tissue plot CLI implementation."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=specs, action="append")
    parser.add_argument("--summary-root", type=Path, default=default_summaries)
    parser.add_argument("--output-dir", type=Path, default=default_figures)
    args = parser.parse_args()
    for name in args.dataset or list(specs):
        spec = specs[name]
        summary = args.summary_root / f"{name}_plot_summary.csv"
        for suffix, include_mean_baseline in (("", False), ("_with_mean_baseline", True)):
            outputs = plot_generalization_panel(
                summary,
                spec,
                args.output_dir / f"{name}_fewshot_clock_mi_svm{suffix}",
                include_mean_baseline=include_mean_baseline,
            )
            print(*outputs, sep="\n")
    return 0
