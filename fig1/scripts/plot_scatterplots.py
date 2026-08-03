#!/usr/bin/env python3
"""Plot the two Fig. 1 MAE--Pearson trade-off panels.

The script consumes compact result tables, not the row-level methylation data.
All defaults are relative to this figure directory and may be overridden by CLI.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

FIGURE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "metaclock-fig1-cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402


MM_PER_INCH = 25.4
FIGURE_MM = (150.0, 75.0)
DATASET_ORDER = ("GSE147221", "GSE157131", "GSE234461", "GSE72680")
MODEL_ORDER = (
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
MODEL_LABELS = {"LightGBM": "LGBM", "SpectralEnsemble": "SE", "SVR": "SVM"}
COMPONENT_ORDER = (
    "altumage",
    "dnamphenoage",
    "encen40",
    "han",
    "hannum",
    "horvath2013",
    "hrsinchphenoage",
    "lin",
    "pchannum",
    "pchorvath2013",
    "pcphenoage",
    "pcskinandblood",
    "skinandblood",
    "zhangblup",
    "zhangen",
)
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
}


def configure_style() -> None:
    """Apply the final print-size typography and vector export settings."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "legend.title_fontsize": 6.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_metrics(path: Path, label_column: str, expected_labels: tuple[str, ...]) -> pd.DataFrame:
    """Read and validate one panel's dataset-level MAE and Pearson r table."""
    required = {"dataset", label_column, "n", "mae", "pearson_r"}
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["dataset", label_column, "n", "mae", "pearson_r"]].copy()
    for column in ("n", "mae", "pearson_r"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric plotting values")
    if set(frame["dataset"]) != set(DATASET_ORDER):
        raise ValueError(f"{path} must contain exactly {list(DATASET_ORDER)}")
    if set(frame[label_column]) != set(expected_labels):
        raise ValueError(f"{path} has an unexpected {label_column} set")
    if frame.duplicated(["dataset", label_column]).any():
        raise ValueError(f"{path} has duplicate dataset/{label_column} rows")
    return frame


def make_axes() -> tuple[plt.Figure, np.ndarray]:
    """Create four equal square panels on the final 150 x 75 mm canvas."""
    size = tuple(value / MM_PER_INCH for value in FIGURE_MM)
    figure, axes = plt.subplots(1, 4, figsize=size, squeeze=False)
    axes = axes.reshape(-1)
    figure.subplots_adjust(left=0.08, right=0.992, bottom=0.425, top=0.91, wspace=0.34)
    for axis in axes:
        axis.set_box_aspect(1)
    return figure, axes


def style_axis(axis: plt.Axes, dataset: str, index: int) -> None:
    """Apply identical labels, ticks, and margins to one dataset panel."""
    axis.set_title(dataset, fontsize=7.0, pad=2.5)
    axis.set_ylabel("Pearson r" if index == 0 else "", labelpad=1.5)
    axis.tick_params(axis="both", labelsize=6.0, width=0.6, length=2.2, pad=1.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.margins(x=0.10, y=0.14)


def add_shared_xlabel(figure: plt.Figure, axes: np.ndarray) -> None:
    """Center a single MAE label below all four dataset panels."""
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    bottom = min(axis.get_position().y0 for axis in axes)
    figure.text((left + right) / 2, bottom - 0.065, "MAE (years)", ha="center", va="top", fontsize=6.5)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    """Write editable PDF and SVG versions without tight-bbox resizing."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def plot_model_architectures(metrics: pd.DataFrame, output_stem: Path) -> None:
    """Plot nine MetaClock architecture/ensemble alternatives by dataset."""
    tab20 = tuple(matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab20"].colors)
    colors = {name: tab20[index] for name, index in zip(MODEL_ORDER, range(0, 18, 2), strict=True)}
    figure, axes = make_axes()
    for index, (dataset, axis) in enumerate(zip(DATASET_ORDER, axes, strict=True)):
        panel = metrics[metrics["dataset"] == dataset].set_index("model")
        for model in MODEL_ORDER:
            axis.scatter(panel.at[model, "mae"], panel.at[model, "pearson_r"], s=20, color=colors[model], edgecolor="#222222", linewidth=0.35)
        style_axis(axis, dataset, index)
    add_shared_xlabel(figure, axes)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=3.8,
               markerfacecolor=colors[model], markeredgecolor="#222222",
               markeredgewidth=0.35, label=MODEL_LABELS.get(model, model))
        for model in MODEL_ORDER
    ]
    figure.legend(handles=handles, title="Model architecture", loc="lower center",
                  bbox_to_anchor=(0.5, 0.25), ncol=5, columnspacing=1.15,
                  handletextpad=0.35, labelspacing=0.35, borderaxespad=0)
    save_figure(figure, output_stem)


def plot_component_clocks(metrics: pd.DataFrame, output_stem: Path) -> None:
    """Plot the component clocks and emphasize the linear-SVR MetaClock."""
    tab20 = tuple(matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab20"].colors)
    component_colors = dict(zip(COMPONENT_ORDER, [c for i, c in enumerate(tab20) if i != 8][:15], strict=True))
    meta_color = tab20[8]
    figure, axes = make_axes()
    for index, (dataset, axis) in enumerate(zip(DATASET_ORDER, axes, strict=True)):
        panel = metrics[metrics["dataset"] == dataset].set_index("predictor")
        for predictor in COMPONENT_ORDER:
            axis.scatter(panel.at[predictor, "mae"], panel.at[predictor, "pearson_r"], s=17,
                         color=component_colors[predictor], edgecolor="#222222",
                         linewidth=0.3, alpha=0.9)
        axis.scatter(panel.at["MetaClock", "mae"], panel.at["MetaClock", "pearson_r"],
                     s=85.5, marker="*", color=meta_color, edgecolor="#111111",
                     linewidth=0.45, zorder=5)
        style_axis(axis, dataset, index)
    add_shared_xlabel(figure, axes)
    handles = [Line2D([0], [0], marker="*", linestyle="None", markersize=9,
                      markerfacecolor=meta_color, markeredgecolor="#111111", label="MetaClock")]
    handles.extend(
        Line2D([0], [0], marker="o", linestyle="None", markersize=3.5,
               markerfacecolor=component_colors[p], markeredgecolor="#222222",
               markeredgewidth=0.3, label=COMPONENT_LABELS[p])
        for p in COMPONENT_ORDER
    )
    figure.legend(handles=handles, title="Clock", loc="lower center",
                  bbox_to_anchor=(0.5, 0.18), ncol=4, columnspacing=0.8,
                  handletextpad=0.25, labelspacing=0.3, borderaxespad=0)
    save_figure(figure, output_stem)


def parse_args() -> argparse.Namespace:
    """Expose relative defaults while permitting explicit path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-metrics", type=Path, default=FIGURE_DIR / "results" / "01_model_architecture_scatter.csv")
    parser.add_argument("--component-metrics", type=Path, default=FIGURE_DIR / "results" / "02_component_clock_scatter.csv")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "figures")
    return parser.parse_args()


def main() -> None:
    """Validate the ordered source tables and regenerate both scatterplots."""
    args = parse_args()
    configure_style()
    model_metrics = read_metrics(args.model_metrics, "model", MODEL_ORDER)
    component_metrics = read_metrics(args.component_metrics, "predictor", (*COMPONENT_ORDER, "MetaClock"))
    plot_model_architectures(model_metrics, args.output_dir / "01_model_architecture_scatter")
    plot_component_clocks(component_metrics, args.output_dir / "02_component_clock_scatter")


if __name__ == "__main__":
    main()
