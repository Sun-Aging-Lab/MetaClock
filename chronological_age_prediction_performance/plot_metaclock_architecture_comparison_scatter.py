#!/usr/bin/env python3
"""Compare individual clock and MetaClock architecture performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .plotting import (
    ARCHITECTURE_COLORS,
    ARCHITECTURES,
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    DATASETS,
    META_COLOR,
    META_LABEL,
    META_LEGEND_MARKER_SIZE,
    META_MARKER_SIZE,
    OUTLINE_WIDTH_PT,
    add_legend,
    configure_manuscript_style,
    finish_layout,
    make_dataset_axes,
    save_vector_pair,
    style_dataset_axis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/chronological_age_prediction/architecture_comparison"
COMPONENT_GROUP = "Single clock"
ARCHITECTURE_GROUP = "MetaClock architecture"
ARCHITECTURES_FOR_COMPARISON = ARCHITECTURES[:7]
FULL_COMPONENTS = tuple(sorted(set(COMPONENT_LABELS).difference({META_LABEL})))
def read_comparison_metrics(path: Path) -> pd.DataFrame:
    """Read the complete semantic source table for the comparison plot."""
    columns = ["dataset", "group", "label", "n", "mae", "pearson_r"]
    table = pd.read_csv(path)
    if missing := set(columns).difference(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table = table.loc[:, columns].copy()
    table[["n", "mae", "pearson_r"]] = table[["n", "mae", "pearson_r"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if table.isna().any().any() or (table["n"] <= 0).any() or (table["n"] % 1 != 0).any():
        raise ValueError(f"{path} contains invalid comparison metrics")
    expected = {
        COMPONENT_GROUP: FULL_COMPONENTS,
        ARCHITECTURE_GROUP: ARCHITECTURES_FOR_COMPARISON,
    }
    if set(table["group"]) != set(expected):
        raise ValueError(f"{path} has unexpected comparison groups")
    for group, labels in expected.items():
        subset = table.loc[table["group"].eq(group), ["dataset", "label"]]
        observed = pd.MultiIndex.from_frame(subset)
        grid = pd.MultiIndex.from_product([DATASETS, labels], names=["dataset", "label"])
        if observed.has_duplicates or set(observed) != set(grid) or len(observed) != len(grid):
            raise ValueError(f"{path} lacks a complete dataset × label grid for {group}")
    return table.sort_values(["dataset", "group", "label"]).reset_index(drop=True)


def add_architecture_boundary(
    axis: Axes, architectures: pd.DataFrame, all_points: pd.DataFrame
) -> None:
    """Mark the compact region occupied by the supervised MetaClock variants."""
    full_x_span = max(float(all_points["mae"].max() - all_points["mae"].min()), 1e-6)
    full_y_span = max(float(all_points["pearson_r"].max() - all_points["pearson_r"].min()), 1e-6)
    x_span = float(architectures["mae"].max() - architectures["mae"].min())
    y_span = float(architectures["pearson_r"].max() - architectures["pearson_r"].min())
    x_pad = max(0.16 * x_span, 0.04 * full_x_span)
    y_pad = max(0.16 * y_span, 0.08 * full_y_span)
    axis.add_patch(
        Rectangle(
            (
                float(architectures["mae"].min()) - x_pad,
                float(architectures["pearson_r"].min()) - y_pad,
            ),
            x_span + 2 * x_pad,
            y_span + 2 * y_pad,
            facecolor=(0.92, 0.92, 0.92, 0.22),
            edgecolor="#555555",
            linewidth=OUTLINE_WIDTH_PT,
            linestyle=(0, (4, 2.5)),
            zorder=1,
        )
    )


def plot_comparison(metrics: pd.DataFrame, output_stem: Path) -> None:
    """Draw all component dots and all supervised MetaClock architecture stars."""
    figure, axes = make_dataset_axes()
    component_labels = FULL_COMPONENTS
    for index, (dataset, axis) in enumerate(zip(DATASETS, axes, strict=True)):
        panel = metrics.loc[metrics["dataset"].eq(dataset)]
        components = panel.loc[panel["group"].eq(COMPONENT_GROUP)].set_index("label")
        architectures = panel.loc[panel["group"].eq(ARCHITECTURE_GROUP)].set_index("label")
        add_architecture_boundary(axis, architectures.reset_index(), panel)
        for label in component_labels:
            axis.scatter(
                components.at[label, "mae"],
                components.at[label, "pearson_r"],
                s=17,
                color=COMPONENT_COLORS[label],
                edgecolor="#222222",
                linewidth=OUTLINE_WIDTH_PT,
                alpha=0.55,
                zorder=3,
            )
        for label in ARCHITECTURES_FOR_COMPARISON:
            axis.scatter(
                architectures.at[label, "mae"],
                architectures.at[label, "pearson_r"],
                s=META_MARKER_SIZE,
                marker="*",
                color=ARCHITECTURE_COLORS[label],
                edgecolor="#111111",
                linewidth=OUTLINE_WIDTH_PT,
                zorder=4,
            )
        style_dataset_axis(axis, dataset, index)
    finish_layout(figure, axes)
    add_legend(
        figure,
        [
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=META_LEGEND_MARKER_SIZE,
                markerfacecolor=META_COLOR,
                markeredgecolor="#111111",
                markeredgewidth=OUTLINE_WIDTH_PT,
                label="MetaClock",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=3.5,
                markerfacecolor="#9E9E9E",
                markeredgecolor="#222222",
                markeredgewidth=OUTLINE_WIDTH_PT,
                alpha=0.55,
                label="Component model",
            ),
        ],
    )
    save_vector_pair(figure, output_stem)


def parse_args() -> argparse.Namespace:
    """Parse optional project-relative or absolute input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Create the component-clock versus architecture comparison plot."""
    args = parse_args()
    configure_manuscript_style()
    metrics = read_comparison_metrics(args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        args.output_dir / "pearson_mae_submodel_architectures_source_data.csv",
        index=False,
    )
    plot_comparison(metrics, args.output_dir / "pearson_mae_submodel_architectures")


if __name__ == "__main__":
    main()
