#!/usr/bin/env python3
"""Compare individual clock and MetaClock architecture performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

try:  # Supports both ``python -m`` and direct-script execution.
    from .plotting import (
        ARCHITECTURE_COLORS,
        ARCHITECTURES,
        COMPONENT_COLORS,
        COMPONENT_LABELS,
        DATASETS,
        META_COLOR,
        META_LABEL,
        add_legend,
        add_shared_mae_label,
        configure_manuscript_style,
        make_dataset_axes,
        save_vector_pair,
        style_dataset_axis,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from plotting import (
        ARCHITECTURE_COLORS,
        ARCHITECTURES,
        COMPONENT_COLORS,
        COMPONENT_LABELS,
        DATASETS,
        META_COLOR,
        META_LABEL,
        add_legend,
        add_shared_mae_label,
        configure_manuscript_style,
        make_dataset_axes,
        save_vector_pair,
        style_dataset_axis,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_METRICS = (
    PROJECT_ROOT / "data/chronological_age_prediction/component_and_architecture_metrics.csv"
)  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/chronological_age_prediction/architecture_comparison"
COMPONENT_GROUP = "Single clock"
ARCHITECTURE_GROUP = "MetaClock architecture"
ARCHITECTURES_FOR_COMPARISON = ARCHITECTURES[:7]


def read_comparison_metrics(path: Path) -> pd.DataFrame:
    """Read exactly one component/architecture value for every dataset-label cell."""
    required = {"dataset", "group", "label", "mae", "pearson_r"}
    table = pd.read_csv(path)
    if missing := required.difference(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table = table.loc[:, ["dataset", "group", "label", "mae", "pearson_r"]].copy()
    table[["mae", "pearson_r"]] = table[["mae", "pearson_r"]].apply(pd.to_numeric, errors="coerce")
    if table.isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric plotting values")
    expected_by_group = {
        COMPONENT_GROUP: tuple(label for label in COMPONENT_LABELS if label != META_LABEL),
        ARCHITECTURE_GROUP: ARCHITECTURES_FOR_COMPARISON,
    }
    if set(table["group"]) != set(expected_by_group):
        raise ValueError(f"{path} has unexpected comparison groups")
    for group, labels in expected_by_group.items():
        subset = table.loc[table["group"].eq(group)]
        observed = pd.MultiIndex.from_frame(subset[["dataset", "label"]])
        expected = pd.MultiIndex.from_product([DATASETS, labels], names=["dataset", "label"])
        if (
            observed.has_duplicates
            or set(observed) != set(expected)
            or len(observed) != len(expected)
        ):
            raise ValueError(f"{path} lacks a complete dataset × label grid for {group}")
    return table


def add_architecture_boundary(axis: Axes, architectures: pd.DataFrame) -> None:
    """Mark the compact region occupied by the supervised MetaClock variants."""
    x_span = max(float(architectures["mae"].max() - architectures["mae"].min()), 1e-6)
    y_span = max(
        float(architectures["pearson_r"].max() - architectures["pearson_r"].min()),
        1e-6,
    )
    axis.add_patch(
        Rectangle(
            (
                float(architectures["mae"].min()) - 0.16 * x_span,
                float(architectures["pearson_r"].min()) - 0.16 * y_span,
            ),
            x_span * 1.32,
            y_span * 1.32,
            facecolor=(0.8, 0.8, 0.8, 0.20),
            edgecolor="#555555",
            linewidth=0.5,
            linestyle=(0, (4, 2.5)),
            zorder=1,
        )
    )


def plot_comparison(metrics: pd.DataFrame, output_stem: Path) -> None:
    """Draw all component dots and all supervised MetaClock architecture stars."""
    figure, axes = make_dataset_axes()
    component_labels = tuple(label for label in COMPONENT_LABELS if label != META_LABEL)
    for index, (dataset, axis) in enumerate(zip(DATASETS, axes, strict=True)):
        panel = metrics.loc[metrics["dataset"].eq(dataset)]
        components = panel.loc[panel["group"].eq(COMPONENT_GROUP)].set_index("label")
        architectures = panel.loc[panel["group"].eq(ARCHITECTURE_GROUP)].set_index("label")
        add_architecture_boundary(axis, architectures.reset_index())
        for label in component_labels:
            axis.scatter(
                components.at[label, "mae"],
                components.at[label, "pearson_r"],
                s=22,
                color=COMPONENT_COLORS[label],
                edgecolor="#202020",
                linewidth=0.45,
                alpha=0.55,
                zorder=3,
            )
        for label in ARCHITECTURES_FOR_COMPARISON:
            axis.scatter(
                architectures.at[label, "mae"],
                architectures.at[label, "pearson_r"],
                s=86,
                marker="*",
                color=ARCHITECTURE_COLORS[label],
                edgecolor="#202020",
                linewidth=0.45,
                zorder=4,
            )
        style_dataset_axis(axis, dataset, index)
    add_shared_mae_label(figure)
    add_legend(
        figure,
        [
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=7,
                markerfacecolor=META_COLOR,
                markeredgecolor="#202020",
                label="MetaClock architecture",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4,
                markerfacecolor="#999999",
                markeredgecolor="#202020",
                alpha=0.55,
                label="Component model",
            ),
        ],
    )
    save_vector_pair(figure, output_stem)


def parse_args() -> argparse.Namespace:
    """Parse optional project-relative or absolute input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_COMPARISON_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Create the component-clock versus architecture comparison plot."""
    args = parse_args()
    configure_manuscript_style()
    plot_comparison(
        read_comparison_metrics(args.metrics),
        args.output_dir / "pearson_mae_component_and_architecture",
    )


if __name__ == "__main__":
    main()
