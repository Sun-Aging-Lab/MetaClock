#!/usr/bin/env python3
"""Draw chronological-age model and component-clock performance scatter plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

try:  # Supports both ``python -m`` and direct-script execution.
    from .plotting import (
        ARCHITECTURE_COLORS,
        ARCHITECTURE_LABELS,
        ARCHITECTURES,
        COMPONENT_COLORS,
        COMPONENT_LABELS,
        DATASETS,
        add_legend,
        add_shared_mae_label,
        configure_manuscript_style,
        make_dataset_axes,
        metric_legend_handles,
        plot_labeled_metrics,
        save_vector_pair,
        style_dataset_axis,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from plotting import (
        ARCHITECTURE_COLORS,
        ARCHITECTURE_LABELS,
        ARCHITECTURES,
        COMPONENT_COLORS,
        COMPONENT_LABELS,
        DATASETS,
        add_legend,
        add_shared_mae_label,
        configure_manuscript_style,
        make_dataset_axes,
        metric_legend_handles,
        plot_labeled_metrics,
        save_vector_pair,
        style_dataset_axis,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHITECTURE_METRICS = (
    PROJECT_ROOT / "data/chronological_age_prediction/model_architecture_metrics.csv"
)  # fill in your path
DEFAULT_COMPONENT_METRICS = (
    PROJECT_ROOT / "data/chronological_age_prediction/component_clock_metrics.csv"
)  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/chronological_age_prediction/scatter"


def read_complete_metrics(
    path: Path,
    *,
    label_column: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Read a numeric, duplicate-free metrics table with a full dataset-label grid."""
    required = {"dataset", label_column, "mae", "pearson_r"}
    metrics = pd.read_csv(path)
    if missing := required.difference(metrics.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    metrics = metrics.loc[:, ["dataset", label_column, "mae", "pearson_r"]].copy()
    metrics[["mae", "pearson_r"]] = metrics[["mae", "pearson_r"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if metrics.isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric plotting values")
    observed = pd.MultiIndex.from_frame(metrics[["dataset", label_column]])
    expected = pd.MultiIndex.from_product([DATASETS, labels], names=["dataset", label_column])
    if observed.has_duplicates or set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError(f"{path} must contain exactly one row for each dataset × {label_column}")
    return metrics


def plot_metric_scatter(
    metrics: pd.DataFrame,
    *,
    label_column: str,
    labels: Sequence[str],
    colors: dict[str, object],
    display_labels: dict[str, str],
    legend_title: str,
    output_stem: Path,
) -> None:
    """Plot MAE against Pearson r in one four-dataset scatter layout."""
    figure, axes = make_dataset_axes()
    for index, (dataset, axis) in enumerate(zip(DATASETS, axes, strict=True)):
        plot_labeled_metrics(
            axis,
            metrics.loc[metrics["dataset"].eq(dataset)],
            label_column=label_column,
            labels=labels,
            colors=colors,
        )
        style_dataset_axis(axis, dataset, index)
    add_shared_mae_label(figure)
    labels_for_legend = {label: display_labels.get(label, label) for label in labels}
    add_legend(
        figure,
        metric_legend_handles(
            labels,
            colors=colors,
            display_labels=labels_for_legend,
        ),
        title=legend_title,
    )
    save_vector_pair(figure, output_stem)


def parse_args() -> argparse.Namespace:
    """Parse optional project-relative or absolute input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-metrics", type=Path, default=DEFAULT_ARCHITECTURE_METRICS)
    parser.add_argument("--component-metrics", type=Path, default=DEFAULT_COMPONENT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Create model-architecture and component-clock chronological-age plots."""
    args = parse_args()
    configure_manuscript_style()
    architecture = read_complete_metrics(
        args.architecture_metrics, label_column="model", labels=ARCHITECTURES
    )
    component = read_complete_metrics(
        args.component_metrics,
        label_column="predictor",
        labels=tuple(COMPONENT_LABELS),
    )
    plot_metric_scatter(
        architecture,
        label_column="model",
        labels=ARCHITECTURES,
        colors=ARCHITECTURE_COLORS,
        display_labels=ARCHITECTURE_LABELS,
        legend_title="Model architecture",
        output_stem=args.output_dir / "pearson_mae_model_architecture",
    )
    plot_metric_scatter(
        component,
        label_column="predictor",
        labels=tuple(COMPONENT_LABELS),
        colors=COMPONENT_COLORS,
        display_labels=COMPONENT_LABELS,
        legend_title="Clock",
        output_stem=args.output_dir / "pearson_mae_component_clock",
    )


if __name__ == "__main__":
    main()
