#!/usr/bin/env python3
"""Plot cumulative component-clock ablation performance scatter plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .plotting import (
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    DATASETS,
    add_legend,
    configure_manuscript_style,
    finish_layout,
    make_dataset_axes,
    metric_legend_handles,
    plot_labeled_metrics,
    save_vector_pair,
    style_dataset_axis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_DIR = (
    Path("/path/to/your/file")  # fill in your path
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/chronological_age_prediction/ablation"
REMOVAL_ORDER = ("zhangblup", "han", "skinandblood")
UNPLOTTED_LABELS = ("yingcausage",)


def read_ablation_metrics(path: Path, step: int) -> pd.DataFrame:
    """Read and validate one cumulative-ablation table without duplicate I/O."""
    raw = pd.read_csv(path)
    required = {"dataset", "predictor", "n", "mae", "pearson_r", "ablation_step", "removed_clocks"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"{path} is missing ablation columns: {sorted(missing)}")
    if set(raw["ablation_step"]) != {step}:
        raise ValueError(f"{path} does not contain only ablation step {step}")
    expected_removed = ";".join(REMOVAL_ORDER[:step])
    if set(raw["removed_clocks"]) != {expected_removed}:
        raise ValueError(f"{path} has an unexpected cumulative removal list")
    expected_labels = tuple(
        label
        for label in (*COMPONENT_LABELS, *UNPLOTTED_LABELS)
        if label not in REMOVAL_ORDER[:step]
    )
    numeric = raw[["n", "mae", "pearson_r", "ablation_step"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_n = (
        numeric["n"].isna() | (numeric["n"] <= 0) | (numeric["n"] % 1 != 0)
    )
    if invalid_n.any():
        raise ValueError(f"{path} must contain positive integer sample counts in n")
    raw[["n", "mae", "pearson_r", "ablation_step"]] = numeric
    if raw[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric plotting values")
    observed = pd.MultiIndex.from_frame(raw[["dataset", "predictor"]])
    expected = pd.MultiIndex.from_product(
        [DATASETS, expected_labels], names=["dataset", "predictor"]
    )
    if observed.has_duplicates or set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError(f"{path} must contain exactly one row for every retained label")
    return raw.loc[:, ["dataset", "predictor", "mae", "pearson_r"]].copy()


def plot_ablation_step(metrics: pd.DataFrame, step: int, output_dir: Path) -> None:
    """Draw one ablation step, omitting YingCausAge from the displayed series."""
    retained = tuple(
        label
        for label in sorted(set(COMPONENT_LABELS).difference({"MetaClock"}))
        if label not in REMOVAL_ORDER[:step]
    )
    figure, axes = make_dataset_axes()
    for index, (dataset, axis) in enumerate(zip(DATASETS, axes, strict=True)):
        plot_labeled_metrics(
            axis,
            metrics.loc[metrics["dataset"].eq(dataset)],
            label_column="predictor",
            labels=retained,
            colors=COMPONENT_COLORS,
        )
        # Draw the highlighted retrained MetaClock after the ordinary clocks,
        # matching the formal SVG PathCollection order and z-order.
        plot_labeled_metrics(
            axis,
            metrics.loc[metrics["dataset"].eq(dataset)],
            label_column="predictor",
            labels=("MetaClock",),
            colors=COMPONENT_COLORS,
            meta_label="MetaClock",
        )
        style_dataset_axis(axis, dataset, index)
    finish_layout(figure, axes)
    removed = COMPONENT_LABELS[REMOVAL_ORDER[step - 1]]
    title = figure.text(
        (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2,
        0.98,
        f"Cumulative ablation {step}/3: {removed} removed",
        ha="center",
        va="top",
        fontsize=9.5 + 1.0,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    dataset_title_top = max(axis.title.get_window_extent(renderer).y1 for axis in axes)
    title_bound = title.get_window_extent(renderer)
    target_bottom = dataset_title_top + 5.0 * figure.dpi / 72.0
    title.set_y(title.get_position()[1] + (target_bottom - title_bound.y0) / figure.bbox.height)
    add_legend(
        figure,
        metric_legend_handles(
            ("MetaClock", *retained),
            colors=COMPONENT_COLORS,
            display_labels=COMPONENT_LABELS,
            meta_label="MetaClock",
            meta_display_label="Retrained MetaClock",
        ),
        title="Clock",
    )
    save_vector_pair(
        figure,
        output_dir / f"ablation_step_{step:02d}_without_{REMOVAL_ORDER[step - 1]}",
    )


def parse_args() -> argparse.Namespace:
    """Parse optional project-relative or absolute input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Create all three cumulative component-clock ablation plots."""
    args = parse_args()
    configure_manuscript_style()
    for step in range(1, len(REMOVAL_ORDER) + 1):
        metrics = read_ablation_metrics(
            args.ablation_dir / f"step_{step:02d}_source_data.csv", step
        )
        plot_ablation_step(metrics, step, args.output_dir)


if __name__ == "__main__":
    main()
