"""Render Figure 4 biological-age performance and replicate-reliability panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import FIGURE4_MODELS
    from .plotting import COLORS, LABELS, clean_axis, ordered_rows, save_vector_pair
except ImportError:  # Direct-script fallback.
    from config import FIGURE4_MODELS
    from plotting import COLORS, LABELS, clean_axis, ordered_rows, save_vector_pair


ORIGINAL_AXES_WIDTH = (0.93 - 0.25) / (2.0 + 1.75)
REFERENCE_AXES_HEIGHT = 0.94 - 0.27
REFERENCE_AXES_WIDTH = ORIGINAL_AXES_WIDTH * REFERENCE_AXES_HEIGHT / (0.82 - 0.33)
LEFT_AXES_CENTER = 0.25 + ORIGINAL_AXES_WIDTH / 2
RIGHT_AXES_CENTER = 0.93 - ORIGINAL_AXES_WIDTH / 2
PREDICTION_LEFT = LEFT_AXES_CENTER - REFERENCE_AXES_WIDTH / 2
PREDICTION_RIGHT = RIGHT_AXES_CENTER + REFERENCE_AXES_WIDTH / 2
PREDICTION_WSPACE = (
    PREDICTION_RIGHT - PREDICTION_LEFT - 2 * REFERENCE_AXES_WIDTH
) / REFERENCE_AXES_WIDTH


def plot_prediction_metrics(source_dir: Path, output_dir: Path) -> None:
    """Plot OOF MAE and Pearson correlation."""
    table = ordered_rows(
        pd.read_csv(source_dir / "bioage_prediction_metrics_source_data.csv"), FIGURE4_MODELS
    )
    figure, axes = plt.subplots(1, 2, figsize=(60 / 25.4, 50 / 25.4))
    figure.subplots_adjust(
        left=PREDICTION_LEFT,
        right=PREDICTION_RIGHT,
        bottom=0.24,
        top=0.91,
        wspace=PREDICTION_WSPACE,
    )
    panels = (
        ("mae", "MAE (years)", "Prediction error", (0, 4.9), np.arange(0, 5)),
        ("pearson_r", "Pearson r", "Correlation", (0.90, 1.00), np.arange(0.90, 1.001, 0.02)),
    )
    for axis, (column, ylabel, title, limits, ticks) in zip(axes, panels, strict=True):
        axis.bar(
            [LABELS[model] for model in FIGURE4_MODELS],
            table[column],
            width=0.8,
            color=[COLORS[model] for model in FIGURE4_MODELS],
            edgecolor="#333333",
            linewidth=0.2,
        )
        axis.set(ylabel=ylabel, title=title, ylim=limits, yticks=ticks)
        axis.tick_params(axis="x", rotation=35)
        clean_axis(axis)
    save_vector_pair(figure, output_dir, "bioage_prediction_mae_pearson")


def plot_duplicate_reliability(source_dir: Path, output_dir: Path) -> None:
    """Plot mean absolute paired-replicate differences."""
    table = ordered_rows(
        pd.read_csv(source_dir / "duplicate_sample_reliability_source_data.csv"),
        FIGURE4_MODELS,
    )
    if any("icc" in column.lower() for column in table.columns):
        raise ValueError("The current reliability panel must not contain ICC.")
    figure, axis = plt.subplots(figsize=(60 / 25.4, 50 / 25.4))
    left = (1 - REFERENCE_AXES_WIDTH) / 2
    figure.subplots_adjust(left=left, right=left + REFERENCE_AXES_WIDTH, bottom=0.24, top=0.91)
    axis.bar(
        [LABELS[model] for model in FIGURE4_MODELS],
        table["mean_absolute_delta"],
        width=0.8,
        color=[COLORS[model] for model in FIGURE4_MODELS],
        edgecolor="#333333",
        linewidth=0.2,
    )
    axis.set(ylabel="Mean |Δ| (years)", title="Variability Across Replicates", ylim=(0, 3.5))
    axis.tick_params(axis="x", rotation=25)
    clean_axis(axis)
    save_vector_pair(figure, output_dir, "duplicate_sample_reliability")
