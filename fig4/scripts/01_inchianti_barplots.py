#!/usr/bin/env python3
"""Summarize and plot InCHIANTI OOF prediction and duplicate reliability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


FIGURE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARTICIPANT_PREDICTIONS = (
    FIGURE_DIR / "datasets/inchianti/oof_participant_predictions.csv"
)
DEFAULT_ALL_RECORD_PREDICTIONS = (
    FIGURE_DIR / "datasets/inchianti/oof_all_record_predictions.csv"
)
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "results"

MODELS = ("bio_metaclock", "baseline")
MODEL_LABELS = {"bio_metaclock": "BioMeta", "baseline": "Baseline"}
MODEL_COLORS = {"bio_metaclock": "#0072B2", "baseline": "#4D4D4D"}
MM_TO_INCH = 1.0 / 25.4
FIGSIZE = (60 * MM_TO_INCH, 50 * MM_TO_INCH)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    """Raise a readable error when an input table is incomplete."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def summarize_prediction_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate participant-level MAE and Pearson r for each focal model."""
    rows: list[dict[str, float | int | str]] = []
    for model in MODELS:
        values = predictions.loc[predictions["model"].eq(model)].copy()
        if values["participant_id"].duplicated().any():
            raise ValueError(f"{model} has more than one row per participant")
        observed = pd.to_numeric(values["observed_age"], errors="raise").to_numpy()
        predicted = pd.to_numeric(values["predicted_age"], errors="raise").to_numpy()
        if len(values) < 2 or not np.isfinite([observed, predicted]).all():
            raise ValueError(f"{model} has incomplete participant predictions")
        rows.append(
            {
                "model": model,
                "model_display": MODEL_LABELS[model],
                "n": len(values),
                "mae": float(mean_absolute_error(observed, predicted)),
                "pearson_r": float(pearsonr(observed, predicted).statistic),
            }
        )
    return pd.DataFrame(rows)


def absolute_agreement_icc(values: np.ndarray) -> float:
    """Return ICC(A,1) for an n-participant by two-replicate matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("ICC(A,1) requires at least two complete replicate pairs")
    n_participants, n_replicates = values.shape
    grand_mean = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_participant = (
        n_replicates
        * float(np.square(row_means - grand_mean).sum())
        / (n_participants - 1)
    )
    ms_replicate = (
        n_participants
        * float(np.square(column_means - grand_mean).sum())
        / (n_replicates - 1)
    )
    residual = values - row_means[:, None] - column_means[None, :] + grand_mean
    ms_error = float(np.square(residual).sum()) / (
        (n_participants - 1) * (n_replicates - 1)
    )
    denominator = (
        ms_participant
        + (n_replicates - 1) * ms_error
        + n_replicates * (ms_replicate - ms_error) / n_participants
    )
    return float((ms_participant - ms_error) / denominator)


def summarize_duplicate_reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize participants having exactly two OOF methylation records."""
    rows: list[dict[str, float | int | str]] = []
    for model in MODELS:
        values = predictions.loc[predictions["model"].eq(model)].copy()
        duplicate_ids = (
            values.groupby("participant_id")["sample_id"].nunique().loc[lambda x: x.eq(2)].index
        )
        values = values[values["participant_id"].isin(duplicate_ids)].copy()
        if values.groupby("participant_id")["fold"].nunique().ne(1).any():
            raise ValueError("Replicates from one participant must share an OOF fold")
        values["replicate"] = values.groupby("participant_id")["sample_id"].rank(
            method="first"
        ).astype(int)
        matrix = values.pivot(
            index="participant_id", columns="replicate", values="predicted_age"
        ).reindex(columns=[1, 2])
        matrix = matrix.apply(pd.to_numeric, errors="raise")
        if matrix.isna().any().any():
            raise ValueError(f"{model} has incomplete duplicate pairs")
        array = matrix.to_numpy(dtype=float)
        delta = np.abs(array[:, 0] - array[:, 1])
        rows.append(
            {
                "model": model,
                "n_pairs": len(matrix),
                "icc_a1": absolute_agreement_icc(array),
                "pearson_r": float(pearsonr(array[:, 0], array[:, 1]).statistic),
                "mean_absolute_delta": float(delta.mean()),
                "median_absolute_delta": float(np.median(delta)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write matching PDF and SVG files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_prediction_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot panel 1 MAE and Pearson r bars."""
    summary = summary.set_index("model").loc[list(MODELS)].reset_index()
    colors = [MODEL_COLORS[model] for model in summary["model"]]
    labels = [MODEL_LABELS[model] for model in summary["model"]]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    axes[0].bar(labels, summary["mae"], color=colors)
    axes[0].set_ylabel("MAE (years)")
    axes[1].bar(labels, summary["pearson_r"], color=colors)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_ylim(0.90, 1.00)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", linewidth=0.4, color="#D9D9D9")
        axis.set_axisbelow(True)
    save_figure(figure, output_dir / "01_prediction_barplot.pdf")


def plot_duplicate_reliability(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot panel 2 mean absolute difference between duplicate predictions."""
    summary = summary.set_index("model").loc[list(MODELS)].reset_index()
    figure, axis = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    axis.bar(
        [MODEL_LABELS[m] for m in summary["model"]],
        summary["mean_absolute_delta"],
        color=[MODEL_COLORS[m] for m in summary["model"]],
    )
    axis.set_ylabel(r"$\Delta$ predicted phenoage")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", linewidth=0.4, color="#D9D9D9")
    axis.set_axisbelow(True)
    save_figure(figure, output_dir / "02_duplicate_reliability_barplot.pdf")


def parse_args() -> argparse.Namespace:
    """Parse relative input paths and optional replacements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-predictions", type=Path, default=DEFAULT_PARTICIPANT_PREDICTIONS)
    parser.add_argument("--all-record-predictions", type=Path, default=DEFAULT_ALL_RECORD_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Build panels 1-2 and their result tables."""
    args = parse_args()
    participant = pd.read_csv(args.participant_predictions)
    require_columns(
        participant,
        {"participant_id", "model", "observed_age", "predicted_age"},
        args.participant_predictions,
    )
    all_records = pd.read_csv(args.all_record_predictions)
    require_columns(
        all_records,
        {"participant_id", "sample_id", "fold", "model", "predicted_age"},
        args.all_record_predictions,
    )
    performance = summarize_prediction_performance(participant)
    reliability = summarize_duplicate_reliability(all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(args.output_dir / "01_prediction_metrics.csv", index=False)
    reliability.to_csv(args.output_dir / "02_duplicate_reliability.csv", index=False)
    plot_prediction_metrics(performance, args.output_dir)
    plot_duplicate_reliability(reliability, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
