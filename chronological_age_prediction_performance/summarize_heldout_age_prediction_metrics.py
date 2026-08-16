#!/usr/bin/env python3
"""Summarize held-out chronological-age predictions into MAE and Pearson metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "data/chronological_age_prediction/heldout_predictions_long.csv"
)  # fill in your path
DEFAULT_OUTPUT = PROJECT_ROOT / "results/chronological_age_prediction/heldout_metrics.csv"
BASE_COLUMNS = {"dataset", "sample_id", "age", "prediction"}


def validate_predictions(predictions: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Validate a long-format held-out prediction table for one model label type."""
    required = {*BASE_COLUMNS, label_column}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    table = predictions.loc[:, ["dataset", "sample_id", "age", label_column, "prediction"]].copy()
    table[["age", "prediction"]] = table[["age", "prediction"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if table.isna().any().any():
        raise ValueError("Prediction table contains missing/non-numeric ages or predictions")
    if table.duplicated(["dataset", "sample_id", label_column]).any():
        raise ValueError("Prediction table contains duplicate dataset/sample/model rows")
    return table


def summarize_metrics(predictions: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Calculate n, MAE, Pearson r, and its two-sided p-value by model label."""
    rows: list[dict[str, float | int | str]] = []
    for (dataset, label), group in predictions.groupby(["dataset", label_column], sort=True):
        if len(group) < 3:
            raise ValueError(f"{dataset}/{label} has fewer than three held-out samples")
        correlation = pearsonr(group["age"], group["prediction"])
        if not np.isfinite(correlation.statistic):
            raise ValueError(f"{dataset}/{label} has a constant age or prediction vector")
        rows.append(
            {
                "dataset": dataset,
                label_column: label,
                "n": len(group),
                "mae": mean_absolute_error(group["age"], group["prediction"]),
                "pearson_r": correlation.statistic,
                "pearson_p": correlation.pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", label_column], ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse a single long prediction table and the label column it contains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--label-column",
        choices=("predictor", "model"),
        default="predictor",
        help="Column identifying component clocks ('predictor') or model architectures ('model').",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Read held-out predictions and write the corresponding long metrics table."""
    args = parse_args()
    predictions = validate_predictions(pd.read_csv(args.predictions), args.label_column)
    metrics = summarize_metrics(predictions, args.label_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
