#!/usr/bin/env python3
"""Quantify prediction differences between paired InCHIANTI DNAm replicates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_PREDICTIONS = Path(
    "results/figure4/duplicate_holdout/duplicate_sample_predictions.csv"
)  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/source_data/duplicate_sample_reliability_source_data.csv")
MODELS = ("bio_metaclock", "baseline")


def pair_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert two rows per participant/model into one paired-difference row."""
    required = {"ID", "sampleID", "model", "prediction"}
    if missing := required.difference(predictions.columns):
        raise KeyError(f"Prediction table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for (participant, model), pair in predictions.groupby(["ID", "model"], sort=True):
        pair = pair.sort_values("sampleID")
        if len(pair) != 2:
            raise ValueError(f"Expected two samples for ID={participant}, model={model}.")
        first, second = pair.iloc[0], pair.iloc[1]
        rows.append(
            {
                "ID": participant,
                "model": model,
                "prediction_1": float(first["prediction"]),
                "prediction_2": float(second["prediction"]),
                "absolute_delta": abs(float(first["prediction"] - second["prediction"])),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation and absolute-difference reliability measures."""
    rows: list[dict[str, object]] = []
    for model in MODELS:
        subset = pairs[pairs["model"].eq(model)]
        if len(subset) != 44:
            raise ValueError(f"Expected 44 duplicate pairs for {model}; found {len(subset)}.")
        delta = subset["absolute_delta"].to_numpy(float)
        rows.append(
            {
                "model": model,
                "n_pairs": len(subset),
                "pearson_r": float(
                    pearsonr(subset["prediction_1"], subset["prediction_2"]).statistic
                ),
                "mean_absolute_delta": float(np.mean(delta)),
                "median_absolute_delta": float(np.median(delta)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """Generate the panel-02 reliability source table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = summarize_pairs(pair_predictions(pd.read_csv(args.predictions)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
