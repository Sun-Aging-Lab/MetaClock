#!/usr/bin/env python3
"""Summarize participant-level five-fold out-of-fold biological-age predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

DEFAULT_PREDICTIONS = Path(
    "results/figure4/oof_predictions/inchianti_5fold_oof_predictions.csv"
)  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/source_data/bioage_prediction_metrics_source_data.csv")
MODELS = ("bio_metaclock", "baseline")


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return MAE and Pearson correlation for the two displayed models."""
    required = {"ID", "model", "clinical_phenoage", "pred_bioage"}
    if missing := required.difference(predictions.columns):
        raise KeyError(f"Prediction table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for model in MODELS:
        subset = predictions[predictions["model"].eq(model)].copy()
        if len(subset) != 672 or subset["ID"].nunique() != 672:
            raise ValueError(f"{model} must have one OOF prediction for each of 672 participants.")
        truth = pd.to_numeric(subset["clinical_phenoage"], errors="raise").to_numpy(float)
        pred = pd.to_numeric(subset["pred_bioage"], errors="raise").to_numpy(float)
        rows.append(
            {
                "model": model,
                "n": len(subset),
                "mae": float(mean_absolute_error(truth, pred)),
                "pearson_r": float(pearsonr(truth, pred).statistic),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """Read OOF predictions and write the biological-age metric source table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = summarize_predictions(pd.read_csv(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
