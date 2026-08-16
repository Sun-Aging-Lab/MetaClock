#!/usr/bin/env python3
"""Join OOF predictions to normalized survival covariates for Cox and KM analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .config import FIGURE4, SURVIVAL_MODELS
    from .provenance import file_record, write_json
except ImportError:  # Direct-script fallback.
    from config import FIGURE4, SURVIVAL_MODELS
    from provenance import file_record, write_json


DEFAULT_PREDICTIONS = Path(
    "results/figure4/oof_predictions/inchianti_5fold_oof_predictions.csv"
)  # fill in your path
DEFAULT_SURVIVAL = Path("data/inchianti/survival_covariates.csv")  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/mortality/inchianti_oof_predictions_with_survival.csv")


def main() -> int:
    """Create one complete row per participant and displayed survival model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--survival", type=Path, default=DEFAULT_SURVIVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    survival = pd.read_csv(args.survival)
    survival_required = {
        "ID",
        "sampleID",
        "age",
        "sex",
        "survival_time",
        "event",
        "clinical_phenoage",
    }
    if missing := survival_required.difference(survival.columns):
        raise KeyError(f"Survival covariates are missing columns: {sorted(missing)}")
    expected_n = int(FIGURE4["participants"])
    if len(survival) != expected_n or survival["ID"].nunique() != expected_n:
        raise ValueError("Survival table must contain exactly 672 participants.")
    direct = predictions[predictions["model"].isin(SURVIVAL_MODELS[:2])].rename(
        columns={"pred_bioage": "prediction"}
    )
    direct = direct[["ID", "sampleID", "model", "prediction"]]
    clinical = survival[["ID", "sampleID", "clinical_phenoage"]].rename(
        columns={"clinical_phenoage": "prediction"}
    )
    clinical["model"] = "clinical_phenoage"
    long = pd.concat([direct, clinical], ignore_index=True)
    cohort = long.merge(
        survival[["ID", "sampleID", "age", "sex", "survival_time", "event"]],
        on=["ID", "sampleID"],
        how="left",
        validate="many_to_one",
    )
    expected_rows = expected_n * len(SURVIVAL_MODELS)
    if len(cohort) != expected_rows or cohort.duplicated(["model", "ID"]).any():
        raise ValueError("Mortality table does not contain one row per model/participant.")
    if cohort.isna().any().any() or tuple(cohort["model"].drop_duplicates()) != SURVIVAL_MODELS:
        raise ValueError("Mortality table is incomplete or model order is invalid.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    write_json(
        args.output.with_suffix(".provenance.json"),
        {
            "inputs": {
                "predictions": file_record(args.predictions),
                "survival": file_record(args.survival),
            },
            "output": file_record(args.output),
            "models": list(SURVIVAL_MODELS),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
