#!/usr/bin/env python3
"""Test within-dataset age-adjusted disease-control prediction differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

try:
    from .config import EXTERNAL_MODELS, FIGURE4
    from .provenance import file_record, software_versions, write_json
except ImportError:  # Direct-script fallback.
    from config import EXTERNAL_MODELS, FIGURE4
    from provenance import file_record, software_versions, write_json


DEFAULT_PREDICTIONS = Path(
    "results/figure4/external_predictions/external_disease_predictions.csv"
)  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/external_disease")
KEYS = ["dataset", "disease", "sample_id", "age", "group"]


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Require identical complete samples for all three displayed models."""
    required = {*KEYS, "model", "pred_age"}
    if missing := required.difference(predictions.columns):
        raise KeyError(f"External predictions are missing columns: {sorted(missing)}")
    expected_n = int(FIGURE4["external_samples"])
    if set(predictions["model"].astype(str)) != set(EXTERNAL_MODELS):
        raise ValueError("External predictions do not contain exactly the three manifest models.")
    reference: set[tuple[object, ...]] | None = None
    for model in EXTERNAL_MODELS:
        frame = predictions[predictions["model"].eq(model)]
        if len(frame) != expected_n or frame.duplicated(KEYS).any():
            raise ValueError(f"{model} must contain one prediction for each of 3,325 samples.")
        keys = set(frame[KEYS].itertuples(index=False, name=None))
        if reference is not None and keys != reference:
            raise ValueError("External model sample keys are not identical.")
        reference = keys


def age_correct_and_test(
    predictions: pd.DataFrame,
    *,
    expected_datasets: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regress predictions on age within dataset/model and run Welch tests."""
    predictions = predictions.copy()
    predictions["difference"] = np.nan
    fit_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for (dataset, disease, model), subset in predictions.groupby(
        ["dataset", "disease", "model"], sort=False
    ):
        age = pd.to_numeric(subset["age"], errors="raise").to_numpy(float)
        pred = pd.to_numeric(subset["pred_age"], errors="raise").to_numpy(float)
        groups = subset["group"].astype(str)
        if not np.isfinite(age).all() or not np.isfinite(pred).all():
            raise ValueError(f"Non-finite age/prediction in {dataset}, {model}.")
        if set(groups) != {"control", "disease"}:
            raise ValueError(f"{dataset}, {model} must contain both control and disease samples.")
        fit = LinearRegression().fit(age[:, None], pred)
        difference = pred - fit.predict(age[:, None])
        predictions.loc[subset.index, "difference"] = difference
        fit_rows.append(
            {
                "dataset": dataset,
                "disease": disease,
                "model": model,
                "age_coef": float(fit.coef_[0]),
                "intercept": float(fit.intercept_),
                "r2_age_explained": float(fit.score(age[:, None], pred)),
                "n_fit": len(subset),
            }
        )
        control = difference[groups.to_numpy() == "control"]
        cases = difference[groups.to_numpy() == "disease"]
        if min(len(control), len(cases)) < 2:
            raise ValueError(
                f"{dataset}, {model} has fewer than two samples in a comparison group."
            )
        test_rows.append(
            {
                "dataset": dataset,
                "disease": disease,
                "model": model,
                "control_n": len(control),
                "disease_n": len(cases),
                "mean_difference_disease_minus_control": float(cases.mean() - control.mean()),
                "welch_ttest_p_value": float(
                    ttest_ind(cases, control, equal_var=False, nan_policy="raise").pvalue
                ),
            }
        )
    tests = pd.DataFrame(test_rows)
    expected = int(FIGURE4["external_datasets"]) if expected_datasets is None else expected_datasets
    if len(tests) != expected * len(EXTERNAL_MODELS):
        raise ValueError(f"Expected {expected} datasets evaluated for each displayed model.")
    tests["bonferroni_p_value"] = (tests["welch_ttest_p_value"] * len(EXTERNAL_MODELS)).clip(
        upper=1.0
    )
    return predictions, pd.DataFrame(fit_rows), tests


def main() -> int:
    """Write corrected predictions, age fits, tests, and reproducibility metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_table = pd.read_csv(args.predictions)
    validate_predictions(input_table)
    predictions, age_fits, tests = age_correct_and_test(input_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "predictions": args.output_dir / "external_disease_age_corrected_predictions.csv",
        "age_fits": args.output_dir / "external_disease_age_fits.csv",
        "tests": args.output_dir / "external_disease_tests.csv",
    }
    predictions.to_csv(output_paths["predictions"], index=False)
    age_fits.to_csv(output_paths["age_fits"], index=False)
    tests.to_csv(output_paths["tests"], index=False)
    write_json(
        args.output_dir / "external_disease_analysis_summary.json",
        {
            "input": file_record(args.predictions),
            "outputs": {name: file_record(path) for name, path in output_paths.items()},
            "age_correction": "linear regression within dataset/model using pooled control+disease",
            "test": "two-sided Welch t-test",
            "multiple_testing": "Bonferroni across the three displayed models within dataset",
            "software": software_versions(("numpy", "pandas", "scikit-learn", "scipy")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
