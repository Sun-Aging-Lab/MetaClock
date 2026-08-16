#!/usr/bin/env python3
"""Fit standardized Cox models and age-corrected Kaplan-Meier analyses."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import logrank_test

try:
    from .config import FIGURE4, SURVIVAL_MODELS
    from .provenance import file_record, software_versions, write_json
except ImportError:  # Direct-script fallback.
    from config import FIGURE4, SURVIVAL_MODELS
    from provenance import file_record, software_versions, write_json


DEFAULT_COHORT = Path(
    "results/figure4/mortality/inchianti_oof_predictions_with_survival.csv"
)  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/source_data")
MODELS = SURVIVAL_MODELS
COX_PANELS = {
    "prediction_plus_chronological_age_sex": ["prediction_z", "age_z", "sex"],
    "prediction_plus_sex": ["prediction_z", "sex"],
    "prediction_plus_chronological_age": ["prediction_z", "age_z"],
}


def zscore(values: pd.Series) -> pd.Series:
    """Standardize a numeric covariate with sample standard deviation."""
    numeric = pd.to_numeric(values, errors="raise")
    return (numeric - numeric.mean()) / numeric.std(ddof=1)


def fit_cox_models(cohort: pd.DataFrame) -> pd.DataFrame:
    """Fit the three displayed Cox specifications for each age model."""
    rows: list[dict[str, object]] = []
    for model in MODELS:
        data = cohort[cohort["model"].eq(model)].copy()
        if len(data) != int(FIGURE4["participants"]) or data["ID"].nunique() != int(
            FIGURE4["participants"]
        ):
            raise ValueError(f"{model} must contain 672 unique participants.")
        for panel, covariates in COX_PANELS.items():
            raw_covariates = [
                "prediction" if value == "prediction_z" else "age" if value == "age_z" else value
                for value in covariates
            ]
            fit_data = (
                data[["survival_time", "event", *raw_covariates]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            fit_data["prediction_z"] = zscore(fit_data["prediction"])
            if "age_z" in covariates:
                fit_data["age_z"] = zscore(fit_data["age"])
            if any(fit_data[column].nunique() < 2 for column in covariates):
                raise ValueError(f"{model}/{panel} contains a constant Cox covariate.")
            model_data = fit_data[["survival_time", "event", *covariates]]
            fit = CoxPHFitter(alpha=0.05, baseline_estimation_method="breslow", penalizer=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fit.fit(
                    model_data,
                    duration_col="survival_time",
                    event_col="event",
                    fit_options={"max_steps": 1000},
                )
            estimate = fit.summary.loc["prediction_z"]
            rows.append(
                {
                    "panel": panel,
                    "model": model,
                    "n": len(fit_data),
                    "events": int(fit_data["event"].sum()),
                    "coefficient": float(estimate["coef"]),
                    "ci_lower": float(estimate["coef lower 95%"]),
                    "ci_upper": float(estimate["coef upper 95%"]),
                    "p_value": float(estimate["p"]),
                    "hazard_ratio": float(estimate["exp(coef)"]),
                    "hr_lower_95": float(estimate["exp(coef) lower 95%"]),
                    "hr_upper_95": float(estimate["exp(coef) upper 95%"]),
                }
            )
    return pd.DataFrame(rows)


def age_correct_predictions(cohort: pd.DataFrame) -> pd.DataFrame:
    """Remove the within-model linear relationship between age and prediction."""
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        data = cohort[cohort["model"].eq(model)].copy()
        if len(data) != int(FIGURE4["participants"]):
            raise ValueError(f"{model} must contain 672 participants before age correction.")
        data = data.copy()
        age = pd.to_numeric(data["age"], errors="raise").to_numpy(float)
        prediction = pd.to_numeric(data["prediction"], errors="raise").to_numpy(float)
        if not np.isfinite(age).all() or not np.isfinite(prediction).all():
            raise ValueError(f"{model} contains non-finite age-correction inputs.")
        design = np.column_stack([np.ones(len(data)), age])
        coefficients, *_ = np.linalg.lstsq(design, prediction, rcond=None)
        data["age_corrected_prediction"] = prediction - design @ coefficients
        data["model"] = model
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def km_curve(group: pd.DataFrame) -> pd.DataFrame:
    """Return the formal event-time KM schema including risk and event counts."""
    fitter = KaplanMeierFitter(alpha=0.05, label="survival").fit(
        group["survival_time"], event_observed=group["event"]
    )
    event_times = fitter.event_table.index[fitter.event_table["observed"].gt(0)].to_numpy(float)
    timeline = np.unique(np.concatenate(([0.0], event_times)))
    survival = fitter.survival_function_.iloc[:, 0].reindex(timeline, method="ffill")
    confidence = fitter.confidence_interval_.reindex(timeline, method="ffill")
    event_table = fitter.event_table.reindex(timeline).fillna(0)
    return pd.DataFrame(
        {
            "time": timeline,
            "survival": survival.to_numpy(float),
            "ci_lower": confidence.iloc[:, 0].to_numpy(float),
            "ci_upper": confidence.iloc[:, 1].to_numpy(float),
            "n_at_risk": event_table["at_risk"].to_numpy(int),
            "n_events": event_table["observed"].to_numpy(int),
        }
    )


def logrank_audit(low: pd.DataFrame, high: pd.DataFrame) -> dict[str, float]:
    """Return lifelines log-rank statistics plus transparent expected-event terms."""
    time = np.concatenate(
        [low["survival_time"].to_numpy(float), high["survival_time"].to_numpy(float)]
    )
    event = np.concatenate([low["event"].to_numpy(bool), high["event"].to_numpy(bool)])
    top = np.concatenate([np.zeros(len(low), dtype=bool), np.ones(len(high), dtype=bool)])
    observed_top = expected_top = variance = 0.0
    for event_time in np.sort(np.unique(time[event])):
        at_risk = time >= event_time
        events = event & (time == event_time)
        n_total, d_total = int(at_risk.sum()), int(events.sum())
        if n_total <= 1 or d_total == 0:
            continue
        n_top, d_top = int((at_risk & top).sum()), int((events & top).sum())
        observed_top += d_top
        expected_top += d_total * n_top / n_total
        variance += (
            n_top
            * (n_total - n_top)
            * d_total
            * (n_total - d_total)
            / (n_total * n_total * (n_total - 1))
        )
    test = logrank_test(
        low["survival_time"],
        high["survival_time"],
        event_observed_A=low["event"],
        event_observed_B=high["event"],
    )
    return {
        "observed_top_events": observed_top,
        "expected_top_events": expected_top,
        "logrank_variance": variance,
        "logrank_chi2": float(test.test_statistic),
        "logrank_p_value": float(test.p_value),
    }


def fit_tail_survival(
    cohort: pd.DataFrame, tail_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit KM curves and log-rank tests for the lowest/highest 20% residuals."""
    curve_rows: list[pd.DataFrame] = []
    test_rows: list[dict[str, object]] = []
    for model in MODELS:
        data = cohort[cohort["model"].eq(model)].sort_values(
            ["age_corrected_prediction", "sampleID"], kind="mergesort"
        )
        n_tail = int(np.floor(len(data) * tail_fraction))
        groups = {"bottom_20": data.head(n_tail), "top_20": data.tail(n_tail)}
        for label, group in groups.items():
            curve = km_curve(group)
            curve.insert(0, "bioage_group", label)
            curve.insert(0, "model", model)
            curve.insert(0, "grouping", "age_corrected")
            curve["n_group"] = len(group)
            curve["n_events_group"] = int(group["event"].sum())
            curve_rows.append(curve)
        low, high = groups["bottom_20"], groups["top_20"]
        test_rows.append(
            {
                "grouping": "age_corrected",
                "model": model,
                "bottom_20_n": len(low),
                "bottom_20_events": int(low["event"].sum()),
                "top_20_n": len(high),
                "top_20_events": int(high["event"].sum()),
                **logrank_audit(low, high),
            }
        )
    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(test_rows)


def main() -> int:
    """Write the standardized Cox and Kaplan-Meier source tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort)
    required = {"ID", "sampleID", "model", "prediction", "age", "sex", "survival_time", "event"}
    if missing := required.difference(cohort.columns):
        raise KeyError(f"Cohort table is missing columns: {sorted(missing)}")
    expected_n = int(FIGURE4["participants"])
    if set(cohort["model"].astype(str)) != set(MODELS):
        raise ValueError("Mortality cohort does not contain exactly the manifest models.")
    reference: set[tuple[object, object]] | None = None
    for model in MODELS:
        subset = cohort[cohort["model"].eq(model)]
        if len(subset) != expected_n or subset.duplicated(["ID", "sampleID"]).any():
            raise ValueError(f"{model} does not contain 672 unique participant-sample rows.")
        keys = set(subset[["ID", "sampleID"]].itertuples(index=False, name=None))
        if reference is not None and keys != reference:
            raise ValueError("Mortality models are not aligned to identical participants.")
        reference = keys
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_cox_models(cohort).to_csv(
        args.output_dir / "standardized_cox_effects_source_data.csv", index=False
    )
    curves, tests = fit_tail_survival(age_correct_predictions(cohort))
    curves.to_csv(args.output_dir / "kaplan_meier_curves_source_data.csv", index=False)
    tests.to_csv(args.output_dir / "kaplan_meier_logrank_source_data.csv", index=False)
    write_json(
        args.output_dir / "mortality_analysis_summary.json",
        {
            "input": file_record(args.cohort),
            "cox": "lifelines CoxPHFitter, Breslow baseline, no penalizer, complete-case scaling",
            "km_tail_rule": "floor(n * 0.20) after deterministic score/sampleID sorting",
            "models": list(MODELS),
            "software": software_versions(("lifelines", "numpy", "pandas")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
