#!/usr/bin/env python3
"""Estimate intervention effects from donor-level biological-age predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "results/intervention/sample_predictions.csv"
)  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/intervention/effects"
MIN_DONORS_PER_GROUP = 2


@dataclass(frozen=True)
class EffectEstimate:
    """A donor-level group contrast and its documented inferential design."""

    effect: float
    p_value: float
    test: str
    control_donors: int
    experiment_donors: int


def validate_constant_comparison_metadata(frame: pd.DataFrame) -> None:
    """Require each comparison ID to map to one GEO ID and one display name."""
    for column in ("gse_id", "comparison_name"):
        counts = frame.groupby("comparison_id", dropna=False)[column].nunique(dropna=False)
        invalid = counts.loc[counts.ne(1)]
        if not invalid.empty:
            raise ValueError(
                f"comparison_id must have one {column}; invalid IDs: {invalid.index.tolist()}"
            )


def unique_samples_for_age_correction(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate repeated comparison membership only after checking value conflicts."""
    value_columns = ["predicted_age", "correction_stratum"]
    if "chronological_age" in frame:
        value_columns.append("chronological_age")
    conflicts = (
        frame.groupby(["gse_id", "sample_id"], dropna=False)[value_columns]
        .nunique(dropna=False)
        .max(axis=1)
    )
    invalid = conflicts.loc[conflicts.gt(1)]
    if not invalid.empty:
        raise ValueError(
            "Repeated gse_id/sample_id rows disagree on predicted or chronological age: "
            f"{invalid.index.tolist()}"
        )
    return frame.drop_duplicates(["gse_id", "sample_id"], keep="first")


def age_correct(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove a complete chronological-age trend or mark a no-age stratum."""
    output = frame.copy()
    output["analysis_predicted_age"] = pd.to_numeric(output["predicted_age"], errors="raise")
    if "chronological_age" not in output or output["chronological_age"].isna().all():
        output["age_adjustment"] = "none_no_chronological_age"
        return output
    ages = pd.to_numeric(output["chronological_age"], errors="coerce")
    if ages.isna().any():
        raise ValueError("Chronological age must be complete within an analyzed stratum")
    fit = LinearRegression().fit(
        ages.to_numpy(float).reshape(-1, 1),
        output["analysis_predicted_age"].to_numpy(float),
    )
    output["analysis_predicted_age"] -= fit.predict(ages.to_numpy(float).reshape(-1, 1))
    output["age_adjustment"] = "linear_within_dataset_and_stratum"
    return output


def donor_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated samples before testing at the donor level."""
    return frame.groupby(["donor_id", "group"], as_index=False, dropna=False).agg(
        analysis_predicted_age=("analysis_predicted_age", "mean")
    )


def classify_donor_design(donors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Classify a design without discarding unmatched donors from a mixed comparison."""
    control = donors.loc[donors["group"].eq("control")].set_index("donor_id")
    experiment = donors.loc[donors["group"].eq("experiment")].set_index("donor_id")
    if len(control) < MIN_DONORS_PER_GROUP or len(experiment) < MIN_DONORS_PER_GROUP:
        raise ValueError(
            f"Each comparison needs at least {MIN_DONORS_PER_GROUP} control and experiment donors"
        )
    if control.index.equals(experiment.index):
        return control, experiment, "fully_paired"
    if control.index.intersection(experiment.index).empty:
        return control, experiment, "fully_unpaired"
    raise ValueError(
        "Partially overlapping donor groups are ambiguous: refusing to drop unmatched donors"
    )


def estimate_effect(
    frame: pd.DataFrame,
    *,
    require_fully_paired: bool,
) -> EffectEstimate:
    """Estimate paired or unpaired effects while retaining every valid donor."""
    control, experiment, design = classify_donor_design(donor_means(frame))
    if require_fully_paired and design != "fully_paired":
        raise ValueError(
            "Comparisons without chronological age must be fully paired with both groups present"
        )
    control_values = control["analysis_predicted_age"].to_numpy(float)
    experiment_values = experiment["analysis_predicted_age"].to_numpy(float)
    if design == "fully_paired":
        p_value = float(ttest_rel(experiment_values, control_values).pvalue)
        effect = float(np.mean(experiment_values - control_values))
        test = "two-sided paired t-test"
    else:
        p_value = float(ttest_ind(experiment_values, control_values, equal_var=False).pvalue)
        effect = float(experiment_values.mean() - control_values.mean())
        test = "two-sided Welch t-test"
    if not np.isfinite(p_value):
        raise ValueError("Intervention test returned a non-finite p-value")
    return EffectEstimate(
        effect=effect,
        p_value=p_value,
        test=test,
        control_donors=int(len(control)),
        experiment_donors=int(len(experiment)),
    )


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct age, test validated comparisons, and apply BH to completed tests."""
    required = {
        "comparison_id",
        "gse_id",
        "comparison_name",
        "sample_id",
        "donor_id",
        "group",
        "predicted_age",
    }
    if missing := required.difference(frame.columns):
        raise KeyError(f"Prediction table is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required comparison metadata and predictions must not be missing")
    if not set(frame["group"].astype(str)).issubset({"control", "experiment"}):
        raise ValueError("group must contain only control and experiment")
    validate_constant_comparison_metadata(frame)
    if "correction_stratum" not in frame:
        frame = frame.assign(correction_stratum="all")
    unique_samples = unique_samples_for_age_correction(frame)
    corrected = pd.concat(
        [
            age_correct(part)
            for _, part in unique_samples.groupby(
                ["gse_id", "correction_stratum"], sort=False, dropna=False
            )
        ],
        ignore_index=True,
    )
    correction_columns = [
        "gse_id",
        "sample_id",
        "analysis_predicted_age",
        "age_adjustment",
    ]
    merged = frame.merge(
        corrected[correction_columns],
        on=["gse_id", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["analysis_predicted_age"].isna().any():
        raise RuntimeError("Age correction failed to return every intervention sample")
    rows: list[dict[str, object]] = []
    for comparison_id, comparison in merged.groupby("comparison_id", sort=False):
        if comparison["age_adjustment"].nunique(dropna=False) != 1:
            raise ValueError(
                f"{comparison_id} mixes age-adjusted and no-age samples; split the design"
            )
        no_age = comparison["age_adjustment"].eq("none_no_chronological_age").all()
        estimate = estimate_effect(comparison, require_fully_paired=no_age)
        first = comparison.iloc[0]
        rows.append(
            {
                "comparison_id": comparison_id,
                "gse_id": first["gse_id"],
                "comparison_name": first["comparison_name"],
                "mean_diff_experiment_minus_control": estimate.effect,
                "two_sided_p_value": estimate.p_value,
                "test": estimate.test,
                "control_n_donors": estimate.control_donors,
                "experiment_n_donors": estimate.experiment_donors,
                "age_adjustment": (
                    "none_fully_paired_no_chronological_age"
                    if no_age
                    else "linear_within_dataset_and_stratum"
                ),
            }
        )
    results = pd.DataFrame(rows)
    results["bh_q_value_all_completed"] = multipletests(
        results["two_sided_p_value"].to_numpy(float), method="fdr_bh"
    )[1]
    results = results.sort_values(
        ["two_sided_p_value", "mean_diff_experiment_minus_control"],
        ascending=[True, False],
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return results, merged


def parse_args() -> argparse.Namespace:
    """Parse optional project-relative or absolute input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Write intervention effects and their age-corrected sample-level inputs."""
    args = parse_args()
    results, corrected = summarize(pd.read_csv(args.predictions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "all_intervention_effects.csv", index=False)
    corrected.to_csv(args.output_dir / "age_corrected_sample_predictions.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
