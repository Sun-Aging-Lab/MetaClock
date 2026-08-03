#!/usr/bin/env python3
"""Fit standardized InCHIANTI Cox models and plot 95% intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


FIGURE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARTICIPANTS = FIGURE_DIR / "datasets/inchianti/cox_analysis_participants.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "results"
CONFIDENCE_LEVEL = 0.95

MODEL_PREDICTORS = {
    "bio_metaclock": "bio_metaclock",
    "baseline": "baseline",
    "clinical_phenoage": "clinical_phenoage",
}
ADJUSTMENT_SETS = {
    "prediction_plus_chronological_age_sex": ("age_years", "sex_male"),
    "prediction_plus_sex": ("sex_male",),
    "prediction_plus_chronological_age": ("age_years",),
}
PANEL_TITLES = {
    "prediction_plus_chronological_age_sex": "Age + sex",
    "prediction_plus_sex": "Sex",
    "prediction_plus_chronological_age": "Age",
}
MODEL_LABELS = {
    "bio_metaclock": "BioMeta",
    "baseline": "Baseline",
    "clinical_phenoage": "PhenoAge",
}
MODEL_COLORS = {
    "bio_metaclock": "#0072B2",
    "baseline": "#4D4D4D",
    "clinical_phenoage": "#CC79A7",
}
MM_TO_INCH = 1.0 / 25.4
FIGSIZE = (60 * MM_TO_INCH, 50 * MM_TO_INCH)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    """Raise a readable error when an input table is incomplete."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def fit_standardized_cox(
    participants: pd.DataFrame,
    predictor: str,
    adjustments: tuple[str, ...],
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Fit one pooled Cox model after population-SD standardization.

    CoxPHFitter uses a Breslow baseline, Efron handling of ties, no penalizer,
    non-robust model-based variance, and two-sided 95% intervals by default.
    """
    covariates = [*adjustments, predictor]
    raw = participants[covariates].apply(pd.to_numeric, errors="raise").to_numpy(float)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=0)
    if not np.isfinite(raw).all() or np.any(scales == 0):
        raise ValueError(f"Non-finite or constant Cox covariates: {covariates}")
    fit_frame = pd.DataFrame((raw - means) / scales, columns=covariates)
    fit_frame["_duration"] = participants["survival_time_years"].to_numpy(float)
    fit_frame["_event"] = participants["event"].to_numpy(int)
    model = CoxPHFitter(
        alpha=1.0 - confidence_level,
        baseline_estimation_method="breslow",
        penalizer=0.0,
        l1_ratio=0.0,
    )
    model.fit(
        fit_frame,
        duration_col="_duration",
        event_col="_event",
        robust=False,
        show_progress=False,
    )
    row = model.summary.loc[predictor]
    interval = model.confidence_intervals_.loc[predictor].to_numpy(float)
    return {
        "coefficient": float(model.params_.loc[predictor]),
        "standard_error": float(model.standard_errors_.loc[predictor]),
        "z_statistic": float(row["z"]),
        "p_value": float(row["p"]),
        "ci_lower": float(interval[0]),
        "ci_upper": float(interval[1]),
        "confidence_level": confidence_level,
        "n_participants": len(participants),
        "n_events": int(fit_frame["_event"].sum()),
    }


def fit_all_models(participants: pd.DataFrame) -> pd.DataFrame:
    """Fit the three scores under each manuscript adjustment set."""
    rows: list[dict[str, Any]] = []
    for panel, adjustments in ADJUSTMENT_SETS.items():
        for model, predictor in MODEL_PREDICTORS.items():
            rows.append(
                {
                    "panel": panel,
                    "model": model,
                    **fit_standardized_cox(participants, predictor, adjustments),
                }
            )
    return pd.DataFrame(rows)


def plot_cox(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot standardized coefficients and their two-sided 95% intervals."""
    xmin = float(summary["ci_lower"].min())
    xmax = float(summary["ci_upper"].max())
    padding = max(0.08 * (xmax - xmin), 0.05)
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    figure.subplots_adjust(left=0.30, right=0.98, bottom=0.28, top=0.84, wspace=0.28)
    for axis, panel in zip(axes, ADJUSTMENT_SETS, strict=True):
        values = summary.loc[summary["panel"].eq(panel)].set_index("model").loc[
            list(MODEL_PREDICTORS)
        ]
        y = np.arange(len(values))
        for y_value, (model, row) in zip(y, values.iterrows(), strict=True):
            point = float(row["coefficient"])
            axis.errorbar(
                point,
                y_value,
                xerr=np.array([[point - row["ci_lower"]], [row["ci_upper"] - point]]),
                fmt="o",
                color=MODEL_COLORS[model],
                capsize=1.5,
                markersize=2.8,
            )
        axis.axvline(0, color="#777777", linewidth=0.5, linestyle="--")
        axis.set_xlim(xmin - padding, xmax + padding)
        axis.set_title(PANEL_TITLES[panel])
        axis.set_yticks(y)
        axis.invert_yaxis()
    axes[0].set_yticklabels([MODEL_LABELS[model] for model in MODEL_PREDICTORS])
    figure.supxlabel(r"Standardized $\beta$ (95% CI)", y=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "04_cox_coefficients.pdf")
    figure.savefig(output_dir / "04_cox_coefficients.svg")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse the relative InCHIANTI path and optional replacement path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Fit panel 4 and write its complete coefficient table."""
    args = parse_args()
    participants = pd.read_csv(args.participants)
    required = {
        "participant_id",
        "survival_time_years",
        "event",
        "age_years",
        "sex_male",
        *MODEL_PREDICTORS.values(),
    }
    require_columns(participants, required, args.participants)
    if participants["participant_id"].duplicated().any():
        raise ValueError("Cox input must contain one row per participant")
    if not participants["event"].isin([0, 1]).all():
        raise ValueError("event must contain only 0 and 1")
    if not pd.to_numeric(participants["survival_time_years"]).gt(0).all():
        raise ValueError("survival_time_years must be positive")
    summary = fit_all_models(participants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "04_cox_standardized_coefficients.csv", index=False)
    plot_cox(summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
