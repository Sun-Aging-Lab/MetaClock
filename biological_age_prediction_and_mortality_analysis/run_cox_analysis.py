#!/usr/bin/env python3
"""Fit standardized Cox models from shared participant-level OOF predictions.

Run the biological-age prediction workflow first. The same primary-profile
OOF scores are used for performance, Cox and Kaplan–Meier analyses.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    INCHIANTI_COX_EVENTS,
    INCHIANTI_PARTICIPANTS,
    OOF_COMPONENTS,
)
from .data import (
    complete_case_mask as parse_complete_case_mask,
)
from .data import (
    read_analysis_dataset,
)
from .plotting import LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions
from .production_oof import (
    PRODUCTION_CONFIG_ID, load_primary_oof_predictions, production_oof_folds,
)

# --- Repository-local generated input, outputs, and cache ---

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "preprocessing"
    / "inchianti_analysis_ready.parquet"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cox"


# --- Fixed formal parameters ---

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
N_FOLDS = 5
SEED = 0
COX_PENALIZER = 0.0
TRAINED_MODELS = ("bio_metaclock", "baseline")
MODEL_ORDER = (*TRAINED_MODELS, "clinical_phenoage")
COX_PANELS = {
    "prediction_plus_chronological_age_sex": ["prediction_z", "age_z", "sex_z"],
    "prediction_plus_sex": ["prediction_z", "sex_z"],
    "prediction_plus_chronological_age": ["prediction_z", "age_z"],
}
REQUIRED_COLUMNS = {
    "ID",
    "sampleID",
    "age_years",
    TARGET,
    COMPLETE_CASE,
    "survival_time_years",
    "event",
    "sex_male",
}


# --- Cohort selection: clinical complete cases, then one baseline sample per ID ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one analysis-ready CSV/CSV.GZ or Parquet cohort."""
    return read_analysis_dataset(
        path, columns=sorted(REQUIRED_COLUMNS), float_precision="round_trip"
    )


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the explicit clinical PhenoAge complete-case flag."""
    return parse_complete_case_mask(values)


def prepare_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep complete cases and deterministically select one DNAm sample per participant."""
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"Cox dataset is missing columns: {sorted(missing)}")
    cohort = frame.loc[complete_case_mask(frame[COMPLETE_CASE])].copy()
    cohort["ID"] = cohort["ID"].astype(str)
    cohort["sampleID"] = cohort["sampleID"].astype(str)
    numeric = ["age_years", TARGET, "survival_time_years", "event", "sex_male"]
    for column in numeric:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    cohort = cohort.dropna(subset=["ID", "sampleID", *numeric])
    cohort = cohort[cohort["survival_time_years"].gt(0) & cohort["event"].isin([0, 1])]
    suffix = pd.to_numeric(cohort["sampleID"].str.extract(r"(\d+)$", expand=False), errors="coerce")
    cohort["_suffix"] = suffix.fillna(np.inf)
    cohort["_numeric_id"] = pd.to_numeric(cohort["ID"], errors="raise")
    cohort = (
        cohort.sort_values(["_numeric_id", "_suffix", "sampleID"], kind="mergesort")
        .drop_duplicates("ID", keep="first")
        .drop(columns=["_numeric_id", "_suffix"])
        .reset_index(drop=True)
    )
    if len(cohort) < N_FOLDS or cohort["event"].sum() < 1:
        raise ValueError("Cox analysis needs at least five participants and one event.")
    return cohort


def load_survival_predictions(
    cohort: pd.DataFrame, *, output_dir: Path | None = None
) -> pd.DataFrame:
    """Load the shared OOF scores and add the observed Clinical PhenoAge comparator."""
    predictions = load_primary_oof_predictions(cohort, output_dir=output_dir)
    predictions = predictions.rename(columns={"pred_bioage": "prediction"})
    clinical = cohort.copy()
    clinical["fold"] = production_oof_folds(cohort)
    clinical["model"] = "clinical_phenoage"
    clinical["prediction"] = clinical[TARGET]
    return pd.concat([predictions, clinical], ignore_index=True)


# --- Standardized pooled Cox models ---


def fit_cox_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fit the exact standardized pooled Cox specifications used by Figure 4."""
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("Install lifelines to fit the Cox models.") from exc
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        data = predictions[predictions["model"].eq(model)].copy()
        if len(data) != INCHIANTI_PARTICIPANTS or data["ID"].nunique() != INCHIANTI_PARTICIPANTS:
            raise ValueError(
                f"{model} must contain {INCHIANTI_PARTICIPANTS} unique participants."
            )
        raw = data[["prediction", "age_years", "sex_male"]]
        if (raw.std(ddof=0) == 0).any():
            raise ValueError("Cox covariate has zero variance.")
        data[["prediction_z", "age_z", "sex_z"]] = StandardScaler().fit_transform(raw)
        for panel, covariates in COX_PANELS.items():
            fit_data = data[["survival_time_years", "event", *covariates]].dropna()
            fitter = CoxPHFitter(
                alpha=0.05,
                baseline_estimation_method="breslow",
                penalizer=COX_PENALIZER,
            )
            fitter.fit(
                fit_data,
                duration_col="survival_time_years",
                event_col="event",
                robust=False,
                show_progress=False,
            )
            estimate = fitter.summary.loc["prediction_z"]
            coefficient = float(estimate["coef"])
            ci_lower = float(estimate["coef lower 95%"])
            ci_upper = float(estimate["coef upper 95%"])
            rows.append(
                {
                    "panel": panel,
                    "model": model,
                    "n": len(fit_data),
                    "events": int(fit_data["event"].sum()),
                    "coefficient": coefficient,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": float(estimate["p"]),
                    "hazard_ratio": float(estimate["exp(coef)"]),
                    "hr_lower_95": float(estimate["exp(coef) lower 95%"]),
                    "hr_upper_95": float(estimate["exp(coef) upper 95%"]),
                }
            )
    return pd.DataFrame(rows)


# --- Forest plot ---


def plot_cox(coefficients: pd.DataFrame) -> None:
    """Draw the manuscript forest panels with three-significant-digit scientific labels."""
    configure_style()
    figure, axes = plt.subplots(1, len(COX_PANELS), figsize=(90 / 25.4, 50 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.30, right=0.98, bottom=0.28, top=0.84, wspace=0.28)
    titles = {
        "prediction_plus_chronological_age_sex": "Age + sex",
        "prediction_plus_sex": "Sex",
        "prediction_plus_chronological_age": "Age",
    }
    colors = {"bio_metaclock": "#BFDB87", "baseline": "#4D4D4D", "clinical_phenoage": "#CC79A7"}
    labels = {**LABELS, "bio_metaclock": "MetaClock-\nbiological", "clinical_phenoage": "Biological Age"}
    y = np.arange(len(MODEL_ORDER))
    for index, (axis, panel) in enumerate(zip(axes, COX_PANELS, strict=True)):
        table = (
            coefficients[coefficients["panel"].eq(panel)].set_index("model").loc[list(MODEL_ORDER)]
        )
        coefficient = table["coefficient"].to_numpy(float)
        lower = table["ci_lower"].to_numpy(float)
        upper = table["ci_upper"].to_numpy(float)
        for row_index, model in enumerate(MODEL_ORDER):
            axis.errorbar(
                coefficient[row_index],
                y[row_index],
                xerr=np.array(
                    [[coefficient[row_index] - lower[row_index]], [upper[row_index] - coefficient[row_index]]]
                ),
                fmt="o",
                color=colors[model],
                ecolor=colors[model],
                markersize=2.8,
                linewidth=0.8,
                capsize=1.5,
                markeredgecolor="#222222",
                markeredgewidth=0.2,
            )
            annotation_y = y[row_index] + (0.24 if row_index < 2 else -0.24)
            axis.text(
                0.25,
                annotation_y,
                f"β={coefficient[row_index]:.2e}",
                ha="left",
                va="center",
                fontsize=6,
            )
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.5)
        axis.set(xlim=(-0.29, 2.23), xticks=[0, 2], yticks=y, title=titles[panel])
        axis.tick_params(axis="x", labelsize=8)
        if index == 0:
            axis.set_yticklabels([labels[model] for model in MODEL_ORDER], fontsize=8)
        clean_axis(axis, grid=False)
    axes[0].invert_yaxis()
    figure.supxlabel("Log Hazard Ratio", y=0.07, fontsize=9.75)
    save_vector_pair(figure, OUTPUT_DIR, "04_cox_coefficients_standardized")


def main() -> int:
    """Load shared primary OOF scores, fit Cox specifications, and plot."""
    cohort = prepare_cohort(read_dataset(DATASET_PATH))
    if len(cohort) != INCHIANTI_PARTICIPANTS or int(cohort["event"].sum()) != INCHIANTI_COX_EVENTS:
        raise ValueError(
            f"Figure 4 Cox requires {INCHIANTI_PARTICIPANTS} participants and "
            f"{INCHIANTI_COX_EVENTS} deaths."
        )
    predictions = load_survival_predictions(cohort)
    coefficients = fit_cox_models(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "cox_oof_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "cox_coefficients_standardized.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "04_cox_coefficients_standardized_source_data.csv", index=False)
    source_metadata = json.dumps(
        {
            "prediction_config_id": PRODUCTION_CONFIG_ID,
            "prediction_source": "shared biological_age_prediction/oof_predictions.csv",
            "outer_folds": "primary participants; age-decade by sex; StratifiedKFold(5, shuffle=True, random_state=0)",
            "prediction_model": "19-component MetaClock and full-CpG baseline; shared fold-local OOF predictions",
            "cox_scope": "pooled OOF predictions; observed clinical PhenoAge comparator",
            "cox_fit": (
                "exact Figure 4 standardized specification: every covariate z-scored with "
                "population SD (ddof=0); CoxPHFitter alpha=0.05, Breslow baseline, "
                "penalizer=0.0; robust=False; show_progress=False"
            ),
            "models": list(MODEL_ORDER),
            "bio_metaclock_components": list(OOF_COMPONENTS),
            "panels": COX_PANELS,
            "software": software_versions(
                ["numpy", "pandas", "scikit-learn", "lifelines", "pyaging", "anndata"]
            ),
        },
        indent=2,
    ) + "\n"
    for filename in ("run_summary.json", "figure_source.json"):
        (OUTPUT_DIR / filename).write_text(source_metadata, encoding="utf-8")
    plot_cox(coefficients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
