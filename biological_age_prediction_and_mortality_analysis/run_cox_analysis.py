#!/usr/bin/env python3
"""Train five-fold OOF age models, fit standardized Cox models, and plot.

Run ``prepare_inchianti_dataset`` first; ``DATASET_PATH`` points to its output.
The row-oriented table must contain the biological age columns/CpGs plus
``survival_time_years``, ``event``, and ``sex_male``.
All Cox estimates use pooled participant-level OOF predictions; the observed
clinical PhenoAge comparator is never trained on its own outcome.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from repository_paths import relative_path

from .config import (
    INCHIANTI_COX_EVENTS,
    INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH,
    INCHIANTI_PARTICIPANTS,
    OOF_COMPONENTS,
)
from .data import (
    complete_case_mask as parse_complete_case_mask,
)
from .data import (
    dnam_feature_columns,
    fixed_figure4_oof_folds,
    numeric_matrix,
    read_analysis_dataset,
)
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions
from .run_biological_age_prediction import fit_oof_elasticnet

# --- Repository-local generated input, outputs, and cache ---

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "preprocessing"
    / "inchianti_analysis_ready.parquet"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cox"
PYAGING_CACHE_DIR = Path(__file__).resolve().parent / "pyaging_data"


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
    return read_analysis_dataset(path)


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
    cohort = (
        cohort.sort_values(["ID", "_suffix", "sampleID"], kind="mergesort")
        .drop_duplicates("ID", keep="first")
        .drop(columns="_suffix")
        .reset_index(drop=True)
    )
    if len(cohort) < N_FOLDS or cohort["event"].sum() < 1:
        raise ValueError("Cox analysis needs at least five participants and one event.")
    return cohort


def cpg_columns(frame: pd.DataFrame) -> list[str]:
    """Return all 485,577 formal baseline features in file order."""
    return dnam_feature_columns(frame)


# --- Formal participant-level age-stratified outer folds ---


def assign_cox_folds(cohort: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Use participant age quintiles with shuffled StratifiedKFold."""
    strata = pd.qcut(cohort["age_years"], q=N_FOLDS, labels=False, duplicates="drop")
    counts = strata.value_counts()
    if len(counts) < 2 or int(counts.min()) < N_FOLDS:
        raise ValueError(
            f"Each Cox age stratum must contain at least {N_FOLDS} IDs: {counts.to_dict()}"
        )
    folds = np.full(len(cohort), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    positions = np.arange(len(cohort))
    for fold, (_, test) in enumerate(splitter.split(positions, strata.to_numpy(int))):
        folds[test] = fold
    if (folds < 0).any():
        raise RuntimeError("Cox fold assignment failed.")
    return folds


# --- PyAging components ---


def component_predictions(cohort: pd.DataFrame, cpgs: list[str]) -> pd.DataFrame:
    """Reuse complete component columns or infer the formal 19 OOF clocks."""
    if set(OOF_COMPONENTS).issubset(cohort.columns):
        return cohort[["sampleID", *OOF_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute BioMeta inputs.") from exc
    matrix = numeric_matrix(cohort, cpgs)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = cohort[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    adata.obs["age"] = adata.obs["age_years"]
    adata.obs["Age"] = adata.obs["age_years"]
    female = 1.0 - cohort["sex_male"].to_numpy(float)
    adata.obs["female"] = female
    adata.obs["Female"] = female
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        list(OOF_COMPONENTS),
        dir=relative_path(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    output = adata.obs[["sampleID", *OOF_COMPONENTS]].reset_index(drop=True)
    del adata, matrix
    gc.collect()
    return output


# --- Cox-specific OOF ENCV training ---


def fit_cox_prediction_model(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    """Reuse the exact Figure 4 panel-1 OOF ElasticNetCV configuration."""
    return fit_oof_elasticnet(x, y, seed=seed)


def train_oof_predictions(
    cohort: pd.DataFrame,
    designs: dict[str, np.ndarray],
    *,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Train each biological target model in five outer folds and add observed PhenoAge."""
    folds = fixed_figure4_oof_folds(cohort) if folds is None else np.asarray(folds, dtype=int)
    if len(folds) != len(cohort) or set(folds) != set(range(N_FOLDS)):
        raise ValueError("Cox OOF folds must cover every cohort row with labels 0 through 4.")
    if cohort.assign(fold=folds).groupby("ID")["fold"].nunique().max() != 1:
        raise ValueError("Cox OOF fold assignment splits samples from the same donor.")
    y = cohort[TARGET].to_numpy(float)
    outputs: list[pd.DataFrame] = []
    for fold in range(N_FOLDS):
        train, test = folds != fold, folds == fold
        for model_name in TRAINED_MODELS:
            model = fit_cox_prediction_model(designs[model_name][train], y[train], SEED + fold)
            output = cohort.loc[
                test,
                ["ID", "sampleID", "age_years", "survival_time_years", "event", "sex_male"],
            ].copy()
            output["fold"] = fold + 1
            output["model"] = model_name
            output["prediction"] = model.predict(designs[model_name][test])
            outputs.append(output)
        clinical = cohort.loc[
            test,
            ["ID", "sampleID", "age_years", "survival_time_years", "event", "sex_male"],
        ].copy()
        clinical["fold"] = fold + 1
        clinical["model"] = "clinical_phenoage"
        clinical["prediction"] = y[test]
        outputs.append(clinical)
    predictions = pd.concat(outputs, ignore_index=True)
    for model in MODEL_ORDER:
        subset = predictions[predictions["model"].eq(model)]
        if len(subset) != len(cohort) or subset["ID"].nunique() != len(cohort):
            raise RuntimeError(f"{model} does not have one OOF/observed score per participant.")
    return predictions


# --- Standardized pooled Cox models ---


def zscore(values: pd.Series) -> pd.Series:
    """Standardize a Cox covariate with the Figure 4 population SD convention."""
    numeric = pd.to_numeric(values, errors="raise")
    scale = numeric.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Cox covariate has zero or invalid sample SD.")
    return (numeric - numeric.mean()) / scale


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
        data["prediction_z"] = zscore(data["prediction"])
        data["age_z"] = zscore(data["age_years"])
        data["sex_z"] = zscore(data["sex_male"])
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
    """Draw one forest subplot per adjustment specification."""
    configure_style()
    figure, axes = plt.subplots(1, len(COX_PANELS), figsize=(150 / 25.4, 60 / 25.4), sharey=True)
    titles = {
        "prediction_plus_chronological_age_sex": "Prediction + age + sex",
        "prediction_plus_sex": "Prediction + sex",
        "prediction_plus_chronological_age": "Prediction + age",
    }
    for index, (axis, panel) in enumerate(zip(axes, COX_PANELS, strict=True)):
        table = (
            coefficients[coefficients["panel"].eq(panel)].set_index("model").loc[list(MODEL_ORDER)]
        )
        y = np.arange(len(MODEL_ORDER))
        coefficient = table["coefficient"].to_numpy(float)
        axis.errorbar(
            coefficient,
            y,
            xerr=np.vstack(
                [
                    coefficient - table["ci_lower"].to_numpy(float),
                    table["ci_upper"].to_numpy(float) - coefficient,
                ]
            ),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=0.8,
        )
        axis.scatter(
            coefficient,
            y,
            color=[COLORS.get(model, "#CC79A7") for model in MODEL_ORDER],
            s=28,
            zorder=2,
        )
        axis.axvline(0, color="#666666", linewidth=0.5)
        axis.set(yticks=y, title=titles[panel], xlabel="Standardized Cox coefficient")
        axis.set_yticklabels([LABELS[model] for model in MODEL_ORDER])
        axis.invert_yaxis()
        clean_axis(axis, grid=False)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.2)
        if index == 0:
            axis.set_ylabel("Age model")
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "04_cox_coefficients_standardized")


def main() -> int:
    """Prepare cohort, train OOF models, fit Cox specifications, and plot."""
    cohort = prepare_cohort(read_dataset(DATASET_PATH))
    if len(cohort) != INCHIANTI_PARTICIPANTS or int(cohort["event"].sum()) != INCHIANTI_COX_EVENTS:
        raise ValueError(
            f"Figure 4 Cox requires {INCHIANTI_PARTICIPANTS} participants and "
            f"{INCHIANTI_COX_EVENTS} deaths."
        )
    cpgs = cpg_columns(cohort)
    components = component_predictions(cohort, cpgs).set_index("sampleID")
    designs = {
        "bio_metaclock": numeric_matrix(
            components.loc[cohort["sampleID"]], OOF_COMPONENTS, dtype=np.float64
        ),
        "baseline": numeric_matrix(cohort, cpgs),
    }
    folds = fixed_figure4_oof_folds(cohort)
    predictions = train_oof_predictions(cohort, designs, folds=folds)
    coefficients = fit_cox_models(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "cox_oof_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "cox_coefficients_standardized.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "outer_folds": "fixed formal Figure 4 donor-level OOF assignment",
                "oof_fold_assignment": {
                    "path": relative_path(INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH),
                    "assignment_labels_internal": list(range(N_FOLDS)),
                    "output_fold_labels": list(range(1, N_FOLDS + 1)),
                },
                "prediction_model": (
                    "exact Figure 4 panel-1 StandardScaler + ElasticNetCV; "
                    "cv=5; max_iter=1000; tol=1e-4; selection=random; "
                    "random_state=seed+fold; n_jobs=1; no second refit"
                ),
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
        )
        + "\n",
        encoding="utf-8",
    )
    plot_cox(coefficients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
