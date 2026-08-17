#!/usr/bin/env python3
"""Train five-fold OOF age models, fit standardized Cox models, and plot.

Edit only ``DATASET_PATH``.  The row-oriented table must contain the biological
age columns/CpGs plus ``survival_time_years``, ``event``, and ``sex_male``.
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
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXTERNAL_COMPONENTS
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair

# --- Paths: this is the only user-edited input ---

DATASET_PATH = Path("/path/to/your/file")  # fill in your path
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cox"
PYAGING_CACHE_DIR = Path(__file__).resolve().parent / "pyaging_data"


# --- Fixed formal parameters ---

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
N_FOLDS = 5
SEED = 0
ALPHAS = np.logspace(0, -5, 30)
COX_PENALIZER = 1e-6
TRAINED_MODELS = ("bio_metaclock", "baseline")
MODEL_ORDER = (*TRAINED_MODELS, "clinical_phenoage")
COX_PANELS = {
    "prediction_plus_chronological_age_sex": ["prediction", "age_years", "sex_male"],
    "prediction_plus_sex": ["prediction", "sex_male"],
    "prediction_plus_chronological_age": ["prediction", "age_years"],
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
    if not path.exists():
        raise FileNotFoundError(f"Set DATASET_PATH to the combined InCHIANTI cohort: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the explicit clinical PhenoAge complete-case flag."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


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
    """Return CpG beta columns in file order."""
    columns = [str(column) for column in frame.columns if str(column).lower().startswith("cg")]
    if not columns:
        raise ValueError("No CpG beta columns named cg... were found.")
    return columns


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
    """Reuse complete component columns or infer the canonical 17 clocks with PyAging."""
    if set(EXTERNAL_COMPONENTS).issubset(cohort.columns):
        return cohort[["sampleID", *EXTERNAL_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute BioMeta inputs.") from exc
    matrix = cohort[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = cohort[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    adata.obs["age"] = adata.obs["age_years"]
    adata.obs["Age"] = adata.obs["age_years"]
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        list(EXTERNAL_COMPONENTS),
        dir=str(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    output = adata.obs[["sampleID", *EXTERNAL_COMPONENTS]].reset_index(drop=True)
    del adata, matrix
    gc.collect()
    return output


# --- Cox-specific OOF ENCV training ---


def fit_cox_prediction_model(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    """Fit the formal Cox workflow scaler + ENCV (no selected-alpha second refit)."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    l1_ratio=0.5,
                    alphas=ALPHAS,
                    cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
                    max_iter=1_000,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def train_oof_predictions(cohort: pd.DataFrame, designs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Train each biological target model in five outer folds and add observed PhenoAge."""
    folds = assign_cox_folds(cohort)
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


def fit_cox_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fit raw Cox covariates, then express prediction coefficients per population SD."""
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("Install lifelines to fit the Cox models.") from exc
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        data = predictions[predictions["model"].eq(model)].copy()
        prediction_scale = float(data["prediction"].to_numpy(float).std(ddof=0))
        if not np.isfinite(prediction_scale) or prediction_scale == 0:
            raise ValueError(f"{model} predictions have zero or invalid population SD.")
        for panel, covariates in COX_PANELS.items():
            fit_data = data[["survival_time_years", "event", *covariates]].dropna()
            fitter = CoxPHFitter(penalizer=COX_PENALIZER, baseline_estimation_method="breslow")
            fitter.fit(fit_data, duration_col="survival_time_years", event_col="event")
            estimate = fitter.summary.loc["prediction"]
            coefficient = float(estimate["coef"]) * prediction_scale
            ci_lower = float(estimate["coef lower 95%"] * prediction_scale)
            ci_upper = float(estimate["coef upper 95%"] * prediction_scale)
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
                    "hazard_ratio": float(np.exp(coefficient)),
                    "hr_lower_95": float(np.exp(ci_lower)),
                    "hr_upper_95": float(np.exp(ci_upper)),
                    "raw_coefficient": float(estimate["coef"]),
                    "prediction_population_sd": prediction_scale,
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
    cpgs = cpg_columns(cohort)
    components = component_predictions(cohort, cpgs).set_index("sampleID")
    designs = {
        "bio_metaclock": components.loc[cohort["sampleID"], list(EXTERNAL_COMPONENTS)].to_numpy(
            float
        ),
        "baseline": (
            cohort[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        ),
    }
    predictions = train_oof_predictions(cohort, designs)
    coefficients = fit_cox_models(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "cox_oof_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "cox_coefficients_standardized.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "outer_folds": "participant-level age-quintile StratifiedKFold",
                "prediction_model": "StandardScaler + ElasticNetCV; max_iter=1000; no second refit",
                "cox_scope": "pooled OOF predictions; observed clinical PhenoAge comparator",
                "cox_fit": (
                    "raw covariates with penalizer=1e-6; displayed prediction coefficient "
                    "multiplied by population SD (ddof=0)"
                ),
                "models": list(MODEL_ORDER),
                "panels": COX_PANELS,
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
