#!/usr/bin/env python3
"""Train BioMeta/direct-CpG models and draw 5-fold OOF performance bars.

Run ``prepare_inchianti_dataset`` first; ``DATASET_PATH`` points to its output.
The input is one sample per row and must contain ID, sampleID, age_years,
clinical_phenoage, clinical_phenoage_complete_case, and all 485,577 InCHIANTI DNAm features.  The
OOF model uses 19 component clocks.
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from repository_paths import relative_path

from .config import (
    EXTERNAL_COMPONENTS,
    INCHIANTI_COMPLETE_CASE_SAMPLES,
    INCHIANTI_PARTICIPANTS,
    PRODUCTION_CONFIG_ID,
    PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH,
    OOF_COMPONENTS,
)
from .data import (
    complete_case_mask as parse_complete_case_mask,
)
from .data import (
    dnam_feature_columns,
    sort_figure4_samples,
    numeric_matrix,
    read_analysis_dataset,
)
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions
from .production_oof import (
    primary_samples, production_oof_folds, save_oof_predictions, save_duplicate_predictions,
)

# --- Repository-local generated input, outputs, and cache ---

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "preprocessing"
    / "inchianti_analysis_ready.parquet"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "biological_age_prediction"
PYAGING_CACHE_DIR = Path(__file__).resolve().parent / "pyaging_data"


# --- Fixed formal analysis parameters ---

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
N_FOLDS = 5
SEED = 0
ALPHAS = np.logspace(0, -5, 30)
PRODUCTION_MAX_ITER = 20_000
SEARCH_WORKERS = 4
MODEL_ORDER = ("bio_metaclock", "baseline")
REQUIRED_COLUMNS = {"ID", "sampleID", "age_years", "sex_male", TARGET, COMPLETE_CASE}


# --- Input and clinical-PhenoAge complete-case selection ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a row-oriented CSV/CSV.GZ or Parquet analysis table."""
    return read_analysis_dataset(path, float_precision="round_trip")


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the explicit clinical PhenoAge completeness flag without truthy strings."""
    return parse_complete_case_mask(values)


def select_valid_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the formal clinical_phenoage_complete_case sample filter."""
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"Dataset is missing columns: {sorted(missing)}")
    valid = frame.loc[complete_case_mask(frame[COMPLETE_CASE])].copy()
    valid["ID"] = valid["ID"].astype(str)
    valid["sampleID"] = valid["sampleID"].astype(str)
    for column in ("age_years", "sex_male", TARGET):
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=["ID", "sampleID", "age_years", "sex_male", TARGET])
    if valid["sampleID"].duplicated().any():
        raise ValueError("sampleID must identify one DNAm row.")
    if valid["ID"].nunique() < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} valid participants.")
    return sort_figure4_samples(valid)


def cpg_columns(frame: pd.DataFrame) -> list[str]:
    """Return all 485,577 formal baseline features in file order."""
    return dnam_feature_columns(frame)


# --- PyAging component-clock inference ---


def run_pyaging_components(frame: pd.DataFrame, cpgs: list[str]) -> pd.DataFrame:
    """Compute the formal 19 OOF component clocks with PyAging."""
    if set(OOF_COMPONENTS).issubset(frame.columns):
        return frame[["sampleID", *OOF_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute component clocks.") from exc

    obs = frame[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    obs["age"] = obs["age_years"]
    obs["Age"] = obs["age_years"]
    for source in ("female", "Female"):
        if source in frame.columns:
            obs["female"] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
            obs["Female"] = obs["female"]
            break
    else:
        if "sex_male" not in frame.columns:
            raise ValueError("The 19-clock OOF model requires sex_male for GrimAge metadata.")
        female = 1.0 - pd.to_numeric(frame["sex_male"], errors="coerce").to_numpy()
        obs["female"] = female
        obs["Female"] = female
    matrix = np.column_stack(
        (numeric_matrix(frame, cpgs), numeric_matrix(obs, ["age", "female"]))
    )
    adata = ad.AnnData(matrix)
    adata.var_names = [*cpgs, "age", "female"]
    adata.obs = obs
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


# --- Cross-validated elastic net and selected-alpha refit ---


def fit_fold_local_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> Pipeline:
    """Fit the fixed five-fold alpha search with all learned preprocessing inside CV."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Invalid feature matrix or nonfinite training target.")
    if np.isinf(x).any():
        x = x.copy()
        x[np.isinf(x)] = np.nan
    splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x))
    steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True))]
    steps.extend([
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(
            l1_ratio=0.5, fit_intercept=True, max_iter=PRODUCTION_MAX_ITER,
            tol=1e-4, selection="cyclic", random_state=SEED,
            warm_start=False, positive=False, precompute=False,
        )),
    ])
    estimator = Pipeline(steps)
    search = GridSearchCV(
        estimator, {"elasticnet__alpha": ALPHAS.copy()}, cv=splits,
        scoring="neg_mean_squared_error", refit=True, n_jobs=SEARCH_WORKERS,
        pre_dispatch=SEARCH_WORKERS, error_score="raise", return_train_score=False,
    )
    with threadpool_limits(limits=1), joblib.parallel_backend("threading", n_jobs=SEARCH_WORKERS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(x, y)
    if not np.isfinite(search.cv_results_["mean_test_score"]).all():
        raise RuntimeError("Nonfinite inner-CV scores.")
    model = search.best_estimator_
    selected = model.named_steps["elasticnet"]
    model.fit_diagnostics_ = {
        "selected_alpha": float(selected.alpha),
        "l1_ratio": float(selected.l1_ratio),
        "selected_refit_n_iter": int(selected.n_iter_),
        "selected_refit_dual_gap": float(selected.dual_gap_),
        "selected_refit_at_iteration_cap": bool(selected.n_iter_ >= PRODUCTION_MAX_ITER),
        "cv_convergence_warnings": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
        "best_mean_validation_mse": float(-search.best_score_),
        "n_features": x.shape[1],
        "selected_features": x.shape[1],
    }
    return model


def fit_full_refit_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select alpha and refit the selected pipeline once on all supplied training rows."""
    model = fit_fold_local_elasticnet(x, y, seed=seed)
    return model, model.fit_diagnostics_


def select_one_sample_per_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest numeric sampleID suffix, matching the formal 672-person refit."""
    return primary_samples(frame)


# --- OOF metrics, full-participant artifacts, and barplot ---


def fit_oof_models(
    frame: pd.DataFrame,
    designs: dict[str, np.ndarray],
    *,
    folds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on primary profiles and score one held-out primary profile per participant."""
    ordered = sort_figure4_samples(frame)
    row_by_sample = {sample: row for row, sample in enumerate(frame.sampleID.astype(str))}
    order = np.asarray([row_by_sample[sample] for sample in ordered.sampleID], dtype=int)
    designs = {name: np.asarray(matrix)[order] for name, matrix in designs.items()}
    expected_folds = production_oof_folds(ordered)
    if folds is not None and not np.array_equal(np.asarray(folds, dtype=int)[order], expected_folds):
        raise ValueError("OOF assignment differs from the production participant folds.")
    frame, folds = ordered, expected_folds
    model_names = tuple(name for name in MODEL_ORDER if name in designs)
    if not model_names or set(model_names) != set(designs):
        raise ValueError("Unknown or empty prediction model collection.")
    y = frame[TARGET].to_numpy(float)
    primary = primary_samples(frame)
    primary_mask = frame.sampleID.isin(primary.sampleID).to_numpy()
    predictions, fit_rows = [], []
    model_dir = OUTPUT_DIR / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        train = primary_mask & (folds != fold)
        test = primary_mask & (folds == fold)
        for name in model_names:
            model = fit_fold_local_elasticnet(
                designs[name][train], y[train],
            )
            joblib.dump(model, model_dir / f"{name}_fold{fold}.joblib")
            fit_rows.append({"fold": fold, "model": name, **model.fit_diagnostics_,
                             "n_train_participants": int(train.sum()), "n_test_samples": int(test.sum())})
            output = frame.loc[test, ["ID", "sampleID", "age_years", "sex_male", TARGET]].copy()
            output["fold"], output["model"] = fold, name
            output["pred_bioage"] = model.predict(designs[name][test])
            predictions.append(output)
    predictions = pd.concat(predictions, ignore_index=True)
    return predictions, summarize_oof_predictions(predictions), pd.DataFrame(fit_rows)


def summarize_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate pooled performance on the 672 primary participant predictions."""
    rows = []
    for name, subset in predictions.groupby("model", sort=False):
        if len(subset) != INCHIANTI_PARTICIPANTS or subset.ID.duplicated().any():
            raise ValueError("OOF metrics require one primary prediction per participant.")
        rows.append({"model": name, "n": len(subset),
                     "mae": float(mean_absolute_error(subset[TARGET], subset.pred_bioage)),
                     "pearson_r": float(pearsonr(subset[TARGET], subset.pred_bioage).statistic)})
    return pd.DataFrame(rows)


def fit_duplicate_holdout_models(
    frame: pd.DataFrame, designs: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on 628 singleton participants and hold out all 88 duplicate profiles."""
    ordered = sort_figure4_samples(frame)
    row_by_sample = {sample: row for row, sample in enumerate(frame.sampleID.astype(str))}
    order = np.asarray([row_by_sample[sample] for sample in ordered.sampleID], dtype=int)
    counts = ordered.groupby("ID").sampleID.transform("size").to_numpy()
    train, test = counts == 1, counts == 2
    if train.sum() != 628 or test.sum() != 88 or not np.all(train | test):
        raise ValueError("Duplicate reliability requires 628 singleton donors and 44 pairs.")
    predictions, diagnostics = [], []
    model_dir = OUTPUT_DIR / "duplicate_holdout_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in designs.items():
        if name not in MODEL_ORDER:
            raise ValueError(f"Unknown model {name}.")
        values = np.asarray(matrix)[order]
        model = fit_fold_local_elasticnet(
            values[train], ordered.loc[train, TARGET].to_numpy(float),
        )
        joblib.dump(model, model_dir / f"{name}.joblib")
        output = ordered.loc[test, ["ID", "sampleID", "age_years", "sex_male", TARGET]].copy()
        output["fold"], output["model"] = -1, name
        output["pred_bioage"] = model.predict(values[test])
        predictions.append(output)
        diagnostics.append({"model": name, **model.fit_diagnostics_,
                            "n_train_participants": int(train.sum()), "n_test_samples": int(test.sum())})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)


def write_full_refit_artifacts(
    frame: pd.DataFrame,
    designs: dict[str, np.ndarray],
    feature_names: dict[str, list[str]],
) -> dict[str, Any]:
    """Refit the requested external artifacts and export the component model formula."""
    from .published_biometaclock import export_published_biometaclock

    model_names = tuple(name for name in MODEL_ORDER if name in designs)
    if not model_names or set(model_names) != set(designs) or set(feature_names) != set(designs):
        raise ValueError("Unknown or incomplete external model collection.")
    if "bio_metaclock" in model_names and feature_names["bio_metaclock"] != list(EXTERNAL_COMPONENTS):
        raise ValueError("External BioMetaClock requires the ordered 17-component manifest.")
    selected = select_one_sample_per_participant(frame)
    position_by_sample = {sample: index for index, sample in enumerate(frame["sampleID"])}
    positions = np.array([position_by_sample[sample] for sample in selected["sampleID"]], dtype=int)
    y = selected[TARGET].to_numpy(float)
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"n_participants": len(selected), "models": {}}
    for model_name in model_names:
        model, fit = fit_full_refit_elasticnet(
            designs[model_name][positions],
            y,
            seed=SEED,
        )
        joblib.dump(model, model_dir / f"{model_name}.joblib")
        pd.Series(feature_names[model_name], name="feature").to_csv(
            model_dir / f"{model_name}_features.csv", index=False
        )
        summary["models"][model_name] = {**fit, "n_features": len(feature_names[model_name])}
        if model_name == "bio_metaclock":
            weights_path, metadata_path = export_published_biometaclock(
                model, feature_names[model_name], model_dir,
                training_participants=len(selected),
                search_config={
                    "implementation": "GridSearchCV(Pipeline(SimpleImputer, StandardScaler, ElasticNet))",
                    "alphas": ALPHAS.tolist(),
                    "inner_cv": {"class": "KFold", "n_splits": N_FOLDS,
                                 "shuffle": True, "random_state": SEED},
                    "scoring": "neg_mean_squared_error", "refit": True,
                    "preprocessing_fit_scope": "each inner CV training fold",
                    "n_jobs": SEARCH_WORKERS, "pre_dispatch": SEARCH_WORKERS,
                },
                cv_convergence_warnings=fit["cv_convergence_warnings"],
            )
            summary["models"][model_name]["published_weights"] = relative_path(weights_path)
            summary["models"][model_name]["published_metadata"] = relative_path(metadata_path)
    selected[["ID", "sampleID", TARGET]].to_csv(model_dir / "training_samples.csv", index=False)
    return summary


def plot_metrics(metrics: pd.DataFrame) -> None:
    """Draw the formal MAE and Pearson-r panels."""
    configure_style()
    table = metrics.set_index("model").loc[list(MODEL_ORDER)]
    figure, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 50 / 25.4))
    specs = (("mae", "MAE (years)", "Prediction error"), ("pearson_r", "Pearson r", "Correlation"))
    for axis, (column, ylabel, title) in zip(axes, specs, strict=True):
        axis.bar(
            [LABELS[model] for model in MODEL_ORDER],
            table[column],
            color=[COLORS[model] for model in MODEL_ORDER],
            edgecolor="#333333",
            linewidth=0.2,
        )
        axis.set(ylabel=ylabel, title=title)
        if column == "pearson_r":
            axis.set_ylim(0.90, 1.00)
        clean_axis(axis)
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "01_bioage_mae_pearson_barplot")


def main() -> int:
    """Run complete-case selection, grouped OOF, full refit, and plotting."""
    frame = select_valid_samples(read_dataset(DATASET_PATH))
    if (
        len(frame) != INCHIANTI_COMPLETE_CASE_SAMPLES
        or frame["ID"].nunique() != INCHIANTI_PARTICIPANTS
    ):
        raise ValueError(
            "Figure 4 OOF requires "
            f"{INCHIANTI_COMPLETE_CASE_SAMPLES} samples from {INCHIANTI_PARTICIPANTS} participants."
        )
    cpgs = cpg_columns(frame)
    components = run_pyaging_components(frame, cpgs)
    training = frame.drop(columns=list(OOF_COMPONENTS), errors="ignore").merge(
        components,
        on="sampleID",
        validate="one_to_one",
    )
    designs = {
        "bio_metaclock": numeric_matrix(training, OOF_COMPONENTS, dtype=np.float64, fill_missing=False),
        "baseline": numeric_matrix(training, cpgs, dtype=np.float64, fill_missing=False),
    }
    external_designs = {
        "bio_metaclock": numeric_matrix(training, EXTERNAL_COMPONENTS, dtype=np.float64, fill_missing=False),
        "baseline": designs["baseline"],
    }
    feature_names = {"bio_metaclock": list(EXTERNAL_COMPONENTS), "baseline": cpgs}
    folds = production_oof_folds(training)
    predictions, metrics, fold_fits = fit_oof_models(training, designs, folds=folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training[["ID", "sampleID", "age_years", TARGET]].assign(
        fold=folds,
        selected_for_training=training["sampleID"].isin(
            select_one_sample_per_participant(training)["sampleID"]
        ),
    ).to_csv(OUTPUT_DIR / "valid_samples_and_folds.csv", index=False)
    save_oof_predictions(predictions, output_dir=OUTPUT_DIR)
    duplicate_predictions, duplicate_fits = fit_duplicate_holdout_models(training, designs)
    save_duplicate_predictions(duplicate_predictions, training, output_dir=OUTPUT_DIR)
    duplicate_fits.to_csv(OUTPUT_DIR / "duplicate_holdout_selected_alphas.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "oof_metrics.csv", index=False)
    fold_fits.to_csv(OUTPUT_DIR / "oof_selected_alphas.csv", index=False)
    component_path = OUTPUT_DIR / "component_clock_predictions.csv"
    components.to_csv(component_path, index=False)
    refit = write_full_refit_artifacts(training, external_designs, feature_names)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "config_id": PRODUCTION_CONFIG_ID,
                "complete_case_filter": COMPLETE_CASE,
                "outer_folds": N_FOLDS,
                "outer_fold_group": "ID",
                "oof_model": (
                    "GridSearchCV(SimpleImputer + StandardScaler + ElasticNet); "
                    "fold-local preprocessing; scoring=neg_mean_squared_error; "
                    "KFold(5, shuffle=True, random_state=0); selection=cyclic; max_iter=20000"
                ),
                "external_full_refit_model": (
                    "Same fixed 20000-iteration search and single selected-alpha pipeline refit on all 672 participants"
                ),
                "oof_component_features": list(OOF_COMPONENTS),
                "external_component_features": list(EXTERNAL_COMPONENTS),
                "oof_training_rows": "one lowest-suffix DNAm sample per participant",
                "oof_prediction_rows": "one primary DNAm sample per participant",
                "baseline": "all 485577 DNAm features; zero imputation and scaling inside each training fold",
                "duplicate_holdout": "628 singleton participants train; all 88 profiles from 44 pairs held out",
                "convergence_policy": "fixed iteration cap; warnings and selected-refit diagnostics retained",
                "metric_level": "672 pooled primary OOF predictions",
                "oof_fold_assignment": {
                    "path": relative_path(PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH),
                    "fold_labels": list(range(N_FOLDS)),
                },
                "full_refit": refit,
                "software": software_versions(
                    ["numpy", "pandas", "scikit-learn", "pyaging", "anndata", "joblib"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
