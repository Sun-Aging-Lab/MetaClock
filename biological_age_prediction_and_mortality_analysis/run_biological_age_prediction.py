#!/usr/bin/env python3
"""Train BioMeta/direct-CpG models and draw 5-fold OOF performance bars.

Edit only ``DATASET_PATH`` below.  The input is one sample per row and must
contain ID, sampleID, age_years, clinical_phenoage,
clinical_phenoage_complete_case, and all 485,577 InCHIANTI DNAm features.  The
OOF model uses 19 component clocks; the external refit uses the 17 clocks that
are available in every external dataset.
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
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    EXTERNAL_COMPONENTS,
    INCHIANTI_COMPLETE_CASE_SAMPLES,
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
from .data import (
    select_one_sample_per_participant as select_primary_samples,
)
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions

# --- Paths: this is the only user-edited input ---

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
OOF_MAX_ITER = 1_000
FULL_REFIT_CV_MAX_ITER = 10_000
REFIT_MAX_ITER = 100_000
MODEL_ORDER = ("bio_metaclock", "baseline")
REQUIRED_COLUMNS = {"ID", "sampleID", "age_years", "sex_male", TARGET, COMPLETE_CASE}


# --- Input and clinical-PhenoAge complete-case selection ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a row-oriented CSV/CSV.GZ or Parquet analysis table."""
    return read_analysis_dataset(path)


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
    return valid.sort_values("sampleID", kind="mergesort").reset_index(drop=True)


def cpg_columns(frame: pd.DataFrame) -> list[str]:
    """Return all 485,577 formal baseline features in file order."""
    return dnam_feature_columns(frame)


# --- Participant-aware five-fold assignment ---


def assign_grouped_age_folds(
    frame: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """Age-bin participants, then map every sample from an ID to one outer fold."""
    participants = (
        frame[["ID", "age_years"]]
        .assign(ID=frame["ID"].astype(str))
        .groupby("ID", as_index=False)
        .agg(age_years=("age_years", "median"))
    )
    bins = (participants["age_years"] // 10 * 10).astype("Int64")
    participant_folds = np.full(len(participants), -1, dtype=int)
    rng = np.random.default_rng(seed)
    positions = pd.Series(np.arange(len(participants), dtype=int))
    for _, indices in positions.groupby(bins, dropna=False):
        shuffled = indices.to_numpy(dtype=int)
        rng.shuffle(shuffled)
        if len(shuffled) < n_folds:
            participant_folds[shuffled] = rng.choice(
                n_folds, size=len(shuffled), replace=False
            )
        else:
            participant_folds[shuffled] = np.arange(len(shuffled), dtype=int) % n_folds
    fold_by_id = dict(zip(participants["ID"], participant_folds, strict=True))
    folds = frame["ID"].astype(str).map(fold_by_id).to_numpy(dtype=int)
    if (folds < 0).any() or frame.assign(fold=folds).groupby("ID")["fold"].nunique().max() != 1:
        raise RuntimeError("Participant-aware fold assignment failed.")
    return folds


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

    matrix = numeric_matrix(frame, cpgs)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = frame[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    adata.obs["age"] = adata.obs["age_years"]
    adata.obs["Age"] = adata.obs["age_years"]
    for source in ("female", "Female"):
        if source in frame.columns:
            adata.obs["female"] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
            adata.obs["Female"] = adata.obs["female"]
            break
    else:
        if "sex_male" not in frame.columns:
            raise ValueError("The 19-clock OOF model requires sex_male for GrimAge metadata.")
        female = 1.0 - pd.to_numeric(frame["sex_male"], errors="coerce").to_numpy()
        adata.obs["female"] = female
        adata.obs["Female"] = female
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        list(OOF_COMPONENTS),
        dir=str(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    output = adata.obs[["sampleID", *OOF_COMPONENTS]].reset_index(drop=True)
    del adata, matrix
    gc.collect()
    return output


# --- StandardScaler + ElasticNetCV training and strict selected-alpha refit ---


def fit_oof_elasticnet(x: np.ndarray, y: np.ndarray, *, seed: int) -> Pipeline:
    """Fit the exact formal OOF scaler + ElasticNetCV pipeline."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    alphas=ALPHAS,
                    cv=N_FOLDS,
                    max_iter=OOF_MAX_ITER,
                    tol=1e-4,
                    selection="random",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def fit_full_refit_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select alpha with the full-data CV, then refit that alpha to convergence."""
    selector = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    alphas=ALPHAS,
                    cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
                    max_iter=FULL_REFIT_CV_MAX_ITER,
                    n_jobs=1,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        selector.fit(x, y)
    selected = selector.named_steps["elasticnetcv"]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=float(selected.alpha_),
                    l1_ratio=float(selected.l1_ratio_),
                    max_iter=REFIT_MAX_ITER,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as refit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    n_iter = int(model.named_steps["elasticnet"].n_iter_)
    failed = any(issubclass(item.category, ConvergenceWarning) for item in refit_warnings)
    if failed or n_iter >= REFIT_MAX_ITER:
        raise RuntimeError("Selected-alpha ElasticNet refit did not converge.")
    return model, {
        "selected_alpha": float(selected.alpha_),
        "l1_ratio": float(selected.l1_ratio_),
        "selected_refit_n_iter": n_iter,
        "cv_convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }


def select_one_sample_per_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest numeric sampleID suffix, matching the formal 672-person refit."""
    return select_primary_samples(frame)


# --- OOF metrics, full-participant artifacts, and barplot ---


def fit_oof_models(
    frame: pd.DataFrame,
    designs: dict[str, np.ndarray],
    *,
    folds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on one row per ID, predict all held-out rows, and summarize by ID."""
    folds = assign_grouped_age_folds(frame) if folds is None else np.asarray(folds, dtype=int)
    if len(folds) != len(frame):
        raise ValueError("OOF fold assignment length does not match the analysis rows.")
    if set(folds) != set(range(N_FOLDS)):
        raise ValueError(f"OOF folds must contain every label from 0 to {N_FOLDS - 1}.")
    if frame.assign(fold=folds).groupby("ID")["fold"].nunique().max() != 1:
        raise ValueError("OOF fold assignment splits samples from the same donor.")
    y = frame[TARGET].to_numpy(float)
    primary = select_one_sample_per_participant(frame)
    primary_samples = set(primary["sampleID"])
    primary_mask = frame["sampleID"].isin(primary_samples).to_numpy()
    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    model_dir = OUTPUT_DIR / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        train = primary_mask & (folds != fold)
        test = folds == fold
        if not test.any():
            raise ValueError(f"Outer fold {fold} is empty.")
        for model_name in MODEL_ORDER:
            model = fit_oof_elasticnet(
                designs[model_name][train],
                y[train],
                seed=SEED + fold,
            )
            joblib.dump(model, model_dir / f"{model_name}_fold{fold}.joblib")
            selected = model.named_steps["elasticnetcv"]
            fit_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "selected_alpha": float(selected.alpha_),
                    "l1_ratio": float(selected.l1_ratio_),
                    "n_iter": int(np.max(np.atleast_1d(selected.n_iter_))),
                    "n_train_participants": int(train.sum()),
                    "n_test_samples": int(test.sum()),
                }
            )
            output = frame.loc[test, ["ID", "sampleID", "age_years", TARGET]].copy()
            output["fold"] = fold
            output["model"] = model_name
            output["pred_bioage"] = model.predict(designs[model_name][test])
            prediction_frames.append(output)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = summarize_oof_predictions(predictions)
    return predictions, metrics, pd.DataFrame(fit_rows)


def summarize_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average held-out replicates within ID before calculating panel-1 metrics."""
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        subset = predictions[predictions["model"].eq(model_name)]
        participant = subset.groupby("ID", as_index=False).agg(
            clinical_phenoage=(TARGET, "mean"),
            pred_bioage=("pred_bioage", "mean"),
        )
        truth = participant["clinical_phenoage"].to_numpy(float)
        pred = participant["pred_bioage"].to_numpy(float)
        rows.append(
            {
                "model": model_name,
                "n": len(participant),
                "mae": float(mean_absolute_error(truth, pred)),
                "pearson_r": float(pearsonr(truth, pred).statistic),
            }
        )
    return pd.DataFrame(rows)


def write_full_refit_artifacts(
    frame: pd.DataFrame,
    designs: dict[str, np.ndarray],
    feature_names: dict[str, list[str]],
) -> dict[str, Any]:
    """Refit both external-use artifacts on one selected sample per participant."""
    selected = select_one_sample_per_participant(frame)
    position_by_sample = {sample: index for index, sample in enumerate(frame["sampleID"])}
    positions = np.array([position_by_sample[sample] for sample in selected["sampleID"]], dtype=int)
    y = selected[TARGET].to_numpy(float)
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"n_participants": len(selected), "models": {}}
    for model_name in MODEL_ORDER:
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
        "bio_metaclock": numeric_matrix(training, OOF_COMPONENTS, dtype=np.float64),
        "baseline": numeric_matrix(training, cpgs),
    }
    external_designs = {
        "bio_metaclock": numeric_matrix(training, EXTERNAL_COMPONENTS, dtype=np.float64),
        "baseline": designs["baseline"],
    }
    feature_names = {"bio_metaclock": list(EXTERNAL_COMPONENTS), "baseline": cpgs}
    folds = fixed_figure4_oof_folds(training)
    predictions, metrics, fold_fits = fit_oof_models(training, designs, folds=folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training[["ID", "sampleID", "age_years", TARGET]].assign(
        fold=folds,
        selected_for_training=training["sampleID"].isin(
            select_one_sample_per_participant(training)["sampleID"]
        ),
    ).to_csv(OUTPUT_DIR / "valid_samples_and_folds.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "oof_metrics.csv", index=False)
    fold_fits.to_csv(OUTPUT_DIR / "oof_selected_alphas.csv", index=False)
    component_path = OUTPUT_DIR / "component_clock_predictions.csv"
    components.to_csv(component_path, index=False)
    refit = write_full_refit_artifacts(training, external_designs, feature_names)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "complete_case_filter": COMPLETE_CASE,
                "outer_folds": N_FOLDS,
                "outer_fold_group": "ID",
                "oof_model": (
                    "StandardScaler + ElasticNetCV; cv=5; selection=random; "
                    "max_iter=1000; no selected-alpha refit"
                ),
                "external_full_refit_model": (
                    "StandardScaler + ElasticNetCV max_iter=10000, followed by a fresh "
                    "selected-alpha ElasticNet max_iter=100000 on one sample per participant"
                ),
                "oof_component_features": list(OOF_COMPONENTS),
                "external_component_features": list(EXTERNAL_COMPONENTS),
                "oof_training_rows": "one lowest-suffix DNAm sample per participant",
                "oof_prediction_rows": "all complete-case DNAm samples",
                "metric_level": "participant mean of held-out DNAm predictions",
                "oof_fold_assignment": {
                    "path": str(INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH.resolve()),
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
