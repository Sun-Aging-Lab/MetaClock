#!/usr/bin/env python3
"""Train on nonduplicate participants and evaluate paired DNAm replicates.

Edit only ``DATASET_PATH``.  The input schema is the same as
``run_biological_age_prediction.py``.  Participant IDs with exactly two DNAm
rows are held out in full; every valid participant with exactly one DNAm row is
used for training.  Thus neither member of a replicate pair can enter training.
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXTERNAL_COMPONENTS
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair

# --- Paths: this is the only user-edited input ---

DATASET_PATH = Path("/path/to/your/file")  # fill in your path
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "duplicate_sample_reliability"
PYAGING_CACHE_DIR = Path(__file__).resolve().parent / "pyaging_data"


# --- Fixed formal analysis parameters ---

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
N_FOLDS = 5
SEED = 0
ALPHAS = np.logspace(0, -5, 30)
CV_MAX_ITER = 1_000
REFIT_MAX_ITER = 100_000
TRAINED_MODELS = ("baseline", "bio_metaclock")
MODEL_ORDER = (*TRAINED_MODELS, "dnamphenoage")
REQUIRED_COLUMNS = {"ID", "sampleID", "age_years", TARGET, COMPLETE_CASE}


# --- Dataset loading and duplicate-ID split ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read CSV/CSV.GZ or Parquet with one DNAm sample per row."""
    if not path.exists():
        raise FileNotFoundError(f"Set DATASET_PATH to the analysis-ready dataset: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the explicit complete-case flag safely."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


def split_training_and_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use all valid singleton IDs for training and exactly-two-sample IDs for inference."""
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"Dataset is missing columns: {sorted(missing)}")
    samples = frame.copy()
    samples["ID"] = samples["ID"].astype(str)
    samples["sampleID"] = samples["sampleID"].astype(str)
    if samples["sampleID"].duplicated().any():
        raise ValueError("sampleID must identify one DNAm row.")
    counts = samples.groupby("ID")["sampleID"].nunique()
    if not counts[counts.gt(2)].empty:
        raise ValueError(f"IDs with more than two DNAm samples: {counts[counts.gt(2)].to_dict()}")
    duplicate_ids = set(counts[counts.eq(2)].index)
    holdout = samples[samples["ID"].isin(duplicate_ids)].copy()
    holdout = holdout.sort_values(["ID", "sampleID"], kind="mergesort").reset_index(drop=True)
    valid_target = complete_case_mask(samples[COMPLETE_CASE])
    singleton_ids = set(counts[counts.eq(1)].index)
    training = samples[valid_target & samples["ID"].isin(singleton_ids)].copy()
    for data in (training, holdout):
        data["age_years"] = pd.to_numeric(data["age_years"], errors="coerce")
    training[TARGET] = pd.to_numeric(training[TARGET], errors="coerce")
    training = training.dropna(subset=["age_years", TARGET]).reset_index(drop=True)
    holdout = holdout.dropna(subset=["age_years"]).reset_index(drop=True)
    if not duplicate_ids or len(holdout) != 2 * len(duplicate_ids):
        raise ValueError("No valid exactly-two-sample participant holdout was found.")
    if set(training["ID"]).intersection(duplicate_ids):
        raise RuntimeError("Duplicate participants leaked into training.")
    if len(training) < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} complete singleton participants for training.")
    return training, holdout


def cpg_columns(frame: pd.DataFrame) -> list[str]:
    """Return CpG columns in their input order."""
    columns = [str(column) for column in frame.columns if str(column).lower().startswith("cg")]
    if not columns:
        raise ValueError("No CpG beta columns named cg... were found.")
    return columns


# --- Component clocks through the maintained PyAging API ---


def component_predictions(frame: pd.DataFrame, cpgs: list[str]) -> pd.DataFrame:
    """Reuse complete component columns or compute all canonical components."""
    if set(EXTERNAL_COMPONENTS).issubset(frame.columns):
        return frame[["sampleID", *EXTERNAL_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute component clocks.") from exc
    matrix = frame[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = frame[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
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


# --- Formal ENCV selection and strict refit on every singleton participant ---


def fit_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit StandardScaler + 5-fold ElasticNetCV, then converged selected-alpha EN."""
    selector = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    alphas=ALPHAS,
                    cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
                    max_iter=CV_MAX_ITER,
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
    if any(issubclass(item.category, ConvergenceWarning) for item in refit_warnings):
        raise RuntimeError("Selected-alpha ElasticNet refit did not converge.")
    return model, {
        "selected_alpha": float(selected.alpha_),
        "l1_ratio": float(selected.l1_ratio_),
        "selected_refit_n_iter": n_iter,
        "cv_convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }


# --- Paired inference, reliability summary, and barplot ---


def paired_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse two predictions per ID/model into absolute paired differences."""
    rows: list[dict[str, object]] = []
    for (participant, model), pair in predictions.groupby(["ID", "model"], sort=True):
        pair = pair.sort_values("sampleID")
        if len(pair) != 2:
            raise ValueError(f"Expected two samples for ID={participant}, model={model}.")
        first, second = pair.iloc[0], pair.iloc[1]
        rows.append(
            {
                "ID": participant,
                "model": model,
                "sampleID_1": first["sampleID"],
                "sampleID_2": second["sampleID"],
                "prediction_1": float(first["prediction"]),
                "prediction_2": float(second["prediction"]),
                "absolute_delta": abs(float(first["prediction"] - second["prediction"])),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute replicate reliability for both trained models and DNAmPhenoAge."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        subset = pairs[pairs["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "n_pairs": len(subset),
                "pearson_r": float(
                    pearsonr(subset["prediction_1"], subset["prediction_2"]).statistic
                ),
                "mean_absolute_delta": float(subset["absolute_delta"].mean()),
                "median_absolute_delta": float(subset["absolute_delta"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(summary: pd.DataFrame) -> None:
    """Draw mean absolute paired differences for all three formal comparators."""
    configure_style()
    table = summary.set_index("model").loc[list(MODEL_ORDER)]
    figure, axis = plt.subplots(figsize=(60 / 25.4, 50 / 25.4))
    axis.bar(
        [LABELS[model] for model in MODEL_ORDER],
        table["mean_absolute_delta"],
        color=[COLORS[model] for model in MODEL_ORDER],
        edgecolor="#333333",
        linewidth=0.2,
    )
    axis.set(ylabel="Mean |Δ| (years)", title="Variability Across Replicates")
    clean_axis(axis)
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "02_duplicate_sample_reliability_barplot")


def main() -> int:
    """Train only on singleton IDs, infer both rows of each duplicate ID, and plot."""
    raw = read_dataset(DATASET_PATH)
    training, holdout = split_training_and_duplicates(raw)
    combined = pd.concat([training, holdout], ignore_index=True)
    cpgs = cpg_columns(combined)
    components = component_predictions(combined, cpgs).set_index("sampleID")
    train_components = components.loc[training["sampleID"], list(EXTERNAL_COMPONENTS)].to_numpy(
        float
    )
    holdout_components = components.loc[holdout["sampleID"], list(EXTERNAL_COMPONENTS)].to_numpy(
        float
    )
    train_cpg = (
        training[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    )
    holdout_cpg = (
        holdout[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    )
    y = training[TARGET].to_numpy(float)
    prediction_frames: list[pd.DataFrame] = []
    fit_summary: dict[str, Any] = {}
    for model_name, train_x, holdout_x in (
        ("baseline", train_cpg, holdout_cpg),
        ("bio_metaclock", train_components, holdout_components),
    ):
        model, fit = fit_elasticnet(train_x, y)
        output = holdout[["ID", "sampleID"]].copy()
        output["model"] = model_name
        output["prediction"] = model.predict(holdout_x)
        prediction_frames.append(output)
        fit_summary[model_name] = {**fit, "n_train": len(training), "n_features": train_x.shape[1]}
    direct = holdout[["ID", "sampleID"]].copy()
    direct["model"] = "dnamphenoage"
    direct["prediction"] = components.loc[holdout["sampleID"], "dnamphenoage"].to_numpy(float)
    prediction_frames.append(direct)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pairs = paired_differences(predictions)
    summary = summarize_pairs(pairs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training[["ID", "sampleID", TARGET]].assign(split="nonduplicate_id_train").to_csv(
        OUTPUT_DIR / "training_samples.csv", index=False
    )
    holdout[["ID", "sampleID", COMPLETE_CASE]].assign(split="duplicate_id_holdout").to_csv(
        OUTPUT_DIR / "holdout_samples.csv", index=False
    )
    predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "duplicate_pair_absolute_deltas.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "duplicate_sample_reliability.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "train_rule": "all clinical-PhenoAge-complete IDs with exactly one DNAm sample",
                "holdout_rule": "all IDs with exactly two DNAm samples",
                "n_train_ids": training["ID"].nunique(),
                "n_holdout_ids": holdout["ID"].nunique(),
                "model_order": list(MODEL_ORDER),
                "trained_models": fit_summary,
                "direct_pyaging_inference_models": ["dnamphenoage"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_reliability(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
