#!/usr/bin/env python3
"""Refit BioMeta and the direct-CpG baseline on all 672 InCHIANTI participants."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import FIGURE4, FIGURE4_COMPONENTS, MANIFEST_PATH
    from .modeling import (
        ALPHAS,
        CV_FOLDS,
        MAX_ITER,
        RANDOM_SEED,
        fit_scaled_elasticnet_cv,
        validate_component_features,
    )
    from .provenance import file_record, software_versions, write_json
except ImportError:  # Direct-script fallback.
    from config import FIGURE4, FIGURE4_COMPONENTS, MANIFEST_PATH
    from modeling import (
        ALPHAS,
        CV_FOLDS,
        MAX_ITER,
        RANDOM_SEED,
        fit_scaled_elasticnet_cv,
        validate_component_features,
    )
    from provenance import file_record, software_versions, write_json


DEFAULT_LABELS = Path("data/inchianti/participant_targets.csv")  # fill in your path
DEFAULT_COMPONENTS = Path("data/inchianti/component_clock_predictions.csv")  # fill in your path
DEFAULT_CPG_MATRIX = Path("data/inchianti/cpg_matrix.parquet")  # fill in your path
DEFAULT_OUTPUT = Path("models/figure4/all672_without_dnamfitage")
EXPECTED_PARTICIPANTS = int(FIGURE4["participants"])
EXPECTED_BASELINE_FEATURES = int(FIGURE4["baseline_cpg_features"])
TARGET = "clinical_phenoage"


def parse_args() -> argparse.Namespace:
    """Parse input-table and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--cpg-matrix", type=Path, default=DEFAULT_CPG_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def read_matrix(path: Path) -> pd.DataFrame:
    """Read a CSV/CSV.GZ or Parquet matrix whose first column is sampleID."""
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def select_participants(labels: pd.DataFrame) -> pd.DataFrame:
    """Require the normalized all-participant table to be exactly one row per ID."""
    required = {"ID", "sampleID", TARGET}
    if missing := required.difference(labels.columns):
        raise KeyError(f"Labels are missing columns: {sorted(missing)}")
    selected = labels.copy().reset_index(drop=True)
    selected["sampleID"] = selected["sampleID"].astype(str)
    if len(selected) != EXPECTED_PARTICIPANTS:
        raise ValueError("Normalized training labels must contain exactly 672 rows.")
    if (
        selected["ID"].nunique() != EXPECTED_PARTICIPANTS
        or selected["sampleID"].nunique() != EXPECTED_PARTICIPANTS
    ):
        raise ValueError(
            "Normalized training labels must contain one unique sample per participant."
        )
    return selected


def metric_row(model: str, y: np.ndarray, pred: np.ndarray, n_features: int) -> dict[str, object]:
    """Compute the in-sample audit metrics saved with external refits."""
    mse = float(mean_squared_error(y, pred))
    return {
        "model": model,
        "n": len(y),
        "n_features": n_features,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, pred)),
        "pearson_r": float(pearsonr(y, pred).statistic),
    }


def main() -> int:
    """Train, validate, and serialize both all-participant models."""
    args = parse_args()
    features = list(FIGURE4_COMPONENTS)
    validate_component_features(features)
    labels = select_participants(pd.read_csv(args.labels))
    components = pd.read_csv(args.components)
    cpg = read_matrix(args.cpg_matrix)
    for table in (components, cpg):
        table["sampleID"] = table["sampleID"].astype(str)
        if table["sampleID"].duplicated().any():
            raise ValueError("Input matrix contains duplicate sampleID values.")

    training = labels.merge(
        components[["sampleID", *features]], on="sampleID", validate="one_to_one"
    ).merge(cpg, on="sampleID", validate="one_to_one")
    if len(training) != EXPECTED_PARTICIPANTS:
        raise ValueError("Not every selected participant has both model inputs.")
    cpg_features = [column for column in cpg.columns if column != "sampleID"]
    if len(cpg_features) != EXPECTED_BASELINE_FEATURES or len(set(cpg_features)) != len(
        cpg_features
    ):
        raise ValueError(
            f"Direct-CpG baseline requires exactly {EXPECTED_BASELINE_FEATURES:,} unique CpGs; "
            f"found {len(cpg_features):,}."
        )
    y = pd.to_numeric(training[TARGET], errors="raise").to_numpy(float)
    designs = {
        "bio_metaclock": training[features].to_numpy(float),
        "baseline": training[cpg_features].to_numpy(float),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for name, x in designs.items():
        fit = fit_scaled_elasticnet_cv(x, y, n_jobs=args.n_jobs)
        pred = np.asarray(fit.model.predict(x), dtype=float)
        model_features = features if name == "bio_metaclock" else cpg_features
        joblib.dump(fit.model, args.output_dir / f"{name}.joblib")
        pd.Series(model_features, name="feature").to_csv(
            args.output_dir / f"{name}_features.csv", index=False
        )
        metric_rows.append(metric_row(name, y, pred, len(model_features)))
        prediction_rows.append(
            pd.DataFrame(
                {
                    "ID": labels["ID"],
                    "sampleID": labels["sampleID"],
                    TARGET: y,
                    "model": name,
                    "pred_bioage": pred,
                }
            )
        )
        summaries[name] = {
            "n_train": len(y),
            "n_holdout": 0,
            "n_features": len(model_features),
            "selected_alpha": fit.alpha,
            "l1_ratio": fit.l1_ratio,
            "selected_refit_n_iter": fit.n_iter,
            "best_cv_mse": fit.best_cv_mse,
            "cv_convergence_warnings": 0,
        }
        pd.DataFrame(fit.alpha_search).sort_values("alpha", ascending=False).to_csv(
            args.output_dir / f"{name}_alpha_search.csv", index=False
        )

    pd.DataFrame(metric_rows).to_csv(args.output_dir / "training_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "training_predictions.csv", index=False
    )
    labels[["ID", "sampleID", TARGET]].to_csv(args.output_dir / "training_samples.csv", index=False)
    summary = {
        "models": summaries,
        "refit_on_all_participants": True,
        "analysis_unit": "participant ID",
        "n_participants": EXPECTED_PARTICIPANTS,
        "excluded_component": "dnamfitage",
        "alpha_grid": [float(value) for value in ALPHAS],
        "cv_folds": CV_FOLDS,
        "cv_splitter": f"KFold(shuffle=True, random_state={RANDOM_SEED})",
        "cv_preprocessing": "StandardScaler fit independently inside each GridSearchCV fold",
        "max_iter": MAX_ITER,
        "seed": RANDOM_SEED,
        "standard_scaler": True,
        "manifest": file_record(MANIFEST_PATH),
        "inputs": {
            "labels": file_record(args.labels),
            "components": file_record(args.components),
            "cpg_matrix": file_record(args.cpg_matrix),
        },
        "software": software_versions(("joblib", "numpy", "pandas", "scikit-learn", "scipy")),
        "artifacts": {
            name: file_record(args.output_dir / f"{name}.joblib")
            for name in ("bio_metaclock", "baseline")
        },
        "feature_manifests": {
            name: file_record(args.output_dir / f"{name}_features.csv")
            for name in ("bio_metaclock", "baseline")
        },
        "derived_outputs": {
            name: file_record(args.output_dir / filename)
            for name, filename in {
                "training_samples": "training_samples.csv",
                "training_predictions": "training_predictions.csv",
                "training_metrics": "training_metrics.csv",
                "bio_metaclock_alpha_search": "bio_metaclock_alpha_search.csv",
                "baseline_alpha_search": "baseline_alpha_search.csv",
            }.items()
        },
    }
    write_json(args.output_dir / "training_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
