#!/usr/bin/env python3
"""Train the biological MetaClock used for intervention prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .biometaclock_modeling import (
    ALPHAS,
    CV_FOLDS,
    CV_MAX_ITER,
    REFIT_MAX_ITER,
    SEED,
    fit_scaled_elastic_net_cv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_TABLE = Path("/path/to/your/file")  # fill in your path
DEFAULT_FEATURE_TABLE = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/intervention/model"
TARGET = "clinical_phenoage"
EXCLUDED_FEATURES = {"dnamfitage"}


def load_features(path: Path) -> list[str]:
    """Load the ordered component-clock manifest and enforce global exclusions."""
    features = pd.read_csv(path)["feature"].astype(str).str.lower().tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature manifest must contain unique feature names.")
    forbidden = sorted(set(features).intersection(EXCLUDED_FEATURES))
    if forbidden:
        raise ValueError(f"Excluded biological clocks are present: {forbidden}")
    return features


def load_training_matrix(path: Path, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return complete finite component predictions and the phenotype target."""
    frame = pd.read_csv(path)
    missing = sorted({TARGET, *features}.difference(frame.columns))
    if missing:
        raise KeyError(f"Training table is missing columns: {missing}")
    matrix = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Training features and target must be finite.")
    return matrix, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table", type=Path, default=DEFAULT_TRAINING_TABLE)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = load_features(args.feature_table)
    matrix, target = load_training_matrix(args.training_table, features)
    fitted = fit_scaled_elastic_net_cv(matrix, target, n_jobs=args.n_jobs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted.model, args.output_dir / "biometaclock.joblib")
    pd.Series(features, name="feature").to_csv(
        args.output_dir / "biometaclock_features.csv", index=False
    )
    summary = {
        "n_samples": int(len(target)),
        "n_features": len(features),
        "excluded_features": sorted(EXCLUDED_FEATURES),
        "pipeline": (
            "Pipeline(StandardScaler, ElasticNetCV) + fresh "
            "Pipeline(StandardScaler, ElasticNet) selected-alpha refit"
        ),
        "cv_folds": CV_FOLDS,
        "alphas": ALPHAS.tolist(),
        "selected_alpha": fitted.alpha,
        "l1_ratio": fitted.l1_ratio,
        "best_cv_mse": fitted.best_cv_mse,
        "refit_n_iter": fitted.n_iter,
        "cv_convergence_warnings": fitted.cv_convergence_warnings,
        "refit_convergence_warnings": fitted.refit_convergence_warnings,
        "cv_max_iter": CV_MAX_ITER,
        "refit_max_iter": REFIT_MAX_ITER,
        "seed": SEED,
        "scikit_learn": sklearn.__version__,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(fitted.alpha_search).to_csv(args.output_dir / "alpha_search.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
