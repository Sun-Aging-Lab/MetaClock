#!/usr/bin/env python3
"""Fit and export the canonical 16-component chronological MetaClock."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .train_test_architectures import (
    COMPONENT_CLOCKS,
    DEFAULT_TRAINING_DATASET,
    read_dataset,
    sklearn_architectures,
)


MODEL_DIR = Path(__file__).resolve().parent / "model_weights"
WEIGHTS_PATH = MODEL_DIR / "metaclock_chronological_3353_samples_weights.csv"
METADATA_PATH = MODEL_DIR / "metaclock_chronological_3353_samples_metadata.json"


def export_weights(
    training_path: Path = DEFAULT_TRAINING_DATASET,
    weights_path: Path = WEIGHTS_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Path, Path]:
    """Fit the formal linear-SVR MetaClock and export its equivalent raw formula."""
    training = read_dataset(training_path)
    x = training.loc[:, COMPONENT_CLOCKS].to_numpy(dtype=float)
    y = training["age"].to_numpy(dtype=float)
    model = sklearn_architectures(len(training))["SVR"]
    model.fit(x, y)

    scaler = model.named_steps["scaler"]
    svr = model.named_steps["model"]
    standardized_coefficient = np.asarray(svr.coef_, dtype=float).reshape(-1)
    standardized_intercept = float(np.asarray(svr.intercept_, dtype=float).item())
    raw_coefficient = standardized_coefficient / scaler.scale_
    raw_intercept = float(
        standardized_intercept - np.dot(standardized_coefficient, scaler.mean_ / scaler.scale_)
    )
    pipeline_prediction = np.asarray(model.predict(x), dtype=float)
    raw_prediction = raw_intercept + x @ raw_coefficient
    max_abs_prediction_delta = float(np.max(np.abs(pipeline_prediction - raw_prediction)))
    if max_abs_prediction_delta > 1e-10:
        raise RuntimeError(
            "Exported raw MetaClock formula does not reproduce the fitted pipeline: "
            f"max_abs_delta={max_abs_prediction_delta}."
        )

    weights = pd.DataFrame(
        {
            "feature": COMPONENT_CLOCKS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "standardized_coefficient": standardized_coefficient,
            "raw_coefficient": raw_coefficient,
        }
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(weights_path, index=False)
    metadata = {
        "model": "MetaClock chronological",
        "target": "chronological age",
        "training_samples": int(len(training)),
        "training_cohorts": sorted(training["dataset"].astype(str).unique().tolist()),
        "training_cohort_count": int(training["dataset"].nunique()),
        "feature_count": len(COMPONENT_CLOCKS),
        "features_file_order": weights_path.name,
        "estimator": "StandardScaler followed by SVR(kernel='linear')",
        "svr_parameters": {
            "C": float(svr.C),
            "epsilon": float(svr.epsilon),
            "tol": float(svr.tol),
            "shrinking": bool(svr.shrinking),
            "max_iter": int(svr.max_iter),
        },
        "support_vectors": int(len(svr.support_)),
        "standardized_intercept": standardized_intercept,
        "raw_intercept": raw_intercept,
        "coefficient_formula": "raw_coefficient = standardized_coefficient / scaler_scale",
        "prediction_formula": "raw_intercept + X @ raw_coefficient",
        "training_mae": float(mean_absolute_error(y, pipeline_prediction)),
        "raw_formula_max_abs_prediction_delta": max_abs_prediction_delta,
        "training_input": str(training_path.resolve()),
        "software": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return weights_path, metadata_path


def main() -> None:
    for path in export_weights():
        print(path)


if __name__ == "__main__":
    main()
