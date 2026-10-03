"""Train, save, load, and apply Figure 3 species ENCV component clocks."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from repository_paths import relative_path

from .species_config import direct_input_path, model_dir

TARGET = "age"
SEED = 105
COMPONENT_ALPHAS = np.logspace(0.0, -5.0, 100)
CV_FOLDS = 5
MAX_ITER = 10_000
COMPONENT_CONFIG_ID = "species-encv-v1-alpha1-1e-5-100-cv5-default-l1"
ENCV_ARCHITECTURE = {
    "config_id": COMPONENT_CONFIG_ID,
    "model_type": "elasticnetcv",
    "pipeline": ["SimpleImputer", "StandardScaler", "ElasticNetCV"],
    "imputer_strategy": "median",
    "alphas": COMPONENT_ALPHAS.tolist(),
    "cv_folds": CV_FOLDS,
    "l1_ratio": float(ElasticNetCV().l1_ratio),
    "max_iter": MAX_ITER,
    "seed": SEED,
}


def train_elasticnetcv(source: str, *, n_jobs: int) -> Path:
    """Train one species or pooled ENCV clock and save its complete pipeline."""
    input_path = direct_input_path(source)
    frame = pd.read_parquet(input_path).dropna(subset=[TARGET])
    features = [str(column) for column in frame if str(column).lower().startswith("cg")]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    cv=CV_FOLDS,
                    alphas=COMPONENT_ALPHAS,
                    max_iter=MAX_ITER,
                    random_state=SEED,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    model.fit(frame[features], frame[TARGET].to_numpy(dtype=float))
    output = model_dir("elasticnetcv", source)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"features": features, "model": model, "config_id": COMPONENT_CONFIG_ID},
        output / "model.joblib",
    )
    fitted = model.named_steps["model"]
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                **ENCV_ARCHITECTURE,
                "source": source,
                "input_path": relative_path(input_path),
                "samples": len(frame),
                "features": len(features),
                "selected_alpha": float(fitted.alpha_),
                "alpha_candidates": len(fitted.alphas_),
                "l1_ratio": float(fitted.l1_ratio_),
                "n_jobs": n_jobs,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def _strict_values(frame: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    """Align required CpGs, coerce missing values, and reject infinite inputs."""
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise ValueError(f"{label} input is missing {len(missing)} trained features")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} input contains infinite values")
    return values


def predict_elasticnetcv(frame: pd.DataFrame, source: str) -> np.ndarray:
    """Apply one current ENCV pipeline, preserving trained CpG order."""
    bundle = joblib.load(model_dir("elasticnetcv", source) / "model.joblib")
    if bundle.get("config_id") != COMPONENT_CONFIG_ID:
        raise ValueError(f"ENCV {source} uses an outdated configuration; retrain the component")
    features = list(bundle["features"])
    return np.asarray(
        bundle["model"].predict(_strict_values(frame, features, f"ENCV {source}")),
        dtype=float,
    )
