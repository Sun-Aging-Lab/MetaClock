"""Train, save, load, and apply Figure 3 species ENCV component clocks."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generalization.estimators import ElasticNetConfig
from repository_paths import relative_path

from .species_config import INPUT_CONFIG_ID, direct_input_path, model_dir

TARGET = "age"
SEED = 105
COMPONENT_ALPHAS = np.asarray(ElasticNetConfig().alphas)
CV_FOLDS = 5
MAX_ITER = 20_000
MI_K = 1_000
MI_NEIGHBORS = 3
MI_RANDOM_STATE = 0
COMPONENT_CONFIG_ID = "species-dogtop10-fold-local-mi1000-encv30-cv5-maxiter20000-v2"
ENCV_ARCHITECTURE = {
    "config_id": COMPONENT_CONFIG_ID,
    "model_type": "elasticnetcv",
    "search": "GridSearchCV",
    "pipeline": ["SimpleImputer", "StandardScaler", "SelectKBest(mutual_info_regression)", "ElasticNet"],
    "scoring": "neg_mean_squared_error",
    "fold_local_preprocessing": True,
    "keep_empty_features": True,
    "imputer_strategy": "median",
    "mi_k": MI_K,
    "mi_neighbors": MI_NEIGHBORS,
    "mi_random_state": MI_RANDOM_STATE,
    "alphas": COMPONENT_ALPHAS.tolist(),
    "cv_folds": CV_FOLDS,
    "l1_ratio": 0.5,
    "tol": 1e-4,
    "selection": "cyclic",
    "max_iter": MAX_ITER,
    "seed": SEED,
}


def train_elasticnetcv(source: str, *, n_jobs: int) -> Path:
    """Train one fold-local MI-1000 ENCV component and save its pipeline."""
    input_path = direct_input_path(source)
    frame = pd.read_parquet(input_path).dropna(subset=[TARGET])
    if frame.attrs.get("input_config_id") != INPUT_CONFIG_ID:
        raise ValueError("Species inputs use an outdated configuration; prepare them again")
    features = [str(column) for column in frame if str(column).lower().startswith("cg")]
    if len(features) < MI_K:
        raise ValueError(f"{source}: requires at least {MI_K} CpGs for MI selection")
    workers = int(n_jobs) if int(n_jobs) > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if workers < 1:
        raise ValueError("n_jobs must be positive")
    score_function = partial(
        mutual_info_regression,
        n_neighbors=MI_NEIGHBORS,
        random_state=MI_RANDOM_STATE,
        n_jobs=workers,
    )
    cache_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    with tempfile.TemporaryDirectory(prefix=f"species_component_{source}_", dir=cache_parent) as cache:
        memory = joblib.Memory(cache, verbose=0)
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("mi", SelectKBest(score_func=score_function, k=MI_K)),
            ("model", ElasticNet(l1_ratio=0.5, max_iter=MAX_ITER, random_state=SEED,
                                  selection="cyclic", tol=1e-4)),
        ]
        values = np.ascontiguousarray(frame.loc[:, features].to_numpy(dtype=np.float32))
        target = frame[TARGET].to_numpy(dtype=np.float64)
        cv = KFold(n_splits=CV_FOLDS, shuffle=False)
        preprocessing = Pipeline([*steps[:-1], ("end", "passthrough")], memory=memory)
        for train_index, _ in cv.split(values):
            clone(preprocessing).fit(values[train_index], target[train_index])
        clone(preprocessing).fit(values, target)
        model = GridSearchCV(
            Pipeline(steps, memory=memory), {"model__alpha": COMPONENT_ALPHAS},
            cv=cv, scoring="neg_mean_squared_error", n_jobs=workers,
            pre_dispatch=workers, error_score="raise",
        )
        with parallel_config(backend="threading"):
            model.fit(values, target)
        model.estimator.memory = None
        model.best_estimator_.memory = None
    output = model_dir("elasticnetcv", source)
    output.mkdir(parents=True, exist_ok=True)
    fitted = model.best_estimator_.named_steps["model"]
    indices = model.best_estimator_.named_steps["mi"].get_support(indices=True).tolist()
    selected_cpgs = [features[index] for index in indices]
    joblib.dump(
        {
            "features": features,
            "model": model,
            "config_id": COMPONENT_CONFIG_ID,
            "selected_feature_indices": indices,
            "selected_features": selected_cpgs,
        },
        output / "model.joblib",
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                **ENCV_ARCHITECTURE,
                "source": source,
                "input_path": relative_path(input_path),
                "samples": len(frame),
                "input_features": len(features),
                "selected_features": MI_K,
                "selected_feature_indices": indices,
                "selected_cpgs": selected_cpgs,
                "selected_alpha": float(fitted.alpha),
                "alpha_candidates": len(COMPONENT_ALPHAS),
                "l1_ratio": float(fitted.l1_ratio),
                "n_jobs": workers,
                "gridsearch_n_jobs": workers,
                "input_dtype": "float32",
                "target_dtype": "float64",
                "refit_n_iter": int(fitted.n_iter_),
                "refit_dual_gap": float(fitted.dual_gap_),
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


def _strict_values(frame: pd.DataFrame, features: list[str], label: str) -> np.ndarray:
    """Align required CpGs, coerce missing values, and reject infinite inputs."""
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise ValueError(f"{label} input is missing {len(missing)} trained features")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} input contains infinite values")
    return np.ascontiguousarray(values.to_numpy(dtype=np.float32), dtype=np.float32)


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
