"""Fold-local feature selection and model searches for Figure 3."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Memory, effective_n_jobs, parallel_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


_GRID = json.loads((Path(__file__).parent / "assets" / "search_grid.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ElasticNetConfig:
    cv_folds: int = 5
    alphas: tuple[float, ...] = tuple(_GRID["alpha"])
    l1_ratio: float = 0.5
    max_iter: int = 20_000
    tolerance: float = 1e-4
    n_jobs: int = field(default_factory=lambda: int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


@dataclass(frozen=True)
class GeneralizationConfig:
    split_seed: int = 105
    repeat_seed: int = 42
    train_fraction: float = 0.8
    test_reserve: int = 10
    n_training_sizes: int = 30
    n_resamples: int = 50
    elastic_net: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    mi_features: int = 1000
    svr_c_values: tuple[float, ...] = tuple(_GRID["C"])
    svr_epsilon_values: tuple[float, ...] = tuple(_GRID["epsilon"])
    svr_c_search_epsilon: float = 0.001
    svr_cv_folds: int = 3
    svr_tolerance: float = 1e-3
    svr_max_iter: int = -1


class TrainingCpGs(TransformerMixin, BaseEstimator):
    """Score nonempty CpGs on training rows and return the original selected values."""

    def __init__(self, matrix_path: Path, k: int, n_jobs: int) -> None:
        self.matrix_path = matrix_path
        self.k = k
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingCpGs:
        matrix = np.load(self.matrix_path, mmap_mode="r")
        values = np.asarray(matrix[X[:, 0]], dtype=np.float32)
        valid = ~np.isnan(values).all(axis=0)
        if valid.sum() < self.k:
            raise ValueError(f"Need {self.k} nonempty training CpGs, found {valid.sum()}")
        candidates = values if valid.all() else values[:, valid]
        if np.isnan(candidates).any():
            candidates = SimpleImputer(strategy="median").fit_transform(candidates)
        score = partial(mutual_info_regression, n_neighbors=3, random_state=0, n_jobs=self.n_jobs)
        selector = SelectKBest(score_func=score, k=self.k).fit(candidates, y)
        self.selected_indices_ = np.flatnonzero(valid)[selector.get_support(indices=True)]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        matrix = np.load(self.matrix_path, mmap_mode="r")
        return np.ascontiguousarray(matrix[np.ix_(X[:, 0], self.selected_indices_)], dtype=np.float32)


def _fit_search(
    label: str, x: np.ndarray, y: np.ndarray, config: GeneralizationConfig,
    domain: str, matrix_path: Path | None, directory: Path,
) -> tuple[GridSearchCV, dict[str, float]]:
    settings = config.elastic_net
    jobs = effective_n_jobs(settings.n_jobs)
    is_en = label in ("EN", "Meta EN")
    steps = []
    if matrix_path is not None:
        steps.append(("mi", TrainingCpGs(matrix_path, config.mi_features, jobs)))
    if domain == "species" and label == "SVM":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)))
    elif domain != "species" or label != "Meta SVM":
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    steps.append(("scaler", StandardScaler()))
    if is_en:
        model = ElasticNet(l1_ratio=settings.l1_ratio, tol=settings.tolerance, max_iter=settings.max_iter)
    else:
        model = SVR(kernel="linear", tol=config.svr_tolerance, max_iter=config.svr_max_iter)
    memory = Memory(directory / "folds", verbose=0) if matrix_path is not None else None
    pipeline = Pipeline([*steps, ("model", model)], memory=memory)
    cv = KFold(n_splits=settings.cv_folds if is_en else config.svr_cv_folds, shuffle=False)
    if memory is not None:
        # Cache each training fold before parallel parameter fits request the same MI scores.
        preprocess = Pipeline([*steps, ("end", "passthrough")], memory=memory)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            for train, _ in cv.split(x):
                clone(preprocess).fit(x[train], y[train])
            clone(preprocess).fit(x, y)

    options = dict(
        estimator=pipeline,
        cv=cv,
        n_jobs=jobs,
        pre_dispatch=jobs,
        error_score="raise",
        scoring="neg_mean_squared_error" if is_en else "neg_mean_absolute_error",
    )
    metrics: dict[str, float] = {}
    with parallel_config(backend="threading"):
        if is_en:
            grid = {"model__alpha": settings.alphas}
        else:
            c_grid = {"model__C": config.svr_c_values, "model__epsilon": [config.svr_c_search_epsilon]}
            c_search = GridSearchCV(param_grid=c_grid, refit=False, **options).fit(x, y)
            if not np.isfinite(c_search.cv_results_["mean_test_score"]).all():
                raise ValueError("Nonfinite scores in SVR C search")
            metrics["c_search_best_cv_mae"] = float(-c_search.best_score_)
            grid = {"model__C": [c_search.best_params_["model__C"]],
                    "model__epsilon": config.svr_epsilon_values}
        search = GridSearchCV(param_grid=grid, refit=True, **options).fit(x, y)
    if not np.isfinite(search.cv_results_["mean_test_score"]).all():
        raise ValueError(f"Nonfinite {label} cross-validation scores")
    return search, metrics


def fit_model(
    label: str, train: pd.DataFrame, test: pd.DataFrame, config: GeneralizationConfig,
    *, domain: str = "array", matrix_path: Path | None = None,
) -> dict[str, object]:
    """Tune on one draw, refit on its training rows, and score its held-out samples."""
    if label not in ("EN", "SVM", "Meta EN", "Meta SVM") or domain not in ("array", "tissue", "species"):
        raise ValueError(f"Unknown model or domain: {label}, {domain}")
    if not train.columns.equals(test.columns):
        raise ValueError("Training and test columns differ")
    direct = label in ("EN", "SVM")
    y = train["age"].to_numpy(dtype=np.float64)
    with TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR")) as tmp, threadpool_limits(limits=1):
        directory = Path(tmp)
        if matrix_path is not None:
            if not direct or list(train.columns.drop("age")) != ["row"]:
                raise ValueError("A CpG matrix requires a direct model and row indices")
            n_rows = np.load(matrix_path, mmap_mode="r").shape[0]
            for frame in (train, test):
                rows = frame["row"]
                if not pd.api.types.is_integer_dtype(rows) or not rows.between(0, n_rows - 1).all():
                    raise ValueError("Invalid CpG matrix row index")
            x, xt = (frame[["row"]].to_numpy(dtype=np.int64) for frame in (train, test))
        elif direct:
            matrix_path = directory / "cpgs.npy"
            values = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32,
                                               shape=(len(train) + len(test), train.shape[1] - 1))
            values[:len(train)] = train.drop(columns="age").to_numpy(dtype=np.float32)
            values[len(train):] = test.drop(columns="age").to_numpy(dtype=np.float32)
            del values
            rows = np.arange(len(train) + len(test))[:, None]
            x, xt = rows[:len(train)], rows[len(train):]
        else:
            x, xt = (np.ascontiguousarray(frame.drop(columns="age"), dtype=np.float64)
                     for frame in (train, test))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search, metrics = _fit_search(label, x, y, config, domain, matrix_path, directory)
        model = search.best_estimator_.named_steps["model"]
        prediction = np.asarray(search.predict(xt), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite {label} predictions")
    record: dict[str, object] = {
        "model": label,
        "mae": float(mean_absolute_error(test["age"], prediction)),
        "prediction_sd": float(np.std(prediction, ddof=1)) if len(prediction) > 1 else 0.0,
        "selected_alpha": np.nan,
        "n_iter": 0,
        "nonzero_coefficients": 0,
        "support_vectors": 0,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
    }
    if isinstance(model, ElasticNet):
        record.update(selected_alpha=float(model.alpha), best_cv_mse=float(-search.best_score_),
                      n_iter=int(model.n_iter_),
                      nonzero_coefficients=int(np.count_nonzero(model.coef_)))
        if effective_n_jobs(config.elastic_net.n_jobs) != 1:
            record["convergence_warnings"] = None
    else:
        if model.fit_status_ != 0:
            raise ValueError("SVR refit did not converge")
        record.update(selected_C=float(model.C), selected_epsilon=float(model.epsilon),
                      best_cv_mae=float(-search.best_score_), support_vectors=int(model.support_.size),
                      **metrics)
    return record
