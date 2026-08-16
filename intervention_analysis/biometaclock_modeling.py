"""Leakage-safe Elastic Net fitting for the intervention BioMeta model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(0.0, -5.0, 30)
CV_FOLDS = 5
CV_MAX_ITER = 20_000
REFIT_MAX_ITER = 100_000
L1_RATIO = 0.5
SEED = 0


@dataclass(frozen=True)
class ElasticNetFit:
    """Selected model and the complete auditable alpha search."""

    model: Pipeline
    alpha: float
    l1_ratio: float
    n_iter: int
    best_cv_mse: float
    alpha_search: tuple[dict[str, float | int], ...]


def _raise_on_convergence_warnings(caught: list[warnings.WarningMessage], stage: str) -> None:
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(
            f"Intervention Elastic Net {stage} emitted {len(convergence)} convergence warnings."
        )


def fit_scaled_elastic_net_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_jobs: int,
) -> ElasticNetFit:
    """Tune alpha with fold-local scaling, then refit the selected alpha."""
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.shape[0]:
        raise ValueError("matrix must be 2-D and aligned to a 1-D target")
    if matrix.shape[0] < CV_FOLDS:
        raise ValueError(f"At least {CV_FOLDS} participants are required for five-fold CV")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Training data contain non-finite values")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    l1_ratio=L1_RATIO,
                    max_iter=CV_MAX_ITER,
                    selection="cyclic",
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"elasticnet__alpha": ALPHAS.tolist()},
        scoring="neg_mean_squared_error",
        cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=n_jobs,
        refit=False,
        error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(matrix, target)
    _raise_on_convergence_warnings(caught, "cross-validation")
    alpha = float(search.best_params_["elasticnet__alpha"])
    refit = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=L1_RATIO,
                    max_iter=REFIT_MAX_ITER,
                    selection="cyclic",
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        refit.fit(matrix, target)
    _raise_on_convergence_warnings(caught, "all-participant refit")
    n_iter = int(refit.named_steps["elasticnet"].n_iter_)
    if n_iter >= REFIT_MAX_ITER:
        raise RuntimeError("Intervention Elastic Net refit reached max_iter")
    cv_results: dict[str, Any] = search.cv_results_
    alpha_search = tuple(
        {
            "alpha": float(params["elasticnet__alpha"]),
            "mean_cv_mse": float(-mean_score),
            "sd_cv_mse": float(sd_score),
            "rank": int(rank),
        }
        for params, mean_score, sd_score, rank in zip(
            cv_results["params"],
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
            cv_results["rank_test_score"],
            strict=True,
        )
    )
    return ElasticNetFit(
        model=refit,
        alpha=alpha,
        l1_ratio=L1_RATIO,
        n_iter=n_iter,
        best_cv_mse=float(-search.best_score_),
        alpha_search=alpha_search,
    )
