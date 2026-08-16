"""Leakage-safe Elastic Net definitions shared by Figure 4 training workflows."""

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

try:
    from .config import FIGURE4, FIGURE4_COMPONENTS, require_exact_sequence
except ImportError:  # Direct-script/import fallback from this directory.
    from config import FIGURE4, FIGURE4_COMPONENTS, require_exact_sequence


MODEL_CONFIG = FIGURE4["elastic_net"]
RANDOM_SEED = int(MODEL_CONFIG["seed"])
ALPHAS = np.logspace(
    np.log10(float(MODEL_CONFIG["alpha_max"])),
    np.log10(float(MODEL_CONFIG["alpha_min"])),
    int(MODEL_CONFIG["alpha_count"]),
)
CV_FOLDS = int(MODEL_CONFIG["cv_folds"])
MAX_ITER = int(MODEL_CONFIG["max_iter"])
L1_RATIO = float(MODEL_CONFIG["l1_ratio"])


@dataclass(frozen=True)
class ElasticNetFit:
    """Best fold-scaled pipeline plus auditable alpha-search results."""

    model: Pipeline
    alpha: float
    l1_ratio: float
    n_iter: int
    best_cv_mse: float
    alpha_search: tuple[dict[str, float | int], ...]


def validate_component_features(features: list[str]) -> None:
    """Require the exact ordered 17-clock Figure 4 feature manifest."""
    require_exact_sequence(features, FIGURE4_COMPONENTS, "Figure 4 component features")


def fit_scaled_elasticnet_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_jobs: int = -1,
    seed: int = RANDOM_SEED,
) -> ElasticNetFit:
    """Tune alpha with a scaler fit independently inside every CV fold.

    GridSearchCV wraps the complete StandardScaler/ElasticNet pipeline, avoiding
    the preprocessing leakage caused by placing ElasticNetCV after a scaler that
    was fit once on the complete training set.
    """
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be 2-D and aligned to a 1-D target.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data contain non-finite values.")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(l1_ratio=L1_RATIO, max_iter=MAX_ITER, selection="cyclic"),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"elasticnet__alpha": ALPHAS.tolist()},
        scoring=str(MODEL_CONFIG["scoring"]),
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(x, y)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(
            f"Elastic Net alpha search emitted {len(convergence)} convergence warnings; "
            "alpha selection is rejected."
        )
    model = search.best_estimator_
    estimator = model.named_steps["elasticnet"]
    n_iter = int(estimator.n_iter_)
    if n_iter >= MAX_ITER:
        raise RuntimeError("Best Elastic Net estimator reached max_iter.")
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
        model=model,
        alpha=float(search.best_params_["elasticnet__alpha"]),
        l1_ratio=L1_RATIO,
        n_iter=n_iter,
        best_cv_mse=float(-search.best_score_),
        alpha_search=alpha_search,
    )
