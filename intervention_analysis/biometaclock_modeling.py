"""Fit the intervention BioMeta model using its canonical two-stage workflow.

The current workflow intentionally places ``StandardScaler`` outside
``ElasticNetCV``.  The scaler therefore sees the complete training matrix
before the five-fold alpha search.  This module uses that behavior for
bit-reproducible model artifacts; the selected alpha is then fitted again with
a fresh scaler and a plain ``ElasticNet`` on all participants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(0.0, -5.0, 30)
CV_FOLDS = 5
CV_MAX_ITER = 10_000
REFIT_MAX_ITER = 100_000
L1_RATIO = 0.5
SEED = 0


@dataclass(frozen=True)
class ElasticNetFit:
    """Selected model and the complete auditable alpha search."""

    model: object
    alpha: float
    l1_ratio: float
    n_iter: int
    best_cv_mse: float
    cv_convergence_warnings: int
    refit_convergence_warnings: int
    alpha_search: tuple[dict[str, float | int], ...]


def _convergence_warning_count(caught: list[warnings.WarningMessage]) -> int:
    """Count warnings without changing the canonical CV warning policy."""
    return sum(issubclass(item.category, ConvergenceWarning) for item in caught)


def fit_scaled_elastic_net_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_jobs: int,
) -> ElasticNetFit:
    """Run canonical CV alpha selection followed by a checked full refit.

    CV warnings are recorded but do not invalidate the alpha search. Warnings
    from the final selected-alpha refit, or a refit that reaches
    ``REFIT_MAX_ITER``, are fatal because the saved model must be converged.
    """
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.shape[0]:
        raise ValueError("matrix must be 2-D and aligned to a 1-D target")
    if matrix.shape[0] < CV_FOLDS:
        raise ValueError(f"At least {CV_FOLDS} participants are required for five-fold CV")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Training data contain non-finite values")
    search = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            alphas=ALPHAS,
            cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED),
            max_iter=CV_MAX_ITER,
            n_jobs=max(1, int(n_jobs)),
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(matrix, target)
    cv_warning_count = _convergence_warning_count(caught)
    elasticnet_cv = search.named_steps["elasticnetcv"]
    alpha = float(elasticnet_cv.alpha_)
    refit = make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=alpha,
            l1_ratio=float(elasticnet_cv.l1_ratio_),
            max_iter=REFIT_MAX_ITER,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        refit.fit(matrix, target)
    refit_warning_count = _convergence_warning_count(caught)
    n_iter = int(refit.named_steps["elasticnet"].n_iter_)
    if refit_warning_count or n_iter >= REFIT_MAX_ITER:
        raise RuntimeError(
            "Selected-alpha ElasticNet refit did not converge: "
            f"warnings={refit_warning_count}, n_iter={n_iter}, "
            f"max_iter={REFIT_MAX_ITER}."
        )

    # ElasticNetCV exposes the exact fold MSE path used to choose alpha.
    # Keep one row per requested alpha in the model-selection audit.
    mse_path = np.asarray(elasticnet_cv.mse_path_, dtype=float)
    if mse_path.ndim == 3:
        mse_path = mse_path[0]
    if mse_path.shape[0] != len(elasticnet_cv.alphas_):
        raise RuntimeError("Unexpected ElasticNetCV mse_path_ shape")
    mean_mse = mse_path.mean(axis=1)
    sd_mse = mse_path.std(axis=1)
    ranks = np.argsort(np.argsort(mean_mse, kind="mergesort"), kind="mergesort") + 1
    alpha_search = tuple(
        {
            "alpha": float(alpha_value),
            "mean_cv_mse": float(mean_value),
            "sd_cv_mse": float(sd_value),
            "rank": int(rank),
        }
        for alpha_value, mean_value, sd_value, rank in zip(
            elasticnet_cv.alphas_, mean_mse, sd_mse, ranks, strict=True
        )
    )
    return ElasticNetFit(
        model=refit,
        alpha=alpha,
        l1_ratio=float(elasticnet_cv.l1_ratio_),
        n_iter=n_iter,
        best_cv_mse=float(mean_mse[np.argmin(mean_mse)]),
        cv_convergence_warnings=cv_warning_count,
        refit_convergence_warnings=refit_warning_count,
        alpha_search=alpha_search,
    )
