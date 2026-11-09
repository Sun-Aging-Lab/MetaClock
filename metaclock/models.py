"""Maintained sklearn and LightGBM MetaClock estimator factory."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


_ARCHITECTURES = (
    "elastic_net",
    "lasso",
    "ridge",
    "linear_regression",
    "random_forest",
    "svr",
    "mlp",
    "lightgbm",
)
_ALIASES = {
    "elasticnet": "elastic_net",
    "enet": "elastic_net",
    "linear": "linear_regression",
    "ols": "linear_regression",
    "rf": "random_forest",
    "randomforest": "random_forest",
    "svm": "svr",
    "mlp_regressor": "mlp",
    "neural_network": "mlp",
    "lgbm": "lightgbm",
}


def available_architectures() -> tuple[str, ...]:
    """Return supported canonical architecture names."""
    return _ARCHITECTURES


def normalize_architecture(architecture: str) -> str:
    """Normalize architecture spelling and aliases."""
    if not isinstance(architecture, str):
        raise TypeError("architecture must be a string.")
    name = architecture.strip().lower().replace("-", "_").replace(" ", "_")
    name = _ALIASES.get(name, name)
    if name not in _ARCHITECTURES:
        choices = ", ".join(_ARCHITECTURES)
        raise ValueError(
            f"Unknown architecture {architecture!r}. Choose one of: {choices}."
        )
    return name


def _scaled(model: BaseEstimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def make_estimator(
    architecture: str,
    *,
    random_state: int = 42,
    n_jobs: int | None = 1,
    model_params: Mapping[str, Any] | None = None,
) -> BaseEstimator:
    """Build an unfitted regressor using maintained package implementations.

    ``model_params`` configure the regressor and its alpha/CV search. Elastic
    net, Lasso, and Ridge search complete scaler/regressor pipelines. Scaled
    pipelines use ``scaler`` and ``model`` steps. LightGBM is imported on demand.
    """
    name = normalize_architecture(architecture)
    params = dict(model_params or {})
    alphas = np.logspace(-5, 0, 30)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    if name == "elastic_net":
        search_alphas = np.asarray(params.pop("alphas", alphas), dtype=float)
        if search_alphas.ndim != 1 or not len(search_alphas):
            raise ValueError("Fold-local search requires a nonempty explicit alpha sequence.")
        search_cv = params.pop("cv", cv)
        search_jobs = params.pop("n_jobs", n_jobs)
        search_verbose = params.pop("verbose", 0)
        params.pop("eps", None)
        params.pop("n_alphas", None)
        if isinstance(params.get("precompute"), str) and params["precompute"] == "auto":
            params["precompute"] = False
        ratios = np.atleast_1d(params.pop("l1_ratio", 0.5)).tolist()
        defaults: dict[str, Any] = {"max_iter": 20_000}
        defaults.update(params)
        return GridSearchCV(
            _scaled(ElasticNet(**defaults)),
            {"model__alpha": search_alphas, "model__l1_ratio": ratios},
            cv=search_cv,
            scoring="neg_mean_squared_error",
            n_jobs=search_jobs,
            verbose=search_verbose,
            error_score="raise",
            refit=True,
        )
    if name in {"lasso", "ridge"}:
        if "alpha" in params:
            if "alphas" in params:
                raise ValueError("Supply either alpha or alphas, not both.")
            search_alphas = np.asarray([params.pop("alpha")], dtype=float)
        else:
            search_alphas = np.asarray(params.pop("alphas", alphas), dtype=float)
        if (search_alphas.ndim != 1 or not len(search_alphas)
                or not np.isfinite(search_alphas).all() or np.any(search_alphas < 0)):
            raise ValueError("Alpha search requires a finite, nonnegative explicit sequence.")
        search_cv = params.pop("cv", cv)
        search_jobs = params.pop("n_jobs", n_jobs)
        search_verbose = params.pop("verbose", 0)
        if name == "ridge":
            unsupported = set(params).intersection(
                {"gcv_mode", "store_cv_results", "store_cv_values", "alpha_per_target"}
            )
            if search_cv is None or unsupported:
                raise ValueError(
                    "Fold-local Ridge requires explicit CV; GCV-only options "
                    "gcv_mode, store_cv_results, store_cv_values, and alpha_per_target "
                    "are unsupported."
                )
            scoring = params.pop("scoring", None)
            scoring = "r2" if scoring is None else scoring
            defaults = {}
            model_class = Ridge
        else:
            if {"eps", "n_alphas"}.intersection(params):
                raise ValueError("Fold-local Lasso requires explicit alphas, not eps or n_alphas.")
            if isinstance(params.get("precompute"), str) and params["precompute"] == "auto":
                params["precompute"] = False
            scoring = "neg_mean_squared_error"
            defaults = {"max_iter": 20_000, "random_state": random_state}
            model_class = Lasso
        defaults.update(params)
        return GridSearchCV(
            _scaled(model_class(**defaults)), {"model__alpha": search_alphas},
            cv=search_cv, scoring=scoring, n_jobs=search_jobs,
            verbose=search_verbose, error_score="raise", refit=True,
        )
    if name == "linear_regression":
        defaults = {"n_jobs": n_jobs}
        defaults.update(params)
        return _scaled(LinearRegression(**defaults))
    if name == "random_forest":
        defaults = {"random_state": random_state, "n_jobs": n_jobs}
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if name == "svr":
        defaults = {"kernel": "linear"}
        defaults.update(params)
        return _scaled(SVR(**defaults))
    if name == "mlp":
        defaults = {"max_iter": 2_000, "random_state": random_state}
        defaults.update(params)
        return _scaled(MLPRegressor(**defaults))

    try:
        module = import_module("lightgbm")
        regressor = module.LGBMRegressor
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "architecture='lightgbm' requires LightGBM, which is a core "
            "metaclock-bio dependency. Reinstall the package dependencies."
        ) from exc
    defaults = {"random_state": random_state, "n_jobs": n_jobs, "verbosity": -1}
    defaults.update(params)
    return regressor(**defaults)
