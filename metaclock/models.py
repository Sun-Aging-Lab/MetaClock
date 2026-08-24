"""Maintained sklearn and LightGBM MetaClock estimator factory."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
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


def _scaled(model: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def make_estimator(
    architecture: str,
    *,
    random_state: int = 42,
    n_jobs: int | None = 1,
    model_params: Mapping[str, Any] | None = None,
):
    """Build an unfitted regressor using maintained package implementations.

    ``model_params`` are passed directly to the final estimator. Scaled linear
    models are returned as sklearn Pipelines with ``scaler`` and ``model``
    steps. LightGBM is imported only when requested.
    """
    name = normalize_architecture(architecture)
    params = dict(model_params or {})
    alphas = np.logspace(-5, 0, 30)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    if name == "elastic_net":
        defaults: dict[str, Any] = {
            "alphas": alphas,
            "cv": cv,
            "max_iter": 20_000,
            "n_jobs": n_jobs,
        }
        defaults.update(params)
        return _scaled(ElasticNetCV(**defaults))
    if name == "lasso":
        defaults = {
            "alphas": alphas,
            "cv": cv,
            "max_iter": 20_000,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        defaults.update(params)
        return _scaled(LassoCV(**defaults))
    if name == "ridge":
        defaults = {"alphas": alphas, "cv": cv}
        defaults.update(params)
        return _scaled(RidgeCV(**defaults))
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
