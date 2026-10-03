"""Public MetaClock fitting API."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ._version import __version__
from .components import (
    InputData,
    PyAgingTransformer,
    load_anndata,
    normalize_component_models,
)
from .models import make_estimator, normalize_architecture

if TYPE_CHECKING:
    from anndata import AnnData


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _extract_target(adata: AnnData, target: str) -> np.ndarray:
    if not isinstance(target, str) or not target.strip():
        raise TypeError("target must be a non-empty adata.obs column name.")
    if target not in adata.obs.columns:
        raise KeyError(f"AnnData obs does not contain target column {target!r}.")
    series = pd.to_numeric(adata.obs[target], errors="coerce")
    values = series.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        affected = adata.obs_names[~np.isfinite(values)].astype(str).tolist()[:5]
        raise ValueError(
            f"Target column {target!r} contains missing or non-finite values; "
            f"first affected samples: {affected}"
        )
    return values


def _safe_params(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_params(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_params(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def fit(
    input_data: InputData,
    component_models: Sequence[str] | str,
    architecture: str,
    *,
    target: str = "age",
    pyaging_dir: str | PathLike[str] | None = "pyaging_data",
    batch_size: int = 1024,
    clean: bool = True,
    verbose: bool = False,
    copy: bool = False,
    random_state: int = 42,
    n_jobs: int | None = 1,
    model_params: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit a MetaClock from a pyaging-standard AnnData input.

    Parameters
    ----------
    input_data
        An ``AnnData`` object or an ``.h5ad`` path. Rows are samples, columns
        are pyaging features, and ``adata.obs[target]`` supplies the response.
    component_models
        One or more pyaging clock names. Their prediction order is preserved.
    architecture
        A name returned by ``metaclock.available_architectures()``.
    target
        Numeric ``adata.obs`` target column. Defaults to ``"age"``.
    pyaging_dir
        PyAging model-cache directory, or ``None`` to use PyAging's default.
    batch_size, clean, verbose
        Values forwarded to PyAging component-clock inference.
    copy
        If true, copy in-memory ``AnnData`` before PyAging inference. With the
        default false, PyAging may update the supplied object in place.
    random_state
        Seed used by regressors and shuffled cross-validation where supported.
    n_jobs
        Worker count forwarded to regressors that support parallel execution.
    model_params
        Overrides passed directly to the final regression estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline. Calling ``predict()`` with another pyaging-standard
        ``AnnData`` object or ``.h5ad`` path reruns the same component clocks
        before applying the fitted regressor.
    """
    adata = load_anndata(input_data)
    components = normalize_component_models(component_models)
    architecture_name = normalize_architecture(architecture)
    if target.lower() in components:
        raise ValueError("target cannot also be a component-model output column.")
    y = _extract_target(adata, target)

    transformer = PyAgingTransformer(
        components,
        pyaging_dir=pyaging_dir,
        batch_size=batch_size,
        clean=clean,
        verbose=verbose,
        copy=copy,
    )
    estimator = make_estimator(
        architecture_name,
        random_state=random_state,
        n_jobs=n_jobs,
        model_params=model_params,
    )
    model = Pipeline([("components", transformer), ("regressor", estimator)])
    model.fit(adata, y)

    model.metaclock_architecture_ = architecture_name
    model.metaclock_component_models_ = components
    model.metaclock_target_ = target
    model.metaclock_provenance_ = {
        "package_versions": {
            "python": platform.python_version(),
            "metaclock": __version__,
            "pyaging": _package_version("pyaging"),
            "anndata": _package_version("anndata"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "lightgbm": _package_version("lightgbm")
            if architecture_name == "lightgbm"
            else None,
        },
        "fit_parameters": {
            "target": target,
            "component_models": list(components),
            "architecture": architecture_name,
            "pyaging_dir": None if pyaging_dir is None else str(pyaging_dir),
            "batch_size": batch_size,
            "clean": clean,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "model_params": _safe_params(dict(model_params or {})),
        },
        "training_samples": int(adata.n_obs),
        "component_missing_feature_percent": dict(transformer.missing_feature_percent_),
    }
    return model
