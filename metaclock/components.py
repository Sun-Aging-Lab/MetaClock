"""pyaging adapter for generating MetaClock component-model features."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


InputData = Any | str | PathLike[str]


def normalize_component_models(
    component_models: Sequence[str] | str,
) -> tuple[str, ...]:
    """Return non-empty, unique, lower-case pyaging clock names."""
    if isinstance(component_models, str):
        raw_models = [component_models]
    else:
        try:
            raw_models = list(component_models)
        except TypeError as exc:
            raise TypeError(
                "component_models must be a clock name or a sequence of names."
            ) from exc

    normalized: list[str] = []
    for model in raw_models:
        if not isinstance(model, str):
            raise TypeError("Every component model name must be a string.")
        name = model.strip().lower()
        if not name:
            raise ValueError("Component model names cannot be empty.")
        normalized.append(name)

    if not normalized:
        raise ValueError("At least one component model is required.")
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"component_models contains duplicates: {duplicates}")
    return tuple(normalized)


def load_anndata(input_data: InputData):
    """Load a pyaging-standard AnnData object from memory or an .h5ad file."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise ImportError("anndata is required to load MetaClock input data.") from exc

    if isinstance(input_data, ad.AnnData):
        adata = input_data
    elif isinstance(input_data, (str, PathLike)):
        path = Path(input_data).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AnnData input file does not exist: {path}")
        try:
            adata = ad.read_h5ad(path)
        except Exception as exc:
            raise ValueError(f"Could not read AnnData .h5ad input: {path}") from exc
    else:
        raise TypeError(
            "input_data must be an anndata.AnnData object or a path to an .h5ad file."
        )

    if adata.n_obs == 0:
        raise ValueError("AnnData input has no samples.")
    if adata.n_vars == 0:
        raise ValueError("AnnData input has no features.")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData obs_names must be unique.")
    if not adata.var_names.is_unique:
        raise ValueError("AnnData var_names must be unique.")
    return adata.copy() if adata.is_view else adata


def _pyaging_predict_age():
    try:
        import pyaging
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise ImportError(
            "pyaging is required to calculate component-model predictions."
        ) from exc

    for namespace_name in ("pred", "predict"):
        namespace = getattr(pyaging, namespace_name, None)
        function = getattr(namespace, "predict_age", None)
        if callable(function):
            return function
    function = getattr(pyaging, "predict_age", None)
    if callable(function):
        return function
    raise ImportError("The installed pyaging version does not expose predict_age().")


class PyAgingTransformer(TransformerMixin, BaseEstimator):
    """Transform a pyaging-standard AnnData object into component predictions.

    The transformer deliberately delegates clock loading, feature alignment,
    preprocessing, inference, and postprocessing to pyaging. Its output column
    order always matches ``component_models``.
    """

    def __init__(
        self,
        component_models: Sequence[str] | str,
        *,
        pyaging_dir: str | PathLike[str] | None = "pyaging_data",
        batch_size: int = 1024,
        clean: bool = True,
        verbose: bool = False,
        copy: bool = False,
    ) -> None:
        self.component_models = component_models
        self.pyaging_dir = pyaging_dir
        self.batch_size = batch_size
        self.clean = clean
        self.verbose = verbose
        self.copy = copy

    def fit(self, X: InputData, y: Any = None) -> PyAgingTransformer:
        del X, y
        self.component_models_ = normalize_component_models(self.component_models)
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.n_features_out_ = len(self.component_models_)
        return self

    def transform(self, X: InputData) -> pd.DataFrame:
        check_is_fitted(self, "component_models_")
        adata = load_anndata(X)
        if self.copy:
            adata = adata.copy()

        kwargs: dict[str, Any] = {
            "clock_names": list(self.component_models_),
            "batch_size": self.batch_size,
            "clean": self.clean,
            "verbose": self.verbose,
        }
        if self.pyaging_dir is not None:
            kwargs["dir"] = str(self.pyaging_dir)

        predicted = _pyaging_predict_age()(adata, **kwargs)
        if predicted is not None:
            adata = predicted

        column_lookup: dict[str, Any] = {}
        for column in adata.obs.columns:
            lowered = str(column).lower()
            if lowered in column_lookup and column_lookup[lowered] != column:
                raise ValueError(
                    f"AnnData obs has ambiguous case-insensitive column {lowered!r}."
                )
            column_lookup[lowered] = column
        missing = [name for name in self.component_models_ if name not in column_lookup]
        if missing:
            raise RuntimeError(
                f"pyaging did not create component output columns: {missing}"
            )

        ordered_columns = [column_lookup[name] for name in self.component_models_]
        frame = adata.obs.loc[:, ordered_columns].apply(pd.to_numeric, errors="coerce")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            bad_rows, bad_columns = np.where(~np.isfinite(values))
            examples = [
                (str(adata.obs_names[row]), self.component_models_[column])
                for row, column in zip(bad_rows[:5], bad_columns[:5])
            ]
            raise ValueError(
                "pyaging component outputs contain missing or non-finite values; "
                f"first affected (sample, component) pairs: {examples}"
            )

        self.missing_feature_percent_ = {
            name: float(adata.uns.get(f"{name}_percent_na", np.nan))
            for name in self.component_models_
        }
        return pd.DataFrame(
            values,
            index=adata.obs_names.copy(),
            columns=self.get_feature_names_out(),
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "component_models_")
        return np.asarray(self.component_models_, dtype=object)


def predict_components(
    input_data: InputData,
    component_models: Sequence[str] | str,
    *,
    pyaging_dir: str | PathLike[str] | None = "pyaging_data",
    batch_size: int = 1024,
    clean: bool = True,
    verbose: bool = False,
    copy: bool = False,
) -> pd.DataFrame:
    """Return ordered pyaging component predictions as a DataFrame."""
    adata = load_anndata(input_data)
    transformer = PyAgingTransformer(
        component_models,
        pyaging_dir=pyaging_dir,
        batch_size=batch_size,
        clean=clean,
        verbose=verbose,
        copy=copy,
    )
    values = transformer.fit_transform(adata)
    return pd.DataFrame(
        values,
        index=adata.obs_names.copy(),
        columns=transformer.get_feature_names_out(),
    )
