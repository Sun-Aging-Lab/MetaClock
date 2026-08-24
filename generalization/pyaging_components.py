"""Run the 16 chronological PyAging clocks once per Figure 3 dataset."""

from __future__ import annotations

import fcntl
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .datasets import CHRONOLOGICAL_COMPONENTS
from .geo_data import (
    GeoDatasetSpec,
    atomic_json,
    atomic_parquet,
    build_beta_age,
    dataset_dir,
    read_wide_parquet,
)

PredictionRunner = Callable[
    [pd.DataFrame, tuple[str, ...], Path, int], tuple[pd.DataFrame, dict[str, object]]
]


def _package_versions() -> dict[str, str]:
    packages = ("anndata", "numpy", "pandas", "pyaging", "torch")
    return {name: importlib.metadata.version(name) for name in packages}


def _pyaging_runner(
    beta: pd.DataFrame,
    clocks: tuple[str, ...],
    cache_dir: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Call the maintained PyAging API and retain missing-feature diagnostics."""
    import anndata as ad
    import pyaging as pya

    cache_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    percent_na: dict[str, list[float]] = {clock: [] for clock in clocks}
    missing_features: dict[str, list[str]] = {}
    # A single process owns the cache while models are downloaded/read.  This avoids
    # partial .pt files when two datasets start on the same node simultaneously.
    with (cache_dir / ".pyaging.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for start in range(0, len(beta), batch_size):
            block = beta.iloc[start : start + batch_size]
            # Figure 3's original workflow supplied zero for unavailable measurements.
            # Keep that policy explicit and record the input missingness below.
            adata = ad.AnnData(block.fillna(0.0).to_numpy(dtype=np.float32))
            adata.obs_names = block.index.astype(str)
            adata.var_names = block.columns.astype(str)
            pya.pred.predict_age(
                adata,
                list(clocks),
                dir=str(cache_dir),
                batch_size=batch_size,
                clean=True,
                verbose=False,
            )
            missing = [clock for clock in clocks if clock not in adata.obs]
            if missing:
                raise RuntimeError(f"PyAging did not return requested clocks: {missing}")
            predictions.append(adata.obs.loc[:, list(clocks)].copy())
            for clock in clocks:
                percent_key = f"{clock}_percent_na"
                missing_key = f"{clock}_missing_features"
                if percent_key in adata.uns:
                    percent_na[clock].append(float(adata.uns[percent_key]))
                observed_missing = [str(value) for value in adata.uns.get(missing_key, [])]
                if clock in missing_features and missing_features[clock] != observed_missing:
                    raise RuntimeError(f"PyAging feature diagnostics changed by batch: {clock}")
                missing_features[clock] = observed_missing
        fcntl.flock(lock, fcntl.LOCK_UN)
    diagnostics = {
        "pyaging_missing_feature_percent": {
            clock: sorted({float(value) for value in values if np.isfinite(value)})
            for clock, values in percent_na.items()
        },
        "pyaging_missing_features": missing_features,
    }
    return pd.concat(predictions, axis=0), diagnostics


def predict_dataset_components(
    spec: GeoDatasetSpec,
    data_root: Path,
    *,
    cache_dir: Path,
    batch_size: int = 32,
    force: bool = False,
    runner: PredictionRunner | None = None,
) -> Path:
    """Create one validated samples-by-16 component prediction table."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    beta_path = build_beta_age(spec, data_root)
    root = dataset_dir(data_root, spec)
    output = root / "intermediate" / "component_predictions.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    if output.is_file() and provenance_path.is_file() and not force:
        frame = read_wide_parquet(output)
        _validate_predictions(frame, spec)
        return output

    beta_age = read_wide_parquet(beta_path)
    beta_age.index = beta_age.index.astype(str)
    age = pd.to_numeric(beta_age.pop("age"), errors="raise")
    beta = beta_age.apply(pd.to_numeric, errors="coerce")
    if np.isinf(beta.to_numpy(dtype=float)).any():
        raise ValueError(f"{spec.gse}: beta matrix contains infinite values")
    missing_values = int(beta.isna().sum().sum())
    prediction_runner = runner or _pyaging_runner
    prediction, diagnostics = prediction_runner(
        beta, CHRONOLOGICAL_COMPONENTS, cache_dir.expanduser().resolve(), batch_size
    )
    prediction.index = prediction.index.astype(str)
    prediction = prediction.reindex(beta.index)
    prediction = prediction.loc[:, list(CHRONOLOGICAL_COMPONENTS)].apply(
        pd.to_numeric, errors="raise"
    )
    prediction.insert(0, "age", age.to_numpy(dtype=float))
    _validate_predictions(prediction, spec)
    atomic_parquet(prediction, output)
    payload: dict[str, object] = {
        "dataset": spec.gse,
        "domain": spec.domain,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "beta_age_input": str(beta_path.resolve()),
        "samples": len(prediction),
        "components": list(CHRONOLOGICAL_COMPONENTS),
        "batch_size": batch_size,
        "input_missing_beta_values": missing_values,
        "input_missing_beta_policy": "zero_imputation_before_PyAging (Figure 3 legacy policy)",
        "model_cache": str(cache_dir.expanduser().resolve()),
        "runtime": {"python": platform.python_version()},
        **diagnostics,
    }
    if runner is None:
        payload["runtime"]["packages"] = _package_versions()  # type: ignore[index]
    atomic_json(payload, provenance_path)
    return output


def _validate_predictions(frame: pd.DataFrame, spec: GeoDatasetSpec) -> None:
    expected = ("age", *CHRONOLOGICAL_COMPONENTS)
    if tuple(frame.columns) != expected:
        raise ValueError(f"{spec.gse}: component columns differ from {expected}")
    if len(frame) != spec.expected_selected_samples or not frame.index.is_unique:
        raise ValueError(f"{spec.gse}: component prediction sample grid is invalid")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{spec.gse}: component predictions contain non-finite values")
