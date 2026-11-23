"""Prepare the two aligned training inputs used by Figure 3 models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from repository_paths import relative_path

from .datasets import CHRONOLOGICAL_COMPONENTS
from .geo_data import (
    GeoDatasetSpec,
    atomic_json,
    atomic_parquet,
    build_beta_age,
    dataset_dir,
    read_wide_parquet,
)
from .pyaging_components import predict_dataset_components


def prepare_dataset_inputs(
    spec: GeoDatasetSpec,
    data_root: Path,
    *,
    cache_dir: Path,
    pyaging_batch_size: int = 32,
    force: bool = False,
) -> tuple[Path, Path]:
    """Return the full direct input and write aligned component predictions."""
    beta_path = build_beta_age(spec, data_root)
    component_path = predict_dataset_components(
        spec,
        data_root,
        cache_dir=cache_dir,
        batch_size=pyaging_batch_size,
    )
    root = dataset_dir(data_root, spec)
    meta_output = root / "prepared" / "component_predictions_age.parquet"
    provenance_path = root / "prepared" / "preparation.provenance.json"
    reusable = meta_output.is_file() and provenance_path.is_file()
    if reusable and not force:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("direct_input") == relative_path(beta_path):
            _validate_prepared(beta_path, meta_output, spec)
            return beta_path, meta_output

    beta_age = read_wide_parquet(beta_path, columns=["age"])
    component = read_wide_parquet(component_path)
    beta_age.index = beta_age.index.astype(str)
    component.index = component.index.astype(str)
    if (not beta_age.index.is_unique or not beta_age.index.equals(component.index)
            or len(beta_age) != spec.expected_selected_samples):
        raise ValueError(f"{spec.gse}: beta and component sample IDs/order differ")
    age = pd.to_numeric(beta_age["age"], errors="raise")
    if not np.allclose(age, pd.to_numeric(component["age"], errors="raise")):
        raise ValueError(f"{spec.gse}: beta and component ages differ")
    meta = component.loc[:, ["age", *CHRONOLOGICAL_COMPONENTS]].copy()
    atomic_parquet(meta, meta_output)
    atomic_json(
        {
            "dataset": spec.gse,
            "domain": spec.domain,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "beta_age_input": relative_path(beta_path),
            "component_input": relative_path(component_path),
            "samples": len(beta_age),
            "direct_input": relative_path(beta_path),
            "direct_feature_selection": "performed within training folds",
        },
        provenance_path,
    )
    return beta_path, meta_output


def _validate_prepared(beta_path: Path, meta_path: Path, spec: GeoDatasetSpec) -> None:
    beta_age = read_wide_parquet(beta_path, columns=["age"])
    meta = read_wide_parquet(meta_path)
    beta_age.index = beta_age.index.astype(str)
    meta.index = meta.index.astype(str)
    if not beta_age.index.equals(meta.index) or len(beta_age) != spec.expected_selected_samples:
        raise ValueError(f"{spec.gse}: prepared sample grids differ")
    if tuple(meta.columns) != ("age", *CHRONOLOGICAL_COMPONENTS):
        raise ValueError(f"{spec.gse}: prepared component columns differ")
    if not np.allclose(beta_age["age"], meta["age"]):
        raise ValueError(f"{spec.gse}: prepared ages differ")
