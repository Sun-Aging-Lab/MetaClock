"""Prepare the two aligned training inputs used by Figure 3 models."""

from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression

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


def _select_cpgs(
    beta: pd.DataFrame,
    age: pd.Series,
    *,
    mi_features: int,
    mrmr_features: int,
) -> list[str]:
    """Apply the manuscript's standard MI-10k then mRMR-1k feature filter."""
    if not 1 <= mrmr_features <= mi_features:
        raise ValueError("Require 1 <= mrmr_features <= mi_features")
    # Match the retained Figure 3 workflow exactly: MI and mRMR both receive
    # float32 beta values and a float32 target.  Keeping the parquet-native
    # target dtype (often float64) can perturb tied/near-tied feature scores.
    numeric = beta.apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)
    target = pd.to_numeric(age, errors="raise").astype(np.float32)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Direct beta input contains infinite values")
    k_mi = min(mi_features, numeric.shape[1])
    if k_mi == numeric.shape[1]:
        mi_columns = list(numeric.columns)
    else:
        selector = SelectKBest(
            lambda x, y: mutual_info_regression(
                x, y, n_neighbors=3, random_state=0, n_jobs=-1
            ),
            k=k_mi,
        )
        selector.fit(numeric, target)
        mi_columns = list(numeric.columns[selector.get_support()])
    k_mrmr = min(mrmr_features, len(mi_columns))
    if k_mrmr == len(mi_columns):
        return mi_columns
    # mrmr-selection is the maintained reference implementation used by the
    # original Figure 3 workflow; custom mRMR logic would change its semantics.
    from mrmr import mrmr_regression

    selected = mrmr_regression(
        X=numeric.loc[:, mi_columns], y=target, K=k_mrmr, show_progress=False, n_jobs=-1
    )
    result = [str(column) for column in selected]
    if len(result) != k_mrmr or len(set(result)) != k_mrmr:
        raise RuntimeError("mrmr-selection returned an incomplete or duplicate feature set")
    return result


def prepare_dataset_inputs(
    spec: GeoDatasetSpec,
    data_root: Path,
    *,
    cache_dir: Path,
    mi_features: int = 10_000,
    mrmr_features: int = 1_000,
    pyaging_batch_size: int = 32,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write aligned direct-CpG+age and 16-component+age Parquets."""
    beta_path = build_beta_age(spec, data_root)
    component_path = predict_dataset_components(
        spec,
        data_root,
        cache_dir=cache_dir,
        batch_size=pyaging_batch_size,
    )
    root = dataset_dir(data_root, spec)
    direct_output = root / "prepared" / "direct_cpg_age.parquet"
    meta_output = root / "prepared" / "component_predictions_age.parquet"
    provenance_path = root / "prepared" / "preparation.provenance.json"
    reusable = direct_output.is_file() and meta_output.is_file() and provenance_path.is_file()
    if reusable and not force:
        _validate_prepared(direct_output, meta_output, spec)
        return direct_output, meta_output

    beta_age = read_wide_parquet(beta_path)
    component = read_wide_parquet(component_path)
    beta_age.index = beta_age.index.astype(str)
    component.index = component.index.astype(str)
    if not beta_age.index.equals(component.index):
        raise ValueError(f"{spec.gse}: beta and component sample IDs/order differ")
    age = pd.to_numeric(beta_age.pop("age"), errors="raise")
    if not np.allclose(age, pd.to_numeric(component["age"], errors="raise")):
        raise ValueError(f"{spec.gse}: beta and component ages differ")
    selected = _select_cpgs(
        beta_age,
        age,
        mi_features=mi_features,
        mrmr_features=mrmr_features,
    )
    direct = beta_age.loc[:, selected].copy()
    direct.insert(0, "age", age)
    meta = component.loc[:, ["age", *CHRONOLOGICAL_COMPONENTS]].copy()
    atomic_parquet(direct, direct_output)
    atomic_parquet(meta, meta_output)
    (root / "prepared" / "selected_cpgs.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )
    atomic_json(
        {
            "dataset": spec.gse,
            "domain": spec.domain,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "beta_age_input": str(beta_path.resolve()),
            "component_input": str(component_path.resolve()),
            "samples": len(direct),
            "input_cpgs": beta_age.shape[1],
            "mutual_information_features": min(mi_features, beta_age.shape[1]),
            "mrmr_features": len(selected),
            "mutual_information": {
                "implementation": "sklearn.feature_selection.mutual_info_regression",
                "n_neighbors": 3,
                "random_state": 0,
                "n_jobs": -1,
            },
            "mrmr": {"implementation": "mrmr-selection", "K": len(selected), "n_jobs": -1},
            "missing_beta_policy_for_selection": "zero_imputation (Figure 3 legacy policy)",
            "feature_selection_numeric_dtype": "float32 (Figure 3 legacy policy)",
            "runtime": {
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("mrmr-selection", "numpy", "pandas", "scikit-learn")
                    if name != "mrmr-selection" or mrmr_features < mi_features
                },
            },
        },
        provenance_path,
    )
    return direct_output, meta_output


def _validate_prepared(direct_path: Path, meta_path: Path, spec: GeoDatasetSpec) -> None:
    direct = read_wide_parquet(direct_path)
    meta = read_wide_parquet(meta_path)
    direct.index = direct.index.astype(str)
    meta.index = meta.index.astype(str)
    if not direct.index.equals(meta.index) or len(direct) != spec.expected_selected_samples:
        raise ValueError(f"{spec.gse}: prepared sample grids differ")
    if tuple(meta.columns) != ("age", *CHRONOLOGICAL_COMPONENTS):
        raise ValueError(f"{spec.gse}: prepared component columns differ")
    cpgs = [column for column in direct if str(column).startswith("cg")]
    if len(cpgs) != direct.shape[1] - 1 or "age" not in direct:
        raise ValueError(f"{spec.gse}: prepared direct columns are invalid")
    if not np.allclose(direct["age"], meta["age"]):
        raise ValueError(f"{spec.gse}: prepared ages differ")
