"""Shared model fitting and validated shard I/O for Figure 3 generalization."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error

from repository_paths import ensure_repo_path, relative_path

from .estimators import ElasticNetConfig, GeneralizationConfig, fit_model
from .geo_data import THRIFT_CONTAINER_LIMIT, THRIFT_STRING_LIMIT

TARGET = "age"
MODEL_NAMES = ("EN", "Meta EN", "SVM", "Meta SVM", "Mean")
CONFIG_ID = "generalization-v6-fold-mi1000-encv5-svrcv3"
RESULT_COLUMNS = (
    "model",
    "training_size",
    "size_index",
    "repeat_id",
    "mae",
    "prediction_sd",
    "selected_alpha",
    "best_cv_mse",
    "selected_C",
    "selected_epsilon",
    "best_cv_mae",
    "c_search_best_cv_mae",
    "n_iter",
    "nonzero_coefficients",
    "support_vectors",
    "convergence_warnings",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Paths and ordered feature contract for one prepared GEO dataset."""

    name: str
    title: str
    domain: str
    direct_path: Path
    meta_path: Path
    expected_meta_columns: tuple[str, ...]


def project_path(path: Path) -> Path:
    """Resolve a path while enforcing repository containment."""
    return ensure_repo_path(path)


def config_record(config: GeneralizationConfig) -> dict[str, Any]:
    """Convert a generalization configuration to a JSON-compatible record."""
    return json.loads(json.dumps(asdict(config)))


def log_spaced_training_sizes(n_train: int, config: GeneralizationConfig) -> np.ndarray:
    """Build the configured unique log-spaced sizes from 10 to the usable maximum."""
    maximum = n_train - config.test_reserve
    if maximum < 10:
        raise ValueError(f"Training pool has only {n_train} samples")
    sizes = np.unique(
        np.rint(np.logspace(np.log10(10), np.log10(maximum), config.n_training_sizes)).astype(int)
    )
    if len(sizes) != config.n_training_sizes or sizes[0] != 10 or sizes[-1] != maximum:
        raise ValueError(
            f"Cannot form {config.n_training_sizes} unique sizes from 10 through {maximum}"
        )
    return sizes


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label}: duplicate feature names")
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError(f"{label}: infinite feature values")
    return result


def load_aligned_inputs(
    spec: DatasetSpec, *, matrix_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the separately prepared direct and component inputs."""
    direct_path, meta_path = project_path(spec.direct_path), project_path(spec.meta_path)
    parquet = pq.ParquetFile(direct_path, thrift_container_size_limit=THRIFT_CONTAINER_LIMIT,
                             thrift_string_size_limit=THRIFT_STRING_LIMIT)
    direct_raw = parquet.read(columns=[TARGET], use_pandas_metadata=True).to_pandas()
    meta_raw = pd.read_parquet(meta_path)
    direct_raw.index = direct_raw.index.astype(str)
    meta_raw.index = meta_raw.index.astype(str)
    if not direct_raw.index.is_unique or not meta_raw.index.is_unique:
        raise ValueError(f"{spec.name}: sample IDs are not unique")
    if not direct_raw.index.equals(meta_raw.index):
        raise ValueError(f"{spec.name}: direct and component sample IDs/order differ")
    direct_age = pd.to_numeric(direct_raw[TARGET], errors="raise").to_numpy(dtype=float)
    meta_age = pd.to_numeric(meta_raw[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(direct_age).all() or not np.allclose(direct_age, meta_age):
        raise ValueError(f"{spec.name}: direct/component ages are invalid or differ")
    cpgs = [f.name for f in parquet.schema_arrow if f.name.startswith("cg")
            and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))]
    if not cpgs or len(set(cpgs)) != len(cpgs):
        raise ValueError(f"{spec.name}: expected unique numeric CpG columns")
    observed = tuple(str(column) for column in meta_raw if column != TARGET)
    if observed != spec.expected_meta_columns:
        raise ValueError(
            f"{spec.name}: expected component columns {spec.expected_meta_columns}, "
            f"found {observed}"
        )
    shape = (len(direct_raw), len(cpgs))
    if matrix_path is None:
        matrix = np.empty(shape, dtype=np.float32)
    else:
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(cpgs), 4096):
        names = cpgs[start:start + 4096]
        block = parquet.read(columns=names)
        for offset, column in enumerate(block.columns):
            matrix[:, start + offset] = column.to_numpy(zero_copy_only=False)
        if np.isinf(matrix[:, start:start + len(names)]).any():
            raise ValueError(f"{spec.name}: infinite CpG values")
    if matrix_path is None:
        direct = pd.DataFrame(matrix, columns=cpgs, index=direct_raw.index)
    else:
        matrix.flush()
        direct = pd.DataFrame({"row": np.arange(len(matrix))}, index=direct_raw.index)
    direct.attrs["cpg_count"] = len(cpgs)
    meta = _numeric_frame(meta_raw, spec.expected_meta_columns, f"{spec.name} component")
    direct[TARGET] = direct_age
    meta[TARGET] = direct_age
    return direct, meta


def split_and_impute(
    direct: pd.DataFrame, meta: pd.DataFrame, config: GeneralizationConfig,
    *, impute: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the fixed split and optionally fill missing feature values with zero."""
    order = direct.index.to_series().sample(frac=1.0, random_state=config.split_seed).index
    boundary = int(len(order) * config.train_fraction)

    def prepare(frame: pd.DataFrame, *, allow_drop: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = frame.loc[order[:boundary]].copy(), frame.loc[order[boundary:]].copy()
        features = [column for column in frame if column != TARGET]
        empty = [column for column in features if train[column].isna().all()]
        if empty and not allow_drop:
            raise ValueError(f"Component columns entirely missing in training: {empty}")
        features = [column for column in features if column not in empty]
        if not features:
            raise ValueError("No usable features remain")
        return (
            pd.concat([(train[features].fillna(0.0) if impute else train[features]), train[[TARGET]]], axis=1),
            pd.concat([(test[features].fillna(0.0) if impute else test[features]), test[[TARGET]]], axis=1),
        )

    direct_train, direct_test = prepare(direct, allow_drop=True)
    meta_train, meta_test = prepare(meta, allow_drop=False)
    return direct_train, direct_test, meta_train, meta_test


def validate_result(
    frame: pd.DataFrame, size: int, size_index: int, config: GeneralizationConfig
) -> None:
    """Require a complete finite model-by-resample shard for one size point."""
    expected = {(model, repeat) for model in MODEL_NAMES for repeat in range(config.n_resamples)}
    observed = set(zip(frame["model"], frame["repeat_id"].astype(int), strict=True))
    invalid_grid = observed != expected or len(frame) != len(expected)
    if set(frame.columns) != set(RESULT_COLUMNS) or invalid_grid:
        raise ValueError("Shard does not contain the complete model x resample grid")
    if set(frame["training_size"].astype(int)) != {size} or set(
        frame["size_index"].astype(int)
    ) != {size_index}:
        raise ValueError("Shard has the wrong training-size identity")
    if not np.isfinite(frame[["mae", "prediction_sd"]].to_numpy(float)).all():
        raise ValueError("Shard contains non-finite metrics")
    en = frame[frame["model"].isin(["EN", "Meta EN"])]
    if (not np.isfinite(en["best_cv_mse"]).all() or (en["best_cv_mse"] < 0).any()
            or not en["selected_alpha"].isin(config.elastic_net.alphas).all()):
        raise ValueError("Shard contains invalid Elastic Net search results")
    svr = frame[frame["model"].isin(["SVM", "Meta SVM"])]
    scores = svr[["best_cv_mae", "c_search_best_cv_mae", "support_vectors"]].to_numpy(float)
    if (not np.isfinite(scores).all() or (scores < 0).any()
            or not svr["selected_C"].isin(config.svr_c_values).all()
            or not svr["selected_epsilon"].isin(config.svr_epsilon_values).all()):
        raise ValueError("Shard contains invalid SVR search results")


def atomic_write_parquet(frame: pd.DataFrame, output: Path, *, index: bool = False) -> None:
    """Write a Parquet table through a temporary file and atomic rename."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=index)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], output: Path) -> None:
    """Write deterministic JSON through a temporary file and atomic rename."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyarrow", "scipy", "joblib", "scikit-learn")
    }


def evaluate_training_size(
    spec: DatasetSpec,
    size_index: int,
    output_root: Path,
    *,
    config: GeneralizationConfig | None = None,
) -> Path:
    """Fit and test all five models for one configured training-size point."""
    config = config or GeneralizationConfig()
    with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR")) as directory:
        return _evaluate_size(spec, size_index, output_root, config, Path(directory) / "cpgs.npy")


def _evaluate_size(
    spec: DatasetSpec, size_index: int, output_root: Path, config: GeneralizationConfig,
    matrix_path: Path,
) -> Path:
    direct, meta = load_aligned_inputs(spec, matrix_path=matrix_path)
    direct_train, direct_test, meta_train, meta_test = split_and_impute(direct, meta, config, impute=False)
    sizes = log_spaced_training_sizes(len(direct_train), config)
    if not 0 <= size_index < config.n_training_sizes:
        raise IndexError(f"size_index must be 0..{config.n_training_sizes - 1}")
    size = int(sizes[size_index])
    output = project_path(output_root) / spec.name / "shards" / f"size_{size_index:02d}.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    direct_path, meta_path = project_path(spec.direct_path), project_path(spec.meta_path)
    if output.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("config_id") == CONFIG_ID:
            existing = pd.read_parquet(output)
            validate_result(existing, size, size_index, config)
            if (provenance.get("config") != config_record(config)
                    or provenance.get("direct_input") != relative_path(direct_path)
                    or provenance.get("meta_input") != relative_path(meta_path)):
                raise ValueError(f"Existing shard configuration or input paths differ: {output}")
            return output
        if (provenance.get("config_id") != "generalization-v5-split105-30x50-fold-local-median-encv5-linear-svr"
                or provenance.get("dataset") != spec.name
                or provenance.get("meta_input") != relative_path(meta_path)):
            raise ValueError(f"Existing shard belongs to another configuration: {output}")

    rows: list[dict[str, object]] = []
    for repeat in range(config.n_resamples):
        seed = config.repeat_seed + repeat + size * 1000
        ids = direct_train.index.to_series().sample(n=size, replace=False, random_state=seed).index
        for label, train, test in (
            ("EN", direct_train.loc[ids], direct_test),
            ("Meta EN", meta_train.loc[ids], meta_test),
            ("SVM", direct_train.loc[ids], direct_test),
            ("Meta SVM", meta_train.loc[ids], meta_test),
        ):
            record = fit_model(label, train, test, config, domain=spec.domain,
                               matrix_path=matrix_path if label in ("EN", "SVM") else None)
            record.update(training_size=size, size_index=size_index, repeat_id=repeat)
            rows.append(record)
        mean_prediction = np.repeat(direct_train.loc[ids, TARGET].mean(), len(direct_test))
        rows.append(
            {
                "model": "Mean",
                "training_size": size,
                "size_index": size_index,
                "repeat_id": repeat,
                "mae": float(mean_absolute_error(direct_test[TARGET], mean_prediction)),
                "prediction_sd": 0.0,
                "selected_alpha": np.nan,
                "n_iter": 0,
                "nonzero_coefficients": 0,
                "support_vectors": 0,
                "convergence_warnings": 0,
            }
        )
    result = pd.DataFrame(rows).loc[:, list(RESULT_COLUMNS)]
    validate_result(result, size, size_index, config)
    provenance = {
        "dataset": spec.name,
        "domain": spec.domain,
        "title": spec.title,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": relative_path(direct_path),
        "direct_input": relative_path(direct_path),
        "meta_input": relative_path(meta_path),
        "training_pool_samples": len(direct_train),
        "test_samples": len(direct_test),
        "direct_features": direct.attrs["cpg_count"],
        "meta_features": list(spec.expected_meta_columns),
        "training_sizes": sizes.astype(int).tolist(),
        "size_index": size_index,
        "training_size": size,
        "config_id": CONFIG_ID,
        "config": config_record(config),
        "runtime": {
            "python": platform.python_version(),
            "packages": _versions(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    atomic_write_parquet(result, output)
    atomic_write_json(provenance, provenance_path)
    return output
