"""Shared model fitting and validated shard I/O for Figure 3 generalization."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from repository_paths import ensure_repo_path, relative_path

TARGET = "age"
MODEL_NAMES = ("EN", "Meta EN", "SVM", "Meta SVM", "Mean")
CONFIG_ID = "generalization-v4-geo-reproduction-split105-30x50-scaled-encv5-linear-svr"
RESULT_COLUMNS = (
    "model",
    "training_size",
    "size_index",
    "repeat_id",
    "mae",
    "prediction_sd",
    "selected_alpha",
    "n_iter",
    "nonzero_coefficients",
    "support_vectors",
    "convergence_warnings",
)


@dataclass(frozen=True)
class ElasticNetConfig:
    cv_folds: int = 5
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(x) for x in np.logspace(0.0, -5.0, 30))
    )
    l1_ratio: float = 0.5
    max_iter: int = 20_000
    tolerance: float = 1e-4
    n_jobs: int = -1


@dataclass(frozen=True)
class GeneralizationConfig:
    split_seed: int = 105
    repeat_seed: int = 42
    train_fraction: float = 0.8
    test_reserve: int = 10
    n_training_sizes: int = 30
    n_resamples: int = 50
    elastic_net: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    svr_c: float = 1.0
    svr_epsilon: float = 0.1
    svr_tolerance: float = 1e-3
    svr_max_iter: int = -1


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    domain: str
    direct_path: Path
    meta_path: Path
    expected_meta_columns: tuple[str, ...]


def project_path(path: Path) -> Path:
    return ensure_repo_path(path)


def config_record(config: GeneralizationConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))


def log_spaced_training_sizes(n_train: int, config: GeneralizationConfig) -> np.ndarray:
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


def load_aligned_inputs(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the separately prepared direct and component inputs."""
    direct_path, meta_path = project_path(spec.direct_path), project_path(spec.meta_path)
    direct_raw = pd.read_parquet(direct_path)
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
    cpgs = [str(column) for column in direct_raw if str(column).startswith("cg")]
    if not cpgs:
        raise ValueError(f"{spec.name}: no CpG features")
    observed = tuple(str(column) for column in meta_raw if column != TARGET)
    if observed != spec.expected_meta_columns:
        raise ValueError(
            f"{spec.name}: expected component columns {spec.expected_meta_columns}, "
            f"found {observed}"
        )
    direct = _numeric_frame(direct_raw, cpgs, f"{spec.name} direct")
    meta = _numeric_frame(meta_raw, spec.expected_meta_columns, f"{spec.name} component")
    direct[TARGET] = direct_age
    meta[TARGET] = direct_age
    return direct, meta


def split_and_impute(
    direct: pd.DataFrame, meta: pd.DataFrame, config: GeneralizationConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the fixed split and the original zero-imputation policy."""
    order = direct.sample(frac=1.0, random_state=config.split_seed).index
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
            pd.concat([train[features].fillna(0.0), train[[TARGET]]], axis=1),
            pd.concat([test[features].fillna(0.0), test[[TARGET]]], axis=1),
        )

    direct_train, direct_test = prepare(direct, allow_drop=True)
    meta_train, meta_test = prepare(meta, allow_drop=False)
    return direct_train, direct_test, meta_train, meta_test


def _elastic_net(config: GeneralizationConfig) -> Pipeline:
    settings = config.elastic_net
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    alphas=np.asarray(settings.alphas),
                    cv=settings.cv_folds,
                    l1_ratio=settings.l1_ratio,
                    max_iter=settings.max_iter,
                    n_jobs=settings.n_jobs,
                    selection="cyclic",
                    tol=settings.tolerance,
                ),
            ),
        ]
    )


def _linear_svr(config: GeneralizationConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="linear",
                    C=config.svr_c,
                    epsilon=config.svr_epsilon,
                    tol=config.svr_tolerance,
                    shrinking=True,
                    max_iter=config.svr_max_iter,
                ),
            ),
        ]
    )


def _fit_model(
    label: str, train: pd.DataFrame, test: pd.DataFrame, config: GeneralizationConfig
) -> dict[str, object]:
    estimator = _elastic_net(config) if label in {"EN", "Meta EN"} else _linear_svr(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train.drop(columns=TARGET), train[TARGET])
    prediction = np.asarray(estimator.predict(test.drop(columns=TARGET)), dtype=float)
    record: dict[str, object] = {
        "model": label,
        "mae": float(mean_absolute_error(test[TARGET], prediction)),
        "prediction_sd": float(np.std(prediction, ddof=1)) if len(prediction) > 1 else 0.0,
        "selected_alpha": np.nan,
        "n_iter": 0,
        "nonzero_coefficients": 0,
        "support_vectors": 0,
        "convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }
    fitted = estimator.named_steps["model"]
    if label in {"EN", "Meta EN"}:
        record.update(
            selected_alpha=float(fitted.alpha_),
            n_iter=int(fitted.n_iter_),
            nonzero_coefficients=int(np.count_nonzero(fitted.coef_)),
        )
    else:
        record["support_vectors"] = int(len(fitted.support_))
    return record


def validate_result(
    frame: pd.DataFrame, size: int, size_index: int, config: GeneralizationConfig
) -> None:
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


def atomic_write_parquet(frame: pd.DataFrame, output: Path, *, index: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=index)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], output: Path) -> None:
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
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn")
    }


def evaluate_training_size(
    spec: DatasetSpec,
    size_index: int,
    output_root: Path,
    *,
    config: GeneralizationConfig | None = None,
) -> Path:
    """Fit/test all five models for one of the 30 parallel size points."""
    config = config or GeneralizationConfig()
    direct, meta = load_aligned_inputs(spec)
    direct_train, direct_test, meta_train, meta_test = split_and_impute(direct, meta, config)
    sizes = log_spaced_training_sizes(len(direct_train), config)
    if not 0 <= size_index < config.n_training_sizes:
        raise IndexError(f"size_index must be 0..{config.n_training_sizes - 1}")
    size = int(sizes[size_index])
    output = project_path(output_root) / spec.name / "shards" / f"size_{size_index:02d}.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    direct_path, meta_path = project_path(spec.direct_path), project_path(spec.meta_path)
    if output.is_file() and provenance_path.is_file():
        existing = pd.read_parquet(output)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_result(existing, size, size_index, config)
        cache_matches = (
            provenance.get("config") == config_record(config)
            and provenance.get("direct_input") == relative_path(direct_path)
            and provenance.get("meta_input") == relative_path(meta_path)
        )
        if not cache_matches:
            raise ValueError(
                f"Existing shard configuration or prepared-input paths differ: {output}"
            )
        return output

    rows: list[dict[str, object]] = []
    for repeat in range(config.n_resamples):
        seed = config.repeat_seed + repeat + size * 1000
        ids = direct_train.sample(n=size, replace=False, random_state=seed).index
        for label, train, test in (
            ("EN", direct_train.loc[ids], direct_test),
            ("Meta EN", meta_train.loc[ids], meta_test),
            ("SVM", direct_train.loc[ids], direct_test),
            ("Meta SVM", meta_train.loc[ids], meta_test),
        ):
            record = _fit_model(label, train, test, config)
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
        "direct_features": direct_train.shape[1] - 1,
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
