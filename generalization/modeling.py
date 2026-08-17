"""Model fitting and validated point-shard I/O for Figure 3 generalization."""

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = "age"
MODEL_NAMES = ("EN", "Meta EN", "SVM", "Meta SVM", "Mean")
POINT_MODEL_SOURCE = "fresh_fit_per_resample"
POINT_COLUMNS = (
    "model",
    "mae",
    "prediction_sd",
    "convergence_warnings",
    "selected_alpha",
    "training_size",
    "size_index",
    "repeat_id",
)


@dataclass(frozen=True)
class ElasticNetConfig:
    """ElasticNetCV settings shared by direct and meta generalization models."""

    cv_folds: int = 5
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(x) for x in np.logspace(0, -5, 30))
    )
    l1_ratio: float = 0.5
    max_iter: int = 20_000
    tolerance: float = 1e-4
    n_jobs: int = -1


@dataclass(frozen=True)
class GeneralizationConfig:
    """Split, resampling, and estimator settings for Figure 3."""

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
    """Inputs, manuscript title, and exact component columns for one dataset."""

    name: str
    title: str
    direct_path: Path
    meta_path: Path
    expected_meta_columns: tuple[str, ...]


def project_path(path: Path) -> Path:
    """Resolve repository-relative defaults while preserving absolute overrides."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_record(config: GeneralizationConfig) -> dict[str, Any]:
    """Return a deterministic JSON-compatible configuration record."""
    return json.loads(json.dumps(asdict(config)))


def log_spaced_training_sizes(n_train: int, config: GeneralizationConfig) -> np.ndarray:
    """Return the exact unique integer log grid requested for Figure 3."""
    maximum = n_train - config.test_reserve
    if maximum < 10:
        raise ValueError(f"Training pool has only {n_train} samples")
    sizes = np.unique(
        np.rint(np.logspace(np.log10(10), np.log10(maximum), config.n_training_sizes)).astype(int)
    )
    if len(sizes) != config.n_training_sizes:
        raise ValueError(f"Cannot form {config.n_training_sizes} unique sizes from 10 to {maximum}")
    return sizes


def elastic_net_cv(config: GeneralizationConfig) -> Pipeline:
    """Build StandardScaler followed by the final five-fold ElasticNetCV."""
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


def linear_svr(config: GeneralizationConfig) -> Pipeline:
    """Build StandardScaler followed by the final unlimited-iteration linear SVR."""
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


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label}: duplicate feature names")
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError(f"{label}: infinite feature values")
    return result


def load_aligned_inputs(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load direct CpGs and the exact declared component-prediction columns."""
    direct_path, meta_path = project_path(spec.direct_path), project_path(spec.meta_path)
    direct_raw = pd.read_parquet(direct_path).dropna(subset=[TARGET])
    meta_raw = pd.read_parquet(meta_path).dropna(subset=[TARGET])
    direct_raw.index = direct_raw.index.astype(str)
    meta_raw.index = meta_raw.index.astype(str)
    if not direct_raw.index.is_unique or not meta_raw.index.is_unique:
        raise ValueError(f"{spec.name}: sample IDs are not unique")
    if not direct_raw.index.equals(meta_raw.index):
        raise ValueError(f"{spec.name}: direct and meta sample IDs/order differ")
    direct_age = pd.to_numeric(direct_raw[TARGET], errors="raise").to_numpy(dtype=float)
    meta_age = pd.to_numeric(meta_raw[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(direct_age).all() or not np.allclose(direct_age, meta_age):
        raise ValueError(f"{spec.name}: direct/meta ages are invalid or differ")

    direct_columns = [str(column) for column in direct_raw if str(column).startswith("cg")]
    if not direct_columns:
        raise ValueError(f"{spec.name}: no CpG features")
    observed_meta = tuple(str(column) for column in meta_raw if column != TARGET)
    if set(observed_meta) != set(spec.expected_meta_columns) or len(observed_meta) != len(
        spec.expected_meta_columns
    ):
        raise ValueError(
            f"{spec.name}: meta columns differ; expected {spec.expected_meta_columns}, "
            f"found {observed_meta}"
        )
    direct = _numeric_frame(direct_raw, direct_columns, f"{spec.name} direct")
    meta = _numeric_frame(meta_raw, spec.expected_meta_columns, f"{spec.name} meta")
    direct[TARGET], meta[TARGET] = direct_age, direct_age
    return direct, meta


def split_and_impute(
    direct: pd.DataFrame,
    meta: pd.DataFrame,
    config: GeneralizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the fixed 80/20 split and zero-impute without dropping meta models."""
    order = direct.sample(frac=1.0, random_state=config.split_seed).index
    boundary = int(len(order) * config.train_fraction)

    def prepare(
        frame: pd.DataFrame, *, allow_empty_feature_drop: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = frame.loc[order[:boundary]].copy()
        test = frame.loc[order[boundary:]].copy()
        features = [column for column in train if column != TARGET]
        empty = [column for column in features if train[column].isna().all()]
        if empty and not allow_empty_feature_drop:
            raise ValueError(f"Meta components are entirely missing after split: {empty}")
        features = [column for column in features if column not in empty]
        if not features:
            raise ValueError("No usable training features remain")
        train = pd.concat([train[features].fillna(0.0), train[[TARGET]]], axis=1)
        test = pd.concat([test.reindex(columns=features).fillna(0.0), test[[TARGET]]], axis=1)
        return train, test

    direct_train, direct_test = prepare(direct, allow_empty_feature_drop=True)
    meta_train, meta_test = prepare(meta, allow_empty_feature_drop=False)
    expected_meta = set(meta.columns) - {TARGET}
    if set(meta_train.columns) - {TARGET} != expected_meta:
        raise RuntimeError("Meta component columns changed during preprocessing")
    return direct_train, direct_test, meta_train, meta_test


def _fit_model(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: GeneralizationConfig,
) -> dict[str, object]:
    model = elastic_net_cv(config) if name in {"EN", "Meta EN"} else linear_svr(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train.drop(columns=TARGET), train[TARGET])
    prediction = np.asarray(model.predict(test.drop(columns=TARGET)), dtype=float)
    record: dict[str, object] = {
        "model": name,
        "mae": float(mean_absolute_error(test[TARGET], prediction)),
        "prediction_sd": float(np.std(prediction, ddof=1)),
        "convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        "selected_alpha": np.nan,
    }
    if name in {"EN", "Meta EN"}:
        record["selected_alpha"] = float(model.named_steps["model"].alpha_)
    return record


def validate_point_shard(
    frame: pd.DataFrame,
    *,
    size_index: int,
    training_size: int,
    config: GeneralizationConfig,
) -> None:
    """Validate the complete model × repeat Cartesian product for one point."""
    if set(frame.columns) != set(POINT_COLUMNS):
        raise ValueError(f"Unexpected point columns: {sorted(set(frame) ^ set(POINT_COLUMNS))}")
    expected_keys = {
        (model, repeat) for model in MODEL_NAMES for repeat in range(config.n_resamples)
    }
    observed_keys = set(zip(frame["model"], frame["repeat_id"].astype(int), strict=True))
    if observed_keys != expected_keys or len(frame) != len(expected_keys):
        raise ValueError("Point shard does not contain the complete model × repeat grid")
    if set(frame["size_index"].astype(int)) != {size_index}:
        raise ValueError("Point shard size_index differs")
    if set(frame["training_size"].astype(int)) != {training_size}:
        raise ValueError("Point shard training_size differs")
    numeric = frame[["mae", "prediction_sd", "convergence_warnings"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame["mae"] < 0).any():
        raise ValueError("Point shard contains invalid metrics")
    en = frame["model"].isin({"EN", "Meta EN"})
    if not np.isfinite(frame.loc[en, "selected_alpha"].to_numpy(float)).all():
        raise ValueError("ENCV rows contain invalid selected alpha")
    if frame.loc[~en, "selected_alpha"].notna().any():
        raise ValueError("Non-ENCV rows unexpectedly contain selected alpha")


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib")
    return {name: importlib.metadata.version(name) for name in packages}


def atomic_write_parquet(frame: pd.DataFrame, output: Path, *, index: bool = False) -> None:
    """Publish parquet atomically so readers never observe a partial shard."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=index)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], output: Path) -> None:
    """Publish a JSON provenance record atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_training_size(
    spec: DatasetSpec,
    size_index: int,
    output_root: Path,
    *,
    config: GeneralizationConfig | None = None,
) -> Path:
    """Run or safely reuse one dataset/training-size point with 50 resamples."""
    config = config or GeneralizationConfig()
    output_root = project_path(output_root)
    direct, meta = load_aligned_inputs(spec)
    direct_train, direct_test, meta_train, meta_test = split_and_impute(direct, meta, config)
    sizes = log_spaced_training_sizes(len(direct_train), config)
    if not 0 <= size_index < config.n_training_sizes:
        raise IndexError(f"size_index must be in [0, {config.n_training_sizes - 1}]")
    size = int(sizes[size_index])
    output = output_root / spec.name / f"size_{size_index:02d}.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    expected_config = config_record(config)
    expected_inputs = {
        "direct_input": str(project_path(spec.direct_path)),
        "meta_input": str(project_path(spec.meta_path)),
        "meta_features": list(spec.expected_meta_columns),
        "point_models": list(MODEL_NAMES),
        "point_model_source": POINT_MODEL_SOURCE,
    }
    if output.is_file() and provenance_path.is_file():
        existing = pd.read_parquet(output)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_point_shard(existing, size_index=size_index, training_size=size, config=config)
        if (
            provenance.get("dataset") != spec.name
            or provenance.get("config") != expected_config
            or any(provenance.get(key) != value for key, value in expected_inputs.items())
        ):
            raise ValueError(f"Existing shard provenance does not match: {output}")
        return output

    rows: list[dict[str, object]] = []
    for repeat in range(config.n_resamples):
        seed = config.repeat_seed + repeat + size * 1000
        ids = direct_train.sample(n=size, replace=False, random_state=seed).index
        for name, train, test in (
            ("EN", direct_train.loc[ids], direct_test),
            ("Meta EN", meta_train.loc[ids], meta_test),
            ("SVM", direct_train.loc[ids], direct_test),
            ("Meta SVM", meta_train.loc[ids], meta_test),
        ):
            record = _fit_model(name, train, test, config)
            record.update(training_size=size, size_index=size_index, repeat_id=repeat)
            rows.append(record)
        mean_prediction = np.repeat(direct_train.loc[ids, TARGET].mean(), len(direct_test))
        rows.append(
            {
                "model": "Mean",
                "mae": float(mean_absolute_error(direct_test[TARGET], mean_prediction)),
                "prediction_sd": 0.0,
                "convergence_warnings": 0,
                "selected_alpha": np.nan,
                "training_size": size,
                "size_index": size_index,
                "repeat_id": repeat,
            }
        )
    result = pd.DataFrame(rows, columns=POINT_COLUMNS)
    validate_point_shard(result, size_index=size_index, training_size=size, config=config)
    provenance = {
        "dataset": spec.name,
        "title": spec.title,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direct_input": expected_inputs["direct_input"],
        "meta_input": expected_inputs["meta_input"],
        "training_pool_samples": len(direct_train),
        "test_samples": len(direct_test),
        "direct_features": direct_train.shape[1] - 1,
        "meta_features": expected_inputs["meta_features"],
        "point_models": expected_inputs["point_models"],
        "point_model_source": expected_inputs["point_model_source"],
        "size_index": size_index,
        "training_size": size,
        "config": expected_config,
        "runtime": {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    atomic_write_parquet(result, output)
    atomic_write_json(provenance, provenance_path)
    return output
