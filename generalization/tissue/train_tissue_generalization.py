#!/usr/bin/env python3
"""Fit and test one healthy-tissue sample-size point (50 resample replicates).

Edit only ``DATASET_PATH``.  The input is one CSV/TSV/parquet table containing
``age``, direct ``cg...`` beta-value columns, and the 16 chronological-clock
component predictions listed in ``COMPONENT_COLUMNS``.  ``sample_id``,
``dataset``, and ``title`` are optional metadata columns.

The Slurm array task ID selects one of 30 log-spaced training sizes.  A local
run without Slurm evaluates size index 0.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

############################## USER INPUT ####################################

DATASET_PATH = Path("/path/to/your/file")  # fill in your path

######################## FIXED SCIENTIFIC CONFIG #############################

TARGET = "age"
COMPONENT_COLUMNS = (
    "horvath2013",
    "hannum",
    "dnamphenoage",
    "altumage",
    "han",
    "encen40",
    "hrsinchphenoage",
    "lin",
    "skinandblood",
    "yingcausage",
    "zhangblup",
    "zhangen",
    "pchorvath2013",
    "pchannum",
    "pcphenoage",
    "pcskinandblood",
)
KNOWN_TITLES = {
    "GSE175458": "Lung",
    "GSE232891": "Saliva",
    "GSE287331": "Breast",
}
MODEL_NAMES = ("EN", "Meta EN", "SVM", "Meta SVM", "Mean")
SPLIT_SEED = 105
RESAMPLE_SEED = 42
TRAIN_FRACTION = 0.8
TEST_RESERVE = 10
N_TRAINING_SIZES = 30
N_RESAMPLES = 50
CV_FOLDS = 5
ALPHAS = np.logspace(0.0, -5.0, 30)
L1_RATIO = 0.5
MAX_ITER = 20_000
TOLERANCE = 1e-4
CONFIG_ID = "generalization-v2-split105-30x50-without-replacement-scaled-encv5-scaled-svr"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "generalization" / "tissue"

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


############################ INPUT AND SPLIT #################################


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported table without silently guessing arbitrary formats."""
    if not path.is_file():
        raise FileNotFoundError(f"Set DATASET_PATH to an existing file; found {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(path)
    elif suffixes.endswith((".tsv", ".tsv.gz")):
        frame = pd.read_csv(path, sep="\t")
    elif suffixes.endswith((".csv", ".csv.gz")):
        frame = pd.read_csv(path)
    else:
        raise ValueError("DATASET_PATH must be CSV, TSV, or parquet")
    if "sample_id" in frame:
        frame = frame.set_index(frame["sample_id"].astype(str)).drop(columns="sample_id")
    else:
        frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError("Sample identifiers must be unique")
    return frame


def constant_text(frame: pd.DataFrame, column: str, fallback: str) -> str:
    """Read one optional constant metadata value."""
    if column not in frame:
        return fallback
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"Optional metadata column {column!r} must be constant")
    return str(values[0])


def dataset_name(frame: pd.DataFrame, path: Path) -> str:
    """Build a filesystem-safe dataset name from metadata or the filename."""
    name = constant_text(frame, "dataset", path.name.split(".")[0])
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("Could not derive a non-empty dataset name")
    return safe


def numeric_features(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    """Convert declared features to finite-or-missing numeric values."""
    converted = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(converted.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains infinite values")
    return converted


def load_model_inputs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Return aligned direct-CpG and component-prediction tables."""
    raw = read_table(path)
    if TARGET not in raw:
        raise ValueError(f"Input is missing required target column {TARGET!r}")
    ages = pd.to_numeric(raw[TARGET], errors="coerce")
    keep = ages.notna() & np.isfinite(ages)
    raw, ages = raw.loc[keep].copy(), ages.loc[keep].astype(float)
    if len(raw) < 25:
        raise ValueError("At least 25 valid age-labelled samples are required")
    missing = sorted(set(COMPONENT_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing chronological component predictions: {missing}")
    cpg_columns = [str(column) for column in raw if str(column).startswith("cg")]
    if not cpg_columns:
        raise ValueError("No direct CpG columns beginning with 'cg' were found")
    direct = numeric_features(raw, cpg_columns, "Direct CpG matrix")
    meta = numeric_features(raw, list(COMPONENT_COLUMNS), "Component matrix")
    direct[TARGET] = ages
    meta[TARGET] = ages
    name = dataset_name(raw, path)
    title = KNOWN_TITLES.get(name, constant_text(raw, "title", name))
    return direct, meta, name, title


def split_and_impute(
    direct: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the formal fixed 80/20 split and training-aware zero imputation."""
    order = direct.sample(frac=1.0, random_state=SPLIT_SEED).index
    boundary = int(len(order) * TRAIN_FRACTION)

    def prepare(
        frame: pd.DataFrame, *, allow_empty_drop: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = frame.loc[order[:boundary]].copy(), frame.loc[order[boundary:]].copy()
        features = [column for column in frame if column != TARGET]
        empty = [column for column in features if train[column].isna().all()]
        if empty and not allow_empty_drop:
            raise ValueError(f"Component columns are entirely missing in training: {empty}")
        features = [column for column in features if column not in empty]
        train.loc[:, features] = train[features].fillna(0.0)
        test.loc[:, features] = test[features].fillna(0.0)
        return train.loc[:, [*features, TARGET]], test.loc[:, [*features, TARGET]]

    direct_train, direct_test = prepare(direct, allow_empty_drop=True)
    meta_train, meta_test = prepare(meta, allow_empty_drop=False)
    return direct_train, direct_test, meta_train, meta_test


def training_sizes(n_train: int) -> np.ndarray:
    """Return 30 unique integer log-space sizes from 10 to n_train - 10."""
    maximum = n_train - TEST_RESERVE
    if maximum < 10:
        raise ValueError(f"The 80% training pool has only {n_train} samples")
    sizes = np.unique(
        np.rint(np.logspace(np.log10(10), np.log10(maximum), N_TRAINING_SIZES)).astype(int)
    )
    if len(sizes) != N_TRAINING_SIZES or sizes[0] != 10 or sizes[-1] != maximum:
        raise ValueError(
            f"Cannot form {N_TRAINING_SIZES} unique sizes from 10 through {maximum}; "
            f"increase the dataset size (observed grid: {sizes.tolist()})"
        )
    return sizes


############################ MODEL FITTING ###################################


def elastic_net() -> Pipeline:
    """Build the formal scaled five-fold ElasticNetCV estimator."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    alphas=ALPHAS,
                    cv=CV_FOLDS,
                    l1_ratio=L1_RATIO,
                    max_iter=MAX_ITER,
                    n_jobs=-1,
                    selection="cyclic",
                    tol=TOLERANCE,
                ),
            ),
        ]
    )


def linear_svr() -> Pipeline:
    """Build the formal scaled linear-SVR estimator."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="linear",
                    C=1.0,
                    epsilon=0.1,
                    tol=1e-3,
                    shrinking=True,
                    max_iter=-1,
                ),
            ),
        ]
    )


def fit_model(
    label: str, train: pd.DataFrame, test: pd.DataFrame, size: int, repeat_id: int
) -> dict[str, object]:
    """Fit one direct or Meta model and calculate held-out MAE."""
    estimator = elastic_net() if label in {"EN", "Meta EN"} else linear_svr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train.drop(columns=TARGET), train[TARGET])
    prediction = np.asarray(estimator.predict(test.drop(columns=TARGET)), dtype=float)
    record: dict[str, object] = {
        "model": label,
        "training_size": size,
        "repeat_id": repeat_id,
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
    model = estimator.named_steps["model"]
    if label in {"EN", "Meta EN"}:
        record.update(
            selected_alpha=float(model.alpha_),
            n_iter=int(model.n_iter_),
            nonzero_coefficients=int(np.count_nonzero(model.coef_)),
        )
    else:
        record["support_vectors"] = int(len(model.support_))
    return record


def evaluate_point(
    direct_train: pd.DataFrame,
    direct_test: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    size: int,
    size_index: int,
) -> pd.DataFrame:
    """Run all five model rows for each of 50 without-replacement resamples."""
    rows: list[dict[str, object]] = []
    for repeat_id in range(N_RESAMPLES):
        rng = np.random.default_rng(RESAMPLE_SEED + repeat_id + size * 1000)
        positions = rng.choice(len(direct_train), size=size, replace=False)
        direct_subset, meta_subset = direct_train.iloc[positions], meta_train.iloc[positions]
        rows.append(fit_model("EN", direct_subset, direct_test, size, repeat_id))
        rows.append(fit_model("Meta EN", meta_subset, meta_test, size, repeat_id))
        rows.append(fit_model("SVM", direct_subset, direct_test, size, repeat_id))
        rows.append(fit_model("Meta SVM", meta_subset, meta_test, size, repeat_id))
        mean_prediction = np.full(len(direct_test), direct_subset[TARGET].mean())
        rows.append(
            {
                "model": "Mean",
                "training_size": size,
                "repeat_id": repeat_id,
                "mae": float(mean_absolute_error(direct_test[TARGET], mean_prediction)),
                "prediction_sd": 0.0,
                "selected_alpha": np.nan,
                "n_iter": 0,
                "nonzero_coefficients": 0,
                "support_vectors": 0,
                "convergence_warnings": 0,
            }
        )
    result = pd.DataFrame(rows)
    result.insert(2, "size_index", size_index)
    return result.loc[:, list(RESULT_COLUMNS)]


########################### OUTPUT AND ENTRYPOINT ############################


def validate_result(frame: pd.DataFrame, size: int, size_index: int) -> None:
    """Reject incomplete shards before they can be reused or plotted."""
    expected = {(model, repeat) for model in MODEL_NAMES for repeat in range(N_RESAMPLES)}
    observed = set(zip(frame["model"], frame["repeat_id"].astype(int), strict=True))
    if (
        set(frame.columns) != set(RESULT_COLUMNS)
        or observed != expected
        or len(frame) != len(expected)
    ):
        raise ValueError("Point shard does not contain the complete model x resample grid")
    if set(frame["training_size"].astype(int)) != {size} or set(
        frame["size_index"].astype(int)
    ) != {size_index}:
        raise ValueError("Point shard has the wrong training-size identity")
    if not np.isfinite(frame[["mae", "prediction_sd"]].to_numpy(float)).all():
        raise ValueError("Point shard contains non-finite metrics")


def atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    """Publish a complete shard atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict[str, object], output: Path) -> None:
    """Publish reproducibility metadata atomically."""
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


def selected_size_index() -> int:
    """Use Slurm's point index, defaulting to the first point for a local run."""
    value = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= value < N_TRAINING_SIZES:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be 0..{N_TRAINING_SIZES - 1}")
    return value


def main() -> int:
    path = DATASET_PATH.expanduser().resolve()
    direct, meta, name, title = load_model_inputs(path)
    direct_train, direct_test, meta_train, meta_test = split_and_impute(direct, meta)
    sizes = training_sizes(len(direct_train))
    size_index = selected_size_index()
    size = int(sizes[size_index])
    output = RESULT_ROOT / name / "shards" / f"size_{size_index:02d}.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    if output.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_result(pd.read_parquet(output), size, size_index)
        if provenance.get("input_path") != str(path) or provenance.get("config_id") != CONFIG_ID:
            raise ValueError(f"Existing shard has a different input or configuration: {output}")
        print(f"[skip] {name}: size_index={size_index}, training_size={size}")
        return 0
    result = evaluate_point(direct_train, direct_test, meta_train, meta_test, size, size_index)
    validate_result(result, size, size_index)
    atomic_parquet(result, output)
    atomic_json(
        {
            "domain": "tissue",
            "config_id": CONFIG_ID,
            "dataset": name,
            "title": title,
            "input_path": str(path),
            "size_index": size_index,
            "training_size": size,
            "training_sizes": sizes.tolist(),
            "training_pool_samples": len(direct_train),
            "test_samples": len(direct_test),
            "direct_features": direct_train.shape[1] - 1,
            "component_features": list(COMPONENT_COLUMNS),
            "split": {"train_fraction": TRAIN_FRACTION, "seed": SPLIT_SEED},
            "resampling": {
                "repeats": N_RESAMPLES,
                "with_replacement": False,
                "seed_formula": "42 + repeat_id + training_size * 1000",
            },
            "elastic_net_cv": {
                "pipeline": ["StandardScaler", "ElasticNetCV"],
                "cv": CV_FOLDS,
                "alphas": ALPHAS.tolist(),
                "l1_ratio": L1_RATIO,
                "max_iter": MAX_ITER,
                "tol": TOLERANCE,
                "n_jobs": -1,
            },
            "svr": {
                "pipeline": ["StandardScaler", "SVR"],
                "kernel": "linear",
                "C": 1.0,
                "epsilon": 0.1,
                "tol": 1e-3,
                "max_iter": -1,
            },
            "runtime": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "pyarrow": importlib.metadata.version("pyarrow"),
            },
        },
        provenance_path,
    )
    print(f"[done] {name}: size_index={size_index}, training_size={size}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
