#!/usr/bin/env python3
"""Fit and test one GSE223748 species sample-size point (50 resample replicates).

Direct CpGs and target samples come from the prepared GSE223748 species input.
Meta features come from nine non-self saved species ENCV clocks plus the pooled
ENCV clock; the nine species clocks also support the evolution coefficients.

The Slurm array task ID selects one of 30 log-spaced training sizes.  A local
run without Slurm evaluates size index 0.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from repository_paths import ensure_repo_path, relative_path

from .species_config import (
    DISPLAY_NAMES,
    GENERALIZATION_CONFIG_ID as CONFIG_ID,
    POOLED_COMPONENT,
    SPECIES,
    direct_input_path,
    loo_feature_path,
)
from .species_submodels import COMPONENT_CONFIG_ID, ENCV_ARCHITECTURE

############################## USER INPUT ####################################

######################## FIXED SCIENTIFIC CONFIG #############################

TARGET = "age"
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "generalization" / "species"

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
    """Read a prepared Parquet input and validate its sample identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"Required prepared species input does not exist: {path}")
    frame = pd.read_parquet(path)
    if "sample_id" in frame:
        frame = frame.set_index(frame["sample_id"].astype(str)).drop(columns="sample_id")
    else:
        frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError("Sample identifiers must be unique")
    return frame


def numeric_features(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    """Convert declared features to finite-or-missing numeric values."""
    converted = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(converted.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains infinite values")
    return converted


def load_model_inputs(
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Load one GSE-derived target and its saved-submodel LOO features."""
    if name not in SPECIES:
        raise ValueError(f"Species must be one of {SPECIES}; found {name!r}")
    raw = read_table(direct_input_path(name))
    loo_path = loo_feature_path(name)
    if not loo_path.is_file():
        raise FileNotFoundError(
            f"Missing saved-submodel LOO features: {loo_path}. Run "
            f"python -m generalization.species.build_species_loo_features --species {name}"
        )
    loo = read_table(loo_path)
    if loo.attrs.get("component_config_id") != COMPONENT_CONFIG_ID:
        raise ValueError("LOO features use an outdated component configuration; rebuild them")
    if not raw.index.equals(loo.index):
        raise ValueError(f"Direct and LOO sample IDs differ for {name}")
    ages = pd.to_numeric(raw[TARGET], errors="coerce")
    if not np.allclose(ages.to_numpy(float), pd.to_numeric(loo[TARGET]).to_numpy(float)):
        raise ValueError(f"Direct and LOO ages differ for {name}")
    keep = ages.notna() & np.isfinite(ages)
    raw, loo, ages = raw.loc[keep].copy(), loo.loc[keep].copy(), ages.loc[keep].astype(float)
    if len(raw) < 25:
        raise ValueError("At least 25 valid age-labelled samples are required")
    cpg_columns = [str(column) for column in raw if str(column).startswith("cg")]
    component_columns = [
        *(f"{species}_elasticnetcv" for species in SPECIES if species != name),
        f"{POOLED_COMPONENT}_elasticnetcv",
    ]
    evolution_columns = [
        f"{species}_elasticnetcv" for species in SPECIES if species != name
    ]
    missing = sorted(set([*component_columns, *evolution_columns]) - set(loo.columns))
    if missing:
        raise ValueError(f"LOO feature table is missing columns: {missing}")
    direct = numeric_features(raw, cpg_columns, "Direct CpG matrix")
    meta = numeric_features(loo, component_columns, "ENCV component matrix")
    evolution = numeric_features(loo, evolution_columns, "ENCV evolution component matrix")
    direct[TARGET] = ages
    meta[TARGET] = ages
    evolution[TARGET] = ages
    return direct, meta, evolution, name, DISPLAY_NAMES[name]


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
        seed = RESAMPLE_SEED + repeat_id + size * 1000
        sample_ids = direct_train.sample(n=size, replace=False, random_state=seed).index
        direct_subset, meta_subset = direct_train.loc[sample_ids], meta_train.loc[sample_ids]
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


######################## EVOLUTION COEFFICIENTS ##############################


def write_evolution_coefficients(
    evolution: pd.DataFrame, species: str, output: Path
) -> Path:
    """Fit the formal directed non-negative component weights once per species.

    Evolution uses the nine non-self species ENCV inputs shared with the few-shot
    Meta models, excluding the pooled clock from directed species coefficients.
    """
    features = [column for column in evolution if column != TARGET]
    complete = evolution[[*features, TARGET]].dropna()
    if len(complete) < CV_FOLDS:
        raise ValueError("Evolution coefficient fitting requires at least five complete samples")
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=1_000,
                    positive=True,
                    selection="cyclic",
                    tol=1e-4,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(complete[features], complete[TARGET])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Evolution component Elastic Net did not converge")
    coefficients = pd.DataFrame(
        {
            "test_species": species,
            "component_species": [column.removesuffix("_elasticnetcv") for column in features],
            "coefficient": estimator.named_steps["model"].coef_.astype(float),
        }
    )
    if len(coefficients) != 9 or coefficients["component_species"].duplicated().any():
        raise RuntimeError("Expected nine unique directed evolution coefficients")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    coefficients.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(temporary, output)
    return output


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
    value = int(os.environ.get("METACLOCK_SIZE_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if not 0 <= value < N_TRAINING_SIZES:
        raise ValueError(f"Size index must be 0..{N_TRAINING_SIZES - 1}")
    return value


def main() -> int:
    """Fit or validate the selected species/size shard and its provenance."""
    name = os.environ.get("TARGET_SPECIES", SPECIES[0])
    path = ensure_repo_path(direct_input_path(name))
    direct, meta, evolution, name, title = load_model_inputs(name)
    direct_train, direct_test, meta_train, meta_test = split_and_impute(direct, meta)
    sizes = training_sizes(len(direct_train))
    size_index = selected_size_index()
    size = int(sizes[size_index])
    output = RESULT_ROOT / name / "shards" / f"size_{size_index:02d}.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    if output.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_result(pd.read_parquet(output), size, size_index)
        if (
            provenance.get("input_path") != relative_path(path)
            or provenance.get("config_id") != CONFIG_ID
        ):
            raise ValueError(f"Existing shard has a different input or configuration: {output}")
        if size_index == 0:
            write_evolution_coefficients(
                evolution, name, RESULT_ROOT / name / "evolution_coefficients.tsv"
            )
        print(f"[skip] {name}: size_index={size_index}, training_size={size}")
        return 0
    result = evaluate_point(direct_train, direct_test, meta_train, meta_test, size, size_index)
    validate_result(result, size, size_index)
    atomic_parquet(result, output)
    atomic_json(
        {
            "domain": "species",
            "config_id": CONFIG_ID,
            "dataset": name,
            "title": title,
            "input_path": relative_path(path),
            "size_index": size_index,
            "training_size": size,
            "training_sizes": sizes.tolist(),
            "training_pool_samples": len(direct_train),
            "test_samples": len(direct_test),
            "direct_features": direct_train.shape[1] - 1,
            "component_features": [column for column in meta if column != TARGET],
            "component_prediction_architecture": ENCV_ARCHITECTURE,
            "evolution_features": [column for column in evolution if column != TARGET],
            "split": {"train_fraction": TRAIN_FRACTION, "seed": SPLIT_SEED},
            "resampling": {
                "repeats": N_RESAMPLES,
                "with_replacement": False,
                "sampler": "pandas.DataFrame.sample",
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
    if size_index == 0:
        write_evolution_coefficients(
            evolution, name, RESULT_ROOT / name / "evolution_coefficients.tsv"
        )
    print(f"[done] {name}: size_index={size_index}, training_size={size}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
