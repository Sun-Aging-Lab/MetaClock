#!/usr/bin/env python3
"""Prepare and evaluate one Figure 2 array/tissue generalization dataset.

Species uses the distinct legacy workflow in `run_species_generalization.py`.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "fig2"
TARGET = "age"
MI_FEATURES = 10_000
MRMR_FEATURES = 1_000
SPLIT_RANDOM_STATE = 105
N_REPEATS = 50
MODEL_ORDER = ("SVM", "Meta SVM", "EN", "Meta EN", "Mean")


def select_cpg_features(
    frame: pd.DataFrame,
    *,
    mi_features: int = MI_FEATURES,
    mrmr_features: int = MRMR_FEATURES,
) -> pd.DataFrame:
    """Select CpGs by mutual information and then mRMR.

    The retained manuscript analysis selects on the complete healthy cohort.
    `mutual_info_regression` uses three neighbors and a fixed random seed;
    `mrmr_regression` receives `K=1000` (or all available MI features).
    """
    cpgs = [column for column in frame if str(column).startswith("cg")]
    if TARGET not in frame or not cpgs:
        raise ValueError("Input must contain age and numeric cg* columns")
    x = frame[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(frame[TARGET], errors="raise")

    selector = SelectKBest(
        score_func=partial(
            mutual_info_regression,
            n_neighbors=3,
            random_state=0,
            n_jobs=-1,
        ),
        k=min(mi_features, x.shape[1]),
    )
    selector.fit(x, y)
    selected = x.loc[:, selector.get_support()]

    keep = min(mrmr_features, selected.shape[1])
    if keep < selected.shape[1]:
        from mrmr import mrmr_regression

        columns = list(
            mrmr_regression(
                X=selected,
                y=y,
                K=keep,
                relevance="f",
                redundancy="c",
                denominator="mean",
                only_same_domain=False,
                return_scores=False,
                n_jobs=-1,
            )
        )
        selected = selected[columns]
    selected[TARGET] = y
    return selected


def training_sizes(n_train: int, n_sizes: int = 30) -> np.ndarray:
    """Return 30 unique log-spaced sample sizes from 10 to n_train minus 10."""
    if n_train < 20:
        raise ValueError("At least 20 training-pool samples are required")
    upper = n_train - 10
    return np.unique(np.rint(np.logspace(1, np.log10(upper), n_sizes)).astype(int))


def make_model(model_name: str) -> Pipeline:
    """Construct one of the two fixed manuscript regression pipelines."""
    if model_name == "EN":
        regressor = ElasticNet(alpha=0.001, l1_ratio=0.5, selection="cyclic")
    elif model_name == "SVM":
        regressor = SVR(kernel="linear", C=1.0, epsilon=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scale", StandardScaler()), ("model", regressor)])


def _impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain shared nonempty features and apply constant-zero imputation."""
    features = [column for column in train if column != TARGET]
    features = [column for column in features if not train[column].isna().all()]
    features = [column for column in features if column in test and not test[column].isna().all()]
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    train_x = pd.DataFrame(imputer.fit_transform(train[features]), index=train.index)
    test_x = pd.DataFrame(imputer.transform(test[features]), index=test.index)
    train_x[TARGET] = train[TARGET].to_numpy()
    test_x[TARGET] = test[TARGET].to_numpy()
    return train_x, test_x


def split_inputs(
    methylation: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align inputs and apply the fixed random-state-105 80/20 split."""
    common = methylation.index.intersection(meta.index)
    methylation = methylation.loc[common].dropna(subset=[TARGET])
    meta = meta.loc[methylation.index].dropna(subset=[TARGET])
    methylation = methylation.loc[meta.index]
    if not np.allclose(methylation[TARGET], meta[TARGET]):
        raise ValueError("Methylation and component-clock ages do not match")
    order = methylation.sample(frac=1.0, random_state=SPLIT_RANDOM_STATE).index
    split = int(len(order) * 0.8)
    raw_train, raw_test = _impute(
        methylation.loc[order[:split]], methylation.loc[order[split:]]
    )
    meta_train, meta_test = _impute(meta.loc[order[:split]], meta.loc[order[split:]])
    return raw_train, raw_test, meta_train, meta_test


def _score(model_name: str, train: pd.DataFrame, test: pd.DataFrame) -> float:
    model = make_model(model_name)
    model.fit(train.drop(columns=TARGET), train[TARGET])
    prediction = model.predict(test.drop(columns=TARGET))
    return float(mean_absolute_error(test[TARGET], prediction))


def evaluate_repeated_few_shot(
    methylation: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Evaluate CpG, Meta, and mean baselines on repeated training subsets."""
    raw_train, raw_test, meta_train, meta_test = split_inputs(methylation, meta)
    rows: list[dict[str, int | float | str]] = []
    for size in training_sizes(len(raw_train)):
        for repeat in range(n_repeats):
            seed = 42 + repeat + int(size) * 1000
            ids = raw_train.sample(n=int(size), random_state=seed).index
            raw_subset, meta_subset = raw_train.loc[ids], meta_train.loc[ids]
            for label, model_name, train, test in (
                ("SVM", "SVM", raw_subset, raw_test),
                ("Meta SVM", "SVM", meta_subset, meta_test),
                ("EN", "EN", raw_subset, raw_test),
                ("Meta EN", "EN", meta_subset, meta_test),
            ):
                rows.append(
                    {
                        "model": label,
                        "training_samples": int(size),
                        "repeat_id": repeat,
                        "mae": _score(model_name, train, test),
                    }
                )
            rows.append(
                {
                    "model": "Mean",
                    "training_samples": int(size),
                    "repeat_id": repeat,
                    "mae": float(
                        mean_absolute_error(
                            raw_test[TARGET],
                            np.repeat(raw_subset[TARGET].mean(), len(raw_test)),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_repeats(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeat-level MAEs for plotting and manuscript reporting."""
    return (
        frame.groupby(["model", "training_samples"], sort=False)["mae"]
        .agg(
            n_repeats="count",
            mean_mae="mean",
            sd_mae="std",
            q025_mae=lambda values: values.quantile(0.025),
            q975_mae=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )


def load_dataset_metadata(manifest: Path, dataset: str) -> pd.Series:
    """Read the single manifest row that controls panel labels and display size."""
    rows = pd.read_csv(manifest, sep="\t")
    rows = rows.loc[rows["dataset"].eq(dataset) & ~rows["group"].eq("species")]
    if len(rows) != 1:
        raise ValueError(f"Expected one array/tissue manifest row for {dataset}")
    return rows.iloc[0]


def build_compact_result(repeats: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:
    """Summarize repeats, apply the declared display floor, and add panel metadata."""
    result = summarize_repeats(repeats)
    minimum = int(metadata["minimum_training_samples"])
    result = result.loc[result["training_samples"].ge(minimum)].copy()
    result.insert(0, "label", str(metadata["label"]))
    result.insert(0, "dataset", str(metadata["dataset"]))
    result.insert(0, "group", str(metadata["group"]))
    result.insert(0, "panel_order", int(metadata["panel_order"]))
    return result


def write_compact_result(result: pd.DataFrame, output: Path) -> Path:
    """Write one panel, retaining other datasets already present in the table."""
    if output.is_file():
        existing = pd.read_csv(output, sep="\t")
        existing = existing.loc[~existing["dataset"].eq(result["dataset"].iloc[0])]
        result = pd.concat([existing, result], ignore_index=True)
    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}
    result["_model_order"] = result["model"].map(model_order)
    result = result.sort_values(
        ["panel_order", "_model_order", "training_samples"], kind="stable"
    ).drop(columns="_model_order")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    return output


def prepare_dataset(dataset_dir: Path) -> Path:
    """Create the selected 1,000-CpG input beside one beta-age table."""
    input_path = dataset_dir / "beta_age.parquet"
    output_path = dataset_dir / "selected_methylation.parquet"
    selected = select_cpg_features(pd.read_parquet(input_path))
    selected.to_parquet(output_path)
    return output_path


def evaluate_dataset(
    dataset_dir: Path,
    manifest: Path,
    output: Path,
    n_repeats: int,
) -> Path:
    """Run one prepared dataset and update its compact plotting table."""
    methylation = pd.read_parquet(dataset_dir / "selected_methylation.parquet")
    meta = pd.read_parquet(dataset_dir / "component_clock_predictions.parquet")
    metadata = load_dataset_metadata(manifest, dataset_dir.name)
    repeats = evaluate_repeated_few_shot(
        methylation,
        meta,
        n_repeats=n_repeats,
    )
    return write_compact_result(build_compact_result(repeats, metadata), output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument(
        "--manifest",
        type=Path,
        default=FIGURE_DIR / "datasets" / "manifest.tsv",
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--n-repeats", type=int, default=N_REPEATS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        print(prepare_dataset(args.dataset_dir))
    else:
        print(
            evaluate_dataset(
                args.dataset_dir,
                args.manifest,
                args.output,
                args.n_repeats,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
