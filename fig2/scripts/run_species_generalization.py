#!/usr/bin/env python3
"""Evaluate the Figure 2 species panel using its legacy feature pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "fig2"
TARGET = "age"
SPLIT_RANDOM_STATE = 105
N_REPEATS = 50
MODEL_ORDER = ("SVM", "EN", "Meta SVM", "Meta EN", "Mean")


def training_sizes(n_train: int, n_sizes: int = 30) -> np.ndarray:
    """Return the legacy 30 log-spaced sizes from 10 to n_train minus 10."""
    if n_train <= 10:
        raise ValueError("Species training pool must contain more than ten rows")
    return np.unique(
        np.rint(np.logspace(1, np.log10(n_train - 10), n_sizes)).astype(int)
    )


def select_meta_features(frame: pd.DataFrame, suffix: str = "elu") -> pd.DataFrame:
    """Keep age and the per-species MLP features for one activation suffix."""
    columns = [
        column
        for column in frame
        if column == TARGET or str(column).endswith(f"_mlp_{suffix}")
    ]
    if TARGET not in columns or len(columns) == 1:
        raise ValueError(f"No _mlp_{suffix} species features were found")
    return frame[columns].copy()


def split_and_impute(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fixed shuffle/split and separate zero imputation per split."""
    frame = frame.dropna(subset=[TARGET]).sample(frac=1.0, random_state=SPLIT_RANDOM_STATE)
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split].copy(), frame.iloc[split:].copy()
    for part in (train, test):
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        part.loc[:, :] = imputer.fit_transform(part)
    return train, test


def fit_mae(model_name: str, train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Fit an unscaled legacy EN or linear SVR and return test MAE."""
    if model_name == "EN":
        model = ElasticNet(alpha=0.001, l1_ratio=0.5)
    elif model_name == "SVM":
        model = SVR(kernel="linear")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(train.drop(columns=TARGET), train[TARGET])
    return float(
        mean_absolute_error(test[TARGET], model.predict(test.drop(columns=TARGET)))
    )


def evaluate_species(
    direct: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Evaluate conserved-CpG and MLP-ELU meta features on repeated subsets."""
    if not direct.index.equals(meta.index):
        raise ValueError("Direct and meta species inputs must have the same row index")
    if not np.allclose(direct[TARGET], meta[TARGET], equal_nan=True):
        raise ValueError("Direct and meta species ages must match row by row")
    direct_train, direct_test = split_and_impute(direct)
    meta_train, meta_test = split_and_impute(select_meta_features(meta))
    direct_sizes = training_sizes(len(direct_train))
    meta_sizes = training_sizes(len(meta_train))
    if not np.array_equal(direct_sizes, meta_sizes):
        raise ValueError("Direct and meta species inputs produce different training sizes")

    rows: list[dict[str, int | float | str]] = []
    for size in direct_sizes:
        for repeat in range(n_repeats):
            seed = 42 + repeat + int(size) * 1000
            direct_subset = direct_train.sample(n=int(size), random_state=seed)
            meta_subset = meta_train.sample(n=int(size), random_state=seed)
            for label, model_name, train, test in (
                ("SVM", "SVM", direct_subset, direct_test),
                ("EN", "EN", direct_subset, direct_test),
                ("Meta SVM", "SVM", meta_subset, meta_test),
                ("Meta EN", "EN", meta_subset, meta_test),
            ):
                rows.append(
                    {
                        "model": label,
                        "training_samples": int(size),
                        "repeat_id": repeat,
                        "mae": fit_mae(model_name, train, test),
                    }
                )
            rows.append(
                {
                    "model": "Mean",
                    "training_samples": int(size),
                    "repeat_id": repeat,
                    "mae": float(
                        mean_absolute_error(
                            meta_test[TARGET],
                            np.repeat(meta_subset[TARGET].mean(), len(meta_test)),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_compact_result(repeats: pd.DataFrame, manifest: Path, species: str) -> pd.DataFrame:
    """Summarize repeat MAEs and add the species panel metadata."""
    metadata = pd.read_csv(manifest, sep="\t")
    metadata = metadata.loc[
        metadata["dataset"].eq(species) & metadata["group"].eq("species")
    ]
    if len(metadata) != 1:
        raise ValueError(f"Expected one species manifest row for {species}")
    row = metadata.iloc[0]
    result = (
        repeats.groupby(["model", "training_samples"], sort=False)["mae"]
        .agg(
            n_repeats="count",
            mean_mae="mean",
            sd_mae="std",
            q025_mae=lambda values: values.quantile(0.025),
            q975_mae=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    result = result.loc[
        result["training_samples"].ge(int(row["minimum_training_samples"]))
    ].copy()
    result.insert(0, "label", str(row["label"]))
    result.insert(0, "dataset", str(row["dataset"]))
    result.insert(0, "group", str(row["group"]))
    result.insert(0, "panel_order", int(row["panel_order"]))
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    result["_model_order"] = result["model"].map(order)
    return result.sort_values(
        ["_model_order", "training_samples"], kind="stable"
    ).drop(columns="_model_order")


def run(
    dataset_dir: Path,
    manifest: Path,
    species: str,
    output: Path,
    n_repeats: int,
) -> Path:
    """Load one species dataset and write its compact plotting table."""
    direct = pd.read_parquet(dataset_dir / f"{species}_fewshot_40k.parquet")
    meta = pd.read_parquet(
        dataset_dir / f"loo_per_species_predictions_{species}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    repeats = evaluate_species(direct, meta, n_repeats=n_repeats)
    build_compact_result(repeats, manifest, species).to_csv(output, sep="\t", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=FIGURE_DIR / "datasets" / "species" / "BosTaurus",
    )
    parser.add_argument("--species", default="BosTaurus")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FIGURE_DIR / "datasets" / "manifest.tsv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        run(
            args.dataset_dir,
            args.manifest,
            args.species,
            args.output,
            args.n_repeats,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
