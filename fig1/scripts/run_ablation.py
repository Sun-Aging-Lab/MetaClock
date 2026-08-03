#!/usr/bin/env python3
"""Run the Fig. 1 cumulative component-clock ablation study.

Expected input is one row per sample with chronological age, dataset name, and
all component-clock predictions. The default path is relative to ``fig1``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


FIGURE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATASETS = (
    "GSE40279", "GSE51057", "GSE42861", "GSE41169", "GSE51032",
    "GSE73103", "GSE69270", "GSE64495", "GSE52588",
)
TEST_DATASETS = ("GSE147221", "GSE157131", "GSE234461", "GSE72680")
COMPONENT_CLOCKS = (
    "horvath2013", "hannum", "dnamphenoage", "altumage", "han", "encen40",
    "hrsinchphenoage", "lin", "skinandblood", "yingcausage", "zhangblup",
    "zhangen", "pchorvath2013", "pchannum", "pcphenoage", "pcskinandblood",
)
N_ABLATIONS = 5
EXCLUDED_FROM_RANKING = {"yingcausage"}


def load_prediction_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the combined prediction table and return train/test subsets."""
    frame = pd.read_csv(path)
    required = {"dataset", "age", *COMPONENT_CLOCKS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    observed = set(frame["dataset"].dropna())
    expected = set(TRAIN_DATASETS).union(TEST_DATASETS)
    if missing_datasets := expected.difference(observed):
        raise ValueError(f"{path} is missing datasets: {sorted(missing_datasets)}")
    columns = [*COMPONENT_CLOCKS, "dataset", "age"]
    train = frame.loc[frame["dataset"].isin(TRAIN_DATASETS), columns].copy()
    test = frame.loc[frame["dataset"].isin(TEST_DATASETS), columns].copy()
    return train, test


def fit_linear_svr(train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    """Fit the standardized linear SVR used for every ablation endpoint."""
    features = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    age = pd.to_numeric(train["age"], errors="coerce")
    valid = age.notna() & ~features.isna().any(axis=1)
    if not valid.any():
        raise ValueError("No complete training rows remain for the selected clocks")
    # Explicit sklearn defaults document the final model: C=1 and epsilon=0.1.
    model = Pipeline(
        [("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1))]
    )
    return model.fit(features.loc[valid], age.loc[valid])


def single_clock_metrics(test: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-dataset MAE and Pearson r for individual clocks."""
    rows: list[dict[str, float | int | str]] = []
    for dataset, group in test.groupby("dataset", sort=True):
        age = pd.to_numeric(group["age"], errors="coerce")
        for predictor in COMPONENT_CLOCKS:
            prediction = pd.to_numeric(group[predictor], errors="coerce")
            valid = age.notna() & prediction.notna()
            rows.append(
                {
                    "dataset": dataset,
                    "predictor": predictor,
                    "n": int(valid.sum()),
                    "mae": mean_absolute_error(age[valid], prediction[valid]),
                    "pearson_r": age[valid].corr(prediction[valid]),
                }
            )
    return pd.DataFrame(rows)


def rank_component_clocks(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank eligible clocks by ascending mean MAE across the four test sets."""
    ranking = (
        metrics.loc[~metrics["predictor"].isin(EXCLUDED_FROM_RANKING)]
        .groupby("predictor", as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_pearson_r=("pearson_r", "mean"))
        .sort_values(["mean_mae", "predictor"])
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def metaclock_metrics(
    test: pd.DataFrame, feature_columns: list[str], model: Any
) -> pd.DataFrame:
    """Evaluate one retrained MetaClock separately in every test dataset."""
    rows: list[dict[str, float | int | str]] = []
    for dataset, group in test.groupby("dataset", sort=True):
        features = group[feature_columns].apply(pd.to_numeric, errors="coerce")
        age = pd.to_numeric(group["age"], errors="coerce")
        valid = age.notna() & ~features.isna().any(axis=1)
        prediction = np.asarray(model.predict(features.loc[valid])).reshape(-1)
        rows.append(
            {
                "dataset": dataset,
                "predictor": "MetaClock",
                "n": int(valid.sum()),
                "mae": mean_absolute_error(age.loc[valid], prediction),
                "pearson_r": np.corrcoef(age.loc[valid].to_numpy(), prediction)[0, 1],
            }
        )
    return pd.DataFrame(rows)


def run_cumulative_ablation(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    """Remove the five best clocks cumulatively and refit the model each time."""
    base_metrics = single_clock_metrics(test)
    ranking = rank_component_clocks(base_metrics)
    removed: list[str] = []
    summaries: list[dict[str, float | int | str]] = []
    step_tables: list[pd.DataFrame] = []

    for step, removed_clock in enumerate(ranking.head(N_ABLATIONS)["predictor"], start=1):
        removed.append(str(removed_clock))
        retained = [clock for clock in COMPONENT_CLOCKS if clock not in removed]
        model = fit_linear_svr(train, retained)
        meta = metaclock_metrics(test, retained, model)
        retained_metrics = base_metrics[base_metrics["predictor"].isin(retained)]
        step_table = pd.concat([retained_metrics, meta], ignore_index=True)
        step_table["ablation_step"] = step
        step_table["removed_clocks"] = ";".join(removed)
        step_tables.append(step_table)
        summaries.append(
            {
                "ablation_step": step,
                "removed_clock": removed_clock,
                "removed_clocks": ";".join(removed),
                "n_retained_clocks": len(retained),
                "metaclock_mean_mae": meta["mae"].mean(),
                "metaclock_mean_pearson_r": meta["pearson_r"].mean(),
            }
        )
    return pd.DataFrame(summaries), step_tables, ranking


def parse_args() -> argparse.Namespace:
    """Parse the replaceable dataset path and ordered result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=FIGURE_DIR / "datasets" / "all_predicted_ages_with_id.csv",
        help="Replace with your combined component-clock prediction table.",
    )
    parser.add_argument("--results-dir", type=Path, default=FIGURE_DIR / "results")
    return parser.parse_args()


def main() -> None:
    """Run the study and write summary/detail tables in manuscript panel order."""
    args = parse_args()
    train, test = load_prediction_data(args.predictions)
    summary, step_tables, ranking = run_cumulative_ablation(train, test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results_dir / "03_ablation_summary.csv", index=False)
    for result_number, table in enumerate(step_tables, start=4):
        table.to_csv(args.results_dir / f"{result_number:02d}_ablation_step_{result_number - 3:02d}.csv", index=False)
    ranking.to_csv(args.results_dir / "09_single_clock_ranking.csv", index=False)


if __name__ == "__main__":
    main()
