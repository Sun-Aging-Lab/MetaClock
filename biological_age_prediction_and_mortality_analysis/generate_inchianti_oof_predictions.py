#!/usr/bin/env python3
"""Generate the five-fold OOF BioMeta and direct-CpG predictions used in Figure 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .config import FIGURE4, FIGURE4_COMPONENTS
    from .modeling import CV_FOLDS, RANDOM_SEED, fit_scaled_elasticnet_cv
    from .provenance import file_record, software_versions, write_json
    from .table_io import align_matrix, read_feature_matrix, read_table
except ImportError:  # Direct-script fallback.
    from config import FIGURE4, FIGURE4_COMPONENTS
    from modeling import CV_FOLDS, RANDOM_SEED, fit_scaled_elasticnet_cv
    from provenance import file_record, software_versions, write_json
    from table_io import align_matrix, read_feature_matrix, read_table


DEFAULT_LABELS = Path("data/inchianti/participant_targets.csv")  # fill in your path
DEFAULT_COMPONENTS = Path("data/inchianti/component_clock_predictions.csv")  # fill in your path
DEFAULT_CPG_MATRIX = Path("data/inchianti/cpg_matrix.parquet")  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/oof_predictions")
TARGET = "clinical_phenoage"


def assign_age_bin_folds(age: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Deterministically balance ten-year age bins across outer folds."""
    bins = (pd.to_numeric(age, errors="raise") // 10 * 10).astype(int)
    folds = np.full(len(age), -1, dtype=int)
    rng = np.random.default_rng(seed)
    for _, indices in pd.Series(np.arange(len(age))).groupby(bins, sort=True):
        shuffled = indices.to_numpy(int)
        rng.shuffle(shuffled)
        if len(shuffled) < n_folds:
            folds[shuffled] = rng.choice(n_folds, size=len(shuffled), replace=False)
        else:
            folds[shuffled] = np.arange(len(shuffled)) % n_folds
    if (folds < 0).any():
        raise RuntimeError("Some samples were not assigned to an OOF fold.")
    return folds


def main() -> int:
    """Train nested fold-scaled models and write one prediction per participant/model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--cpg-matrix", type=Path, default=DEFAULT_CPG_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    labels = read_table(args.labels)
    required = {"ID", "sampleID", "age_years", TARGET}
    if missing := required.difference(labels.columns):
        raise KeyError(f"Labels are missing columns: {sorted(missing)}")
    labels["sampleID"] = labels["sampleID"].astype(str)
    expected_n = int(FIGURE4["participants"])
    if (
        len(labels) != expected_n
        or labels["ID"].nunique() != expected_n
        or labels["sampleID"].nunique() != expected_n
    ):
        raise ValueError("OOF labels must contain exactly 672 unique participant-sample rows.")
    labels["fold"] = assign_age_bin_folds(labels["age_years"], CV_FOLDS, RANDOM_SEED)

    component_features = list(FIGURE4_COMPONENTS)
    components = read_feature_matrix(args.components, component_features)
    cpg = read_feature_matrix(args.cpg_matrix)
    cpg_features = [column for column in cpg.columns if column != "sampleID"]
    if len(cpg_features) != int(FIGURE4["baseline_cpg_features"]):
        raise ValueError("The direct-CpG matrix must contain exactly 485,577 CpGs.")
    designs = {
        "bio_metaclock": align_matrix(labels, components, component_features)[
            component_features
        ].to_numpy(float),
        "baseline": align_matrix(labels, cpg, cpg_features)[cpg_features].to_numpy(float),
    }
    y = pd.to_numeric(labels[TARGET], errors="raise").to_numpy(float)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    alpha_rows: list[dict[str, object]] = []
    for fold in range(CV_FOLDS):
        train = labels["fold"].to_numpy(int) != fold
        test = ~train
        for model_name, design in designs.items():
            fit = fit_scaled_elasticnet_cv(
                design[train], y[train], n_jobs=args.n_jobs, seed=RANDOM_SEED + fold
            )
            joblib.dump(fit.model, args.output_dir / f"{model_name}_fold{fold}.joblib")
            alpha_rows.append(
                {
                    "outer_fold": fold,
                    "model": model_name,
                    "selected_alpha": fit.alpha,
                    "l1_ratio": fit.l1_ratio,
                    "n_iter": fit.n_iter,
                    "best_inner_cv_mse": fit.best_cv_mse,
                    "convergence_warnings": 0,
                }
            )
            frame = labels.loc[test, ["ID", "sampleID", "fold", "age_years", TARGET]].copy()
            frame["model"] = model_name
            frame["pred_bioage"] = fit.model.predict(design[test])
            prediction_rows.append(frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    expected_rows = expected_n * len(designs)
    if len(predictions) != expected_rows or predictions.duplicated(["model", "ID"]).any():
        raise RuntimeError(
            "OOF generation did not produce one prediction per model and participant."
        )
    predictions.to_csv(args.output_dir / "inchianti_5fold_oof_predictions.csv", index=False)
    labels.to_csv(args.output_dir / "oof_fold_assignments.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(args.output_dir / "oof_selected_alphas.csv", index=False)
    write_json(
        args.output_dir / "oof_run_summary.json",
        {
            "analysis": "five-fold participant-level out-of-fold prediction",
            "outer_folds": CV_FOLDS,
            "fold_balance": "round-robin within shuffled ten-year age bins",
            "models": list(designs),
            "n_participants": expected_n,
            "convergence_warnings": 0,
            "inputs": {
                name: file_record(path)
                for name, path in {
                    "labels": args.labels,
                    "components": args.components,
                    "cpg_matrix": args.cpg_matrix,
                }.items()
            },
            "software": software_versions(("joblib", "numpy", "pandas", "scikit-learn")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
