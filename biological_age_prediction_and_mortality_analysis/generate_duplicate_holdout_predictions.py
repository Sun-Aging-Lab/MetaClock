#!/usr/bin/env python3
"""Train outside duplicate participant IDs and predict both samples in each pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .config import FIGURE4, FIGURE4_COMPONENTS
    from .modeling import fit_scaled_elasticnet_cv
    from .provenance import file_record, write_json
    from .table_io import align_matrix, read_feature_matrix, read_table
except ImportError:  # Direct-script fallback.
    from config import FIGURE4, FIGURE4_COMPONENTS
    from modeling import fit_scaled_elasticnet_cv
    from provenance import file_record, write_json
    from table_io import align_matrix, read_feature_matrix, read_table


DEFAULT_LABELS = Path("data/inchianti/all_sample_targets.csv")  # fill in your path
DEFAULT_COMPONENTS = Path(
    "data/inchianti/all_sample_component_predictions.csv"
)  # fill in your path
DEFAULT_CPG_MATRIX = Path("data/inchianti/all_sample_cpg_matrix.parquet")  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/duplicate_holdout")
TARGET = "clinical_phenoage"


def main() -> int:
    """Fit two models on nonduplicate IDs and write duplicate-pair predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--cpg-matrix", type=Path, default=DEFAULT_CPG_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    labels = read_table(args.labels)
    required = {"ID", "sampleID", TARGET}
    if missing := required.difference(labels.columns):
        raise KeyError(f"Labels are missing columns: {sorted(missing)}")
    labels["sampleID"] = labels["sampleID"].astype(str)
    if labels["sampleID"].duplicated().any():
        raise ValueError("Labels contain duplicate sampleID rows.")
    counts = labels.groupby("ID")["sampleID"].nunique()
    if (counts > 2).any():
        raise ValueError("A participant has more than two DNAm samples.")
    holdout_ids = counts[counts.eq(2)].index
    if len(holdout_ids) != int(FIGURE4["duplicate_pairs"]):
        raise ValueError("Expected exactly 44 duplicate participant IDs.")
    train = labels[~labels["ID"].isin(holdout_ids)].dropna(subset=[TARGET]).copy()
    holdout = labels[labels["ID"].isin(holdout_ids)].sort_values(["ID", "sampleID"]).copy()

    component_features = list(FIGURE4_COMPONENTS)
    components = read_feature_matrix(args.components, component_features)
    cpg = read_feature_matrix(args.cpg_matrix)
    cpg_features = [column for column in cpg if column != "sampleID"]
    if len(cpg_features) != int(FIGURE4["baseline_cpg_features"]):
        raise ValueError("The direct-CpG matrix must contain exactly 485,577 CpGs.")
    matrices = {"bio_metaclock": (components, component_features), "baseline": (cpg, cpg_features)}
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(float)
    rows: list[pd.DataFrame] = []
    model_audit: dict[str, object] = {}
    for model_name, (matrix, features) in matrices.items():
        x_train = align_matrix(train, matrix, features)[features].to_numpy(float)
        x_holdout = align_matrix(holdout, matrix, features)[features].to_numpy(float)
        fit = fit_scaled_elasticnet_cv(x_train, y, n_jobs=args.n_jobs)
        frame = holdout[["ID", "sampleID"]].copy()
        frame["model"] = model_name
        frame["prediction"] = fit.model.predict(x_holdout)
        rows.append(frame)
        model_audit[model_name] = {
            "n_train": len(train),
            "n_features": len(features),
            "alpha": fit.alpha,
            "n_iter": fit.n_iter,
            "convergence_warnings": 0,
        }
    predictions = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "duplicate_sample_predictions.csv", index=False)
    write_json(
        args.output_dir / "duplicate_holdout_summary.json",
        {
            "holdout_rule": "all participant IDs with exactly two DNAm samples",
            "n_holdout_ids": len(holdout_ids),
            "models": model_audit,
            "inputs": {
                name: file_record(path)
                for name, path in {
                    "labels": args.labels,
                    "components": args.components,
                    "cpg_matrix": args.cpg_matrix,
                }.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
