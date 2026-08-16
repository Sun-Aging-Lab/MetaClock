#!/usr/bin/env python3
"""Predict BioMeta, direct-CpG baseline, and DNAmPhenoAge in normalized external cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .config import EXTERNAL_MODELS, FIGURE4, FIGURE4_COMPONENTS
    from .provenance import file_record, validate_all672_model_directory, write_json
    from .table_io import align_matrix, read_feature_matrix, read_table
except ImportError:  # Direct-script fallback.
    from config import EXTERNAL_MODELS, FIGURE4, FIGURE4_COMPONENTS
    from provenance import file_record, validate_all672_model_directory, write_json
    from table_io import align_matrix, read_feature_matrix, read_table


DEFAULT_METADATA = Path("data/figure4/external_sample_metadata.csv")  # fill in your path
DEFAULT_COMPONENTS = Path("data/figure4/external_component_predictions.csv")  # fill in your path
DEFAULT_CPG_MATRIX = Path("data/figure4/external_cpg_matrix.parquet")  # fill in your path
DEFAULT_MODEL_DIR = Path("models/figure4/all672_without_dnamfitage")  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/external_predictions")
KEYS = ["dataset", "disease", "sample_id", "age", "group"]


def main() -> int:
    """Apply validated all-672 artifacts to one normalized 3,325-sample bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--cpg-matrix", type=Path, default=DEFAULT_CPG_MATRIX)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    validate_all672_model_directory(args.model_dir)

    metadata = read_table(args.metadata)
    required = {"sampleID", *KEYS}
    if missing := required.difference(metadata.columns):
        raise KeyError(f"External metadata are missing columns: {sorted(missing)}")
    metadata["sampleID"] = metadata["sampleID"].astype(str)
    expected_n = int(FIGURE4["external_samples"])
    if len(metadata) != expected_n or metadata["sampleID"].nunique() != expected_n:
        raise ValueError("External metadata must contain 3,325 unique samples.")
    if metadata[["dataset", "disease"]].drop_duplicates().shape[0] != int(
        FIGURE4["external_datasets"]
    ):
        raise ValueError("Expected exactly 15 external dataset/disease combinations.")
    if not set(metadata["group"].astype(str)).issubset({"control", "disease"}):
        raise ValueError("External group must be control or disease.")

    component_features = list(FIGURE4_COMPONENTS)
    components = read_feature_matrix(args.components, component_features)
    cpg = read_feature_matrix(args.cpg_matrix)
    cpg_features = (
        pd.read_csv(args.model_dir / "baseline_features.csv")["feature"].astype(str).tolist()
    )
    if [column for column in cpg if column != "sampleID"] != cpg_features:
        raise ValueError(
            "External CpG matrix does not exactly match the baseline feature manifest."
        )
    component_x = align_matrix(metadata, components, component_features)
    cpg_x = align_matrix(metadata, cpg, cpg_features)
    models = {
        "bio_metaclock": (
            joblib.load(args.model_dir / "bio_metaclock.joblib"),
            component_x[component_features],
        ),
        "baseline": (joblib.load(args.model_dir / "baseline.joblib"), cpg_x[cpg_features]),
    }
    rows: list[pd.DataFrame] = []
    for name, (model, design) in models.items():
        frame = metadata[KEYS].copy()
        frame["model"] = name
        frame["pred_age"] = model.predict(design)
        rows.append(frame)
    pheno = metadata[KEYS].copy()
    pheno["model"] = "dnamphenoage"
    pheno["pred_age"] = component_x["dnamphenoage"].to_numpy(float)
    rows.append(pheno)
    predictions = pd.concat(rows, ignore_index=True)
    if tuple(predictions["model"].drop_duplicates()) != EXTERNAL_MODELS:
        raise RuntimeError("External prediction model order differs from the analysis manifest.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "external_disease_predictions.csv"
    predictions.to_csv(output, index=False)
    write_json(
        args.output_dir / "external_prediction_summary.json",
        {
            "models": list(EXTERNAL_MODELS),
            "n_samples": expected_n,
            "model_training_summary": file_record(args.model_dir / "training_summary.json"),
            "inputs": {
                name: file_record(path)
                for name, path in {
                    "metadata": args.metadata,
                    "components": args.components,
                    "cpg_matrix": args.cpg_matrix,
                }.items()
            },
            "output": file_record(output),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
