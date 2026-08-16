"""Validated readers for normalized sample-by-feature matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read Parquet or CSV/CSV.GZ according to the filename suffix."""
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def read_feature_matrix(path: Path, expected_features: list[str] | None = None) -> pd.DataFrame:
    """Read a matrix with exactly one row per sampleID and optional exact features."""
    table = read_table(path)
    if "sampleID" not in table:
        raise KeyError(f"{path} is missing sampleID.")
    table["sampleID"] = table["sampleID"].astype(str)
    if table["sampleID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sampleID rows.")
    features = [str(column) for column in table.columns if column != "sampleID"]
    if expected_features is not None and features != expected_features:
        raise ValueError(f"{path} does not have the exact expected ordered features.")
    return table


def align_matrix(samples: pd.DataFrame, matrix: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One-to-one align a feature matrix to an ordered sample table."""
    aligned = samples[["sampleID"]].merge(
        matrix[["sampleID", *features]], on="sampleID", how="left", validate="one_to_one"
    )
    if aligned[features].isna().any().any():
        raise ValueError("At least one requested sample has missing feature values.")
    return aligned
