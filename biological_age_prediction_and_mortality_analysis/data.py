"""Shared InCHIANTI table handling for the Figure 4 workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DNAM_FEATURE_PREFIX_COUNTS,
    DNAM_FEATURE_PREFIXES,
    INCHIANTI_DNAM_FEATURE_COUNT,
)

# PyArrow's default Thrift container limit is too small for a Parquet schema
# containing all 485,577 InCHIANTI array features.
PARQUET_THRIFT_CONTAINER_SIZE_LIMIT = 2_000_000
PARQUET_THRIFT_STRING_SIZE_LIMIT = 2_000_000_000


def read_analysis_dataset(
    path: Path, columns: list[str] | None = None, *, float_precision: str | None = None,
) -> pd.DataFrame:
    """Read CSV or the ultra-wide canonical Parquet table with PyArrow."""
    if not path.exists():
        raise FileNotFoundError(f"Analysis-ready dataset does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(
            path,
            columns=columns,
            engine="pyarrow",
            thrift_container_size_limit=PARQUET_THRIFT_CONTAINER_SIZE_LIMIT,
            thrift_string_size_limit=PARQUET_THRIFT_STRING_SIZE_LIMIT,
        )
    return pd.read_csv(path, usecols=columns, float_precision=float_precision)


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the explicit Clinical PhenoAge completeness flag safely."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


def numeric_matrix(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    dtype: np.dtype | type = np.float32,
    fill_missing: bool = True,
) -> np.ndarray:
    """Build numeric inputs, optionally retaining missing values for a fitted imputer."""
    numeric = (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return (numeric.fillna(0.0) if fill_missing else numeric).to_numpy(dtype)


def dnam_feature_columns(
    frame: pd.DataFrame,
    *,
    require_figure4_manifest: bool = True,
) -> list[str]:
    """Return all Figure 4 baseline features in their original table order."""
    features = [
        str(column)
        for column in frame.columns
        if str(column).startswith(DNAM_FEATURE_PREFIXES)
    ]
    return validate_dnam_feature_names(
        features, require_figure4_manifest=require_figure4_manifest
    )


def validate_dnam_feature_names(
    features: list[str],
    *,
    require_figure4_manifest: bool = True,
) -> list[str]:
    """Validate uniqueness, composition, and formal order of baseline features."""
    if not features:
        raise ValueError(
            f"No DNAm features with prefixes {list(DNAM_FEATURE_PREFIXES)} were found."
        )
    if len(features) != len(set(features)):
        raise ValueError("DNAm feature names must be unique.")
    if require_figure4_manifest:
        observed = {
            prefix: sum(feature.startswith(prefix) for feature in features)
            for prefix in DNAM_FEATURE_PREFIXES
        }
        if (
            len(features) != INCHIANTI_DNAM_FEATURE_COUNT
            or observed != DNAM_FEATURE_PREFIX_COUNTS
        ):
            raise ValueError(
                "DNAm features do not match the Figure 4 InCHIANTI manifest: "
                f"n={len(features)}, prefix_counts={observed}; expected "
                f"n={INCHIANTI_DNAM_FEATURE_COUNT}, "
                f"prefix_counts={DNAM_FEATURE_PREFIX_COUNTS}."
            )
    return features


def _canonical_participant_ids(values: pd.Series) -> pd.Series:
    """Normalize numeric donor IDs so values such as 1 and 1.0 match."""
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    integer = numeric.notna() & np.isfinite(numeric) & numeric.eq(np.floor(numeric))
    text.loc[integer] = numeric.loc[integer].astype("Int64").astype("string")
    if text.isna().any() or text.eq("").any():
        raise ValueError("Participant IDs must be non-empty.")
    return text


def sort_figure4_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Order samples by numeric participant ID and numeric sample suffix."""
    ordered = frame.copy()
    ordered["ID"] = _canonical_participant_ids(ordered["ID"])
    ordered["sampleID"] = ordered["sampleID"].astype(str)
    ordered["_numeric_id"] = pd.to_numeric(ordered["ID"], errors="raise")
    ordered["_numeric_suffix"] = pd.to_numeric(
        ordered["sampleID"].str.extract(r"(\d+)$", expand=False), errors="raise"
    )
    if ordered[["_numeric_id", "_numeric_suffix"]].isna().any().any():
        raise ValueError("Figure 4 requires numeric participant IDs and sample suffixes.")
    return (ordered.sort_values(["_numeric_id", "_numeric_suffix", "sampleID"], kind="mergesort")
            .drop(columns=["_numeric_id", "_numeric_suffix"]).reset_index(drop=True))
