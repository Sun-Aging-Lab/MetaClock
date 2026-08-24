"""Shared InCHIANTI table handling for the Figure 4 workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DNAM_FEATURE_PREFIX_COUNTS,
    DNAM_FEATURE_PREFIXES,
    INCHIANTI_DNAM_FEATURE_COUNT,
    INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH,
    INCHIANTI_OOF_FOLDS,
    INCHIANTI_PARTICIPANTS,
)

# PyArrow's default Thrift container limit is too small for a Parquet schema
# containing all 485,577 InCHIANTI array features.
PARQUET_THRIFT_CONTAINER_SIZE_LIMIT = 2_000_000
PARQUET_THRIFT_STRING_SIZE_LIMIT = 2_000_000_000


def read_analysis_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
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
    return pd.read_csv(path, usecols=columns)


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
) -> np.ndarray:
    """Build a numeric matrix using the formal zero-imputation policy."""
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype)
    )


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


def select_one_sample_per_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest numeric sample-ID suffix for every participant."""
    selected = frame.copy()
    suffix = pd.to_numeric(
        selected["sampleID"].astype(str).str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    )
    selected["_sample_suffix"] = suffix.fillna(np.inf)
    return (
        selected.sort_values(["ID", "_sample_suffix", "sampleID"], kind="mergesort")
        .drop_duplicates("ID", keep="first")
        .drop(columns="_sample_suffix")
        .reset_index(drop=True)
    )


def _canonical_participant_ids(values: pd.Series) -> pd.Series:
    """Normalize numeric donor IDs so values such as 1 and 1.0 match."""
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    integer = numeric.notna() & np.isfinite(numeric) & numeric.eq(np.floor(numeric))
    text.loc[integer] = numeric.loc[integer].astype("Int64").astype("string")
    if text.isna().any() or text.eq("").any():
        raise ValueError("Participant IDs must be non-empty.")
    return text


def fixed_figure4_oof_folds(
    frame: pd.DataFrame,
    path: Path = INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH,
) -> np.ndarray:
    """Map every sample to the versioned donor-level Figure 4 OOF assignment."""
    assignment = pd.read_csv(path, dtype={"ID": "string"})
    if set(assignment.columns) != {"ID", "fold"}:
        raise ValueError("Figure 4 OOF fold assignment must contain exactly ID and fold.")
    assignment["ID"] = _canonical_participant_ids(assignment["ID"])
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise")
    if (
        len(assignment) != INCHIANTI_PARTICIPANTS
        or assignment["ID"].nunique() != INCHIANTI_PARTICIPANTS
        or assignment["ID"].duplicated().any()
    ):
        raise ValueError(
            f"Figure 4 OOF fold assignment must contain {INCHIANTI_PARTICIPANTS} unique IDs."
        )
    if not assignment["fold"].isin(range(INCHIANTI_OOF_FOLDS)).all():
        raise ValueError(
            f"Figure 4 OOF folds must be integers from 0 to {INCHIANTI_OOF_FOLDS - 1}."
        )
    assignment["fold"] = assignment["fold"].astype(int)

    frame_ids = _canonical_participant_ids(frame["ID"])
    observed_ids = set(frame_ids)
    expected_ids = set(assignment["ID"])
    if observed_ids != expected_ids:
        raise ValueError(
            "Input donors do not match the fixed Figure 4 OOF assignment; "
            f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}."
        )
    fold_by_id = assignment.set_index("ID")["fold"]
    folds = frame_ids.map(fold_by_id).to_numpy(dtype=int)
    if frame.assign(_fixed_fold=folds).groupby("ID")["_fixed_fold"].nunique().max() != 1:
        raise RuntimeError("A donor was mapped to more than one fixed Figure 4 OOF fold.")
    return folds
