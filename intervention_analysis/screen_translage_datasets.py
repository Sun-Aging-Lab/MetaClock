#!/usr/bin/env python3
"""Screen the frozen TranslAGE catalog and register 19 nonredundant analyses."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd

from intervention_analysis.fetch_translage_datasets import (
    CONFIG_DIR,
    DEFAULT_CATALOG_DIR,
    INVENTORY_PATH,
    load_frozen_inventory,
    sha256,
    write_json,
)
from intervention_analysis.paths import relative_path, repo_path

NESTED_AGGREGATES = {
    "E-MTAB-8956_Exercise",
    "E-MTAB-8956_LowCarb",
    "E-MTAB-8956_LowFat",
}
EXCLUDED_DATASETS = NESTED_AGGREGATES | {"GSE191297 (2 weeks)"}
MANIFEST_COLUMNS = [
    "TranslAGE Dataset ID",
    "Public ID",
    "Name",
    "Dataset Type",
    "Intervention/Event Subtype",
    "Time (Days)",
    "Number of Subjects",
    "Number of Samples",
    "Description",
    "Need permission to release",
    "root_url",
    "file_url",
    "file_name",
    "selection_status",
    "selection_reason",
    "target_time_days",
]


def parse_args() -> argparse.Namespace:
    """Parse repository-local catalog and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=repo_path,
        default=DEFAULT_CATALOG_DIR / "dataset_metadata.csv",
    )
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_CATALOG_DIR)
    return parser.parse_args()


def precheck(row: pd.Series) -> tuple[str, str]:
    """Apply the original longitudinal, exposure, responsiveness, and donor gates."""
    if row["Longitudinal or Cross-Sectional"] != "Longitudinal":
        return "excluded_catalog", "not_longitudinal"
    if row["Dataset Type"] not in {"Intervention", "Event"}:
        return "excluded_catalog", "not_intervention_or_event"
    analysis = "" if pd.isna(row["Analysis"]) else str(row["Analysis"])
    if "Responsive" not in {item.strip() for item in analysis.split(",")}:
        return "excluded_catalog", "not_translage_responsive"
    # Missing reported counts contribute zero to this catalog gate; complete
    # donor pairs are counted separately during sample-level validation.
    counts = pd.to_numeric(
        row[["Number of Subjects", "Number of Controls"]], errors="coerce"
    ).fillna(0)
    if counts.max() < 4:
        return "excluded_catalog", "reported_donor_upper_bound_below_minimum"
    if row["Dataset Type"] == "Event":
        return (
            "candidate_manual_event_review",
            "event_requires_mutable_exposure_or_procedure_review",
        )
    return (
        "candidate_sample_level_review",
        "intervention_requires_sample_level_validation",
    )


def screen_catalog(
    catalog_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reproduce 179 → 95 → 23 → 19 without removing prior GEO overlaps."""
    provenance, inventory = load_frozen_inventory()
    if sha256(catalog_path) != provenance["catalog_sha256"]:
        raise ValueError(
            "TranslAGE catalog differs from the frozen snapshot; explicit re-curation is required"
        )
    catalog = pd.read_csv(catalog_path)
    if len(catalog) != 179 or not catalog["TranslAGE Dataset ID"].is_unique:
        raise ValueError("Frozen catalog must contain 179 unique dataset IDs")
    statuses = catalog.apply(precheck, axis=1)
    catalog["screen_status"] = [status for status, _ in statuses]
    catalog["screen_reason"] = [reason for _, reason in statuses]
    catalog["sample_level_validation_required"] = catalog[
        "screen_status"
    ].str.startswith("candidate_")
    previous_gses = set(
        pd.read_csv(CONFIG_DIR / "matrix_acquisition_manifest_64.csv")["gse_id"]
    )
    catalog["overlaps_previous_geo"] = catalog["Public ID"].isin(previous_gses)
    candidates = catalog.loc[catalog["sample_level_validation_required"]].copy()
    candidates = candidates.sort_values(
        ["Dataset Type", "Public ID", "TranslAGE Dataset ID"]
    ).reset_index(drop=True)
    candidates.insert(0, "candidate_index", range(len(candidates)))
    if len(candidates) != 95 or not set(inventory["TranslAGE Dataset ID"]).issubset(
        candidates["TranslAGE Dataset ID"]
    ):
        raise ValueError("Frozen candidate or official-cleaned membership changed")

    direct = candidates.merge(
        inventory,
        on=["TranslAGE Dataset ID", "Public ID"],
        how="inner",
        validate="one_to_one",
    )
    if len(direct) != 23:
        raise ValueError(
            "Expected 23 catalog candidates with official cleaned matrices"
        )
    direct["selection_status"] = "selected_sample_level_validation"
    direct["selection_reason"] = (
        "official cleaned beta matrix; unique pre-specified intervention concept"
    )
    mask = direct["TranslAGE Dataset ID"].eq("GSE191297 (2 weeks)")
    direct.loc[mask, "selection_status"] = "excluded_concept_duplicate"
    direct.loc[mask, "selection_reason"] = (
        "same study/intervention as 6-week row; longest follow-up pre-selected"
    )
    mask = direct["TranslAGE Dataset ID"].isin(NESTED_AGGREGATES)
    direct.loc[mask, "selection_status"] = "excluded_nested_aggregate"
    direct.loc[mask, "selection_reason"] = (
        "pooled samples overlap four disjoint diet-by-exercise arm files"
    )
    direct["target_time_days"] = float("nan")
    direct.loc[
        direct["TranslAGE Dataset ID"].eq("GSE191297 (6 weeks)"), "target_time_days"
    ] = 42.0

    selected = (
        direct.loc[direct["selection_status"].eq("selected_sample_level_validation")]
        .sort_values("TranslAGE Dataset ID")
        .reset_index(drop=True)
    )
    manifest = selected[MANIFEST_COLUMNS].copy()
    manifest.insert(0, "analysis_index", range(len(selected)))
    manifest["concept_id"] = manifest["TranslAGE Dataset ID"]
    manifest["expected_cleaned_bytes"] = selected["content_length_bytes"].astype(
        "int64"
    )
    manifest["cleaned_sha256"] = selected["cleaned_sha256"]
    manifest["overlaps_previous_geo"] = selected["overlaps_previous_geo"]
    if (
        len(manifest) != 19
        or manifest["Public ID"].nunique() != 13
        or manifest["cleaned_sha256"].isna().any()
    ):
        raise ValueError("Expected 19 fingerprinted analyses from 13 public accessions")
    if set(manifest.loc[manifest["overlaps_previous_geo"], "Public ID"]) != {
        "GSE201532",
        "GSE74548",
    }:
        raise ValueError("Expected prior-GEO overlap only at GSE201532 and GSE74548")

    selection_columns = ["TranslAGE Dataset ID", "selection_status", "selection_reason"]
    audit = catalog.merge(
        direct[selection_columns],
        on="TranslAGE Dataset ID",
        how="left",
        validate="one_to_one",
    )
    unavailable = (
        audit["sample_level_validation_required"] & audit["selection_status"].isna()
    )
    audit.loc[unavailable, "selection_status"] = (
        "excluded_no_frozen_official_cleaned_file"
    )
    audit.loc[unavailable, "selection_reason"] = (
        "not among the 23 reviewed official cleaned RData files"
    )
    excluded = ~audit["sample_level_validation_required"]
    audit.loc[excluded, "selection_status"] = audit.loc[excluded, "screen_status"]
    audit.loc[excluded, "selection_reason"] = audit.loc[excluded, "screen_reason"]
    summary = {
        "snapshot_date": provenance["snapshot_date"],
        "source_catalog_sha256": provenance["catalog_sha256"],
        "inventory_sha256": sha256(INVENTORY_PATH),
        "catalog_rows": len(catalog),
        "candidate_rows": len(candidates),
        "official_cleaned_rows": len(direct),
        "selected_analyses": len(manifest),
        "selected_public_accessions": int(manifest["Public ID"].nunique()),
        "candidate_previous_geo_accessions": sorted(
            set(candidates.loc[candidates["overlaps_previous_geo"], "Public ID"])
        ),
        "selected_previous_geo_accessions": ["GSE201532", "GSE74548"],
        "cross_source_deduplication": "Retain both source analyses here; replace the two prior GEO results with TranslAGE at combined BH correction.",
        "selection_frozen_before_prediction": True,
        "sample_level_validation_required": True,
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    return audit, candidates, manifest, summary


def main() -> int:
    """Write the complete catalog audit and the registered analysis manifest."""
    args = parse_args()
    audit, candidates, manifest, summary = screen_catalog(args.catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "dataset_screen_audit.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_datasets.csv", index=False)
    manifest_path = args.output_dir / "official_analysis_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.update(
        {
            "catalog": relative_path(args.catalog),
            "manifest": relative_path(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        }
    )
    write_json(summary, args.output_dir / "screen_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
