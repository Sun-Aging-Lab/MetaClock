#!/usr/bin/env python3
"""Fetch the TranslAGE catalog or its reviewed official cleaned matrices."""

from __future__ import annotations

import argparse
import json
import platform
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from intervention_analysis.paths import (
    DEFAULT_TRANSLAGE_DATA_ROOT,
    DEFAULT_TRANSLAGE_MANIFEST,
    relative_path,
    repo_child,
    repo_path,
)

CONFIG_DIR = Path(__file__).resolve().parent / "config"
PROVENANCE_PATH = CONFIG_DIR / "translage_provenance.json"
INVENTORY_PATH = CONFIG_DIR / "translage_cleaned_inventory.csv"
DEFAULT_CATALOG_DIR = DEFAULT_TRANSLAGE_MANIFEST.parent
NESTED_AGGREGATES = {
    "E-MTAB-8956_Exercise",
    "E-MTAB-8956_LowCarb",
    "E-MTAB-8956_LowFat",
}
EXCLUDED_DATASETS = NESTED_AGGREGATES | {"GSE191297 (2 weeks)"}


def parse_args() -> argparse.Namespace:
    """Parse the catalog and matrices download stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    catalog = commands.add_parser(
        "catalog", help="Fetch the catalog used to locate reviewed datasets."
    )
    catalog.add_argument("--output-dir", type=repo_path, default=DEFAULT_CATALOG_DIR)
    matrices = commands.add_parser(
        "matrices", help="Fetch selected official cleaned RData."
    )
    matrices.add_argument(
        "--manifest", type=repo_path, default=DEFAULT_TRANSLAGE_MANIFEST
    )
    matrices.add_argument(
        "--data-root", type=repo_path, default=DEFAULT_TRANSLAGE_DATA_ROOT
    )
    matrices.add_argument("--analysis-index", type=int, nargs="+")
    return parser.parse_args()


def safe_id(value: str) -> str:
    """Use the original analysis directory naming convention."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_frozen_inventory() -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the reviewed dataset IDs and their source locations."""
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    inventory = pd.read_csv(INVENTORY_PATH)
    validate_inventory(inventory)
    return provenance, inventory


def validate_inventory(inventory: pd.DataFrame) -> None:
    """Require the 23 reviewed records and usable dataset/source identifiers."""
    required = ["TranslAGE Dataset ID", "Public ID", "file_url", "file_name"]
    if missing := set(required).difference(inventory.columns):
        raise ValueError(f"TranslAGE inventory is missing columns: {sorted(missing)}")
    if inventory[required].isna().any().any() or any(
        inventory[column].astype(str).str.strip().eq("").any()
        for column in required
    ):
        raise ValueError("TranslAGE inventory identifiers and source locations must be nonempty")
    if len(inventory) != 23 or not inventory["TranslAGE Dataset ID"].is_unique:
        raise ValueError(
            "TranslAGE inventory must contain 23 unique reviewed dataset IDs"
        )


def validate_manifest(
    manifest: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    """Match the 19 analysis records to reviewed IDs, independent of row order."""
    validate_inventory(inventory)
    required = [
        "TranslAGE Dataset ID", "Public ID", "file_url", "file_name",
        "analysis_index", "concept_id",
    ]
    if missing := set(required).difference(manifest.columns):
        raise ValueError(f"TranslAGE manifest is missing columns: {sorted(missing)}")
    if manifest[required].isna().any().any() or any(
        manifest[column].astype(str).str.strip().eq("").any()
        for column in required
    ):
        raise ValueError("TranslAGE manifest identifiers and source locations must be nonempty")
    expected_ids = set(inventory["TranslAGE Dataset ID"]) - EXCLUDED_DATASETS
    observed_ids = set(manifest["TranslAGE Dataset ID"])
    if len(manifest) != 19 or not manifest["TranslAGE Dataset ID"].is_unique:
        raise ValueError("TranslAGE manifest must contain 19 unique analysis IDs")
    if observed_ids != expected_ids:
        raise ValueError(
            f"TranslAGE analysis IDs differ: missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )
    if manifest["Public ID"].nunique() != 13:
        raise ValueError("TranslAGE manifest must contain 13 public accessions")
    result = manifest.copy()
    indices = pd.to_numeric(result["analysis_index"], errors="raise")
    if not indices.is_unique or set(indices) != set(range(19)):
        raise ValueError("TranslAGE analysis_index must contain each integer 0..18 once")
    result["analysis_index"] = indices.astype(int)
    if not result["concept_id"].eq(result["TranslAGE Dataset ID"]).all():
        raise ValueError("TranslAGE concept_id must match its reviewed dataset ID")
    observed = result.set_index("TranslAGE Dataset ID").sort_index()
    expected = inventory.set_index("TranslAGE Dataset ID").loc[observed.index]
    for column in ("Public ID", "file_url", "file_name"):
        if not observed[column].eq(expected[column]).all():
            raise ValueError(f"TranslAGE manifest differs from reviewed inventory: {column}")
    return result.sort_values("analysis_index").reset_index(drop=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write one JSON provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def download_file(url: str, output: Path) -> str:
    """Download atomically, or reuse an existing nonempty file."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "www.translage.io",
        "files.dborr.us",
    }:
        raise ValueError(f"Unexpected TranslAGE download URL: {url}")
    cached = output.is_file()
    temporary = output.with_suffix(output.suffix + ".part")
    if cached:
        if output.stat().st_size == 0:
            raise ValueError(f"Empty TranslAGE file: {relative_path(output)}")
        return "cached"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "MetaClock-TranslAGE/1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            temporary.open("wb") as handle,
        ):
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                handle.write(block)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty TranslAGE download: {url}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def fetch_catalog(output_dir: Path) -> dict[str, Any]:
    """Fetch the catalog; screening selects only the reviewed analysis IDs."""
    provenance, _ = load_frozen_inventory()
    output = output_dir / "dataset_metadata.csv"
    status = download_file(provenance["catalog_url"], output)
    summary = {
        "source_url": provenance["catalog_url"],
        "snapshot_date": provenance["snapshot_date"],
        "path": relative_path(output),
        "downloaded_bytes": output.stat().st_size,
        "status": status,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
    }
    write_json(summary, output_dir / "catalog_download_provenance.json")
    return summary


def fetch_matrices(
    manifest_path: Path, data_root: Path, indices: list[int] | None
) -> dict[str, Any]:
    """Fetch the screened 19 analyses, optionally selecting explicit task indices."""
    provenance, inventory = load_frozen_inventory()
    manifest = validate_manifest(pd.read_csv(manifest_path), inventory)
    if indices is not None:
        unknown = sorted(set(indices) - set(manifest["analysis_index"]))
        if unknown:
            raise ValueError(f"Unknown analysis indices: {unknown}")
        manifest = manifest.loc[manifest["analysis_index"].isin(indices)]
    records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        dataset_id = str(row["TranslAGE Dataset ID"])
        output = repo_child(data_root / safe_id(dataset_id), str(row["file_name"]))
        status = download_file(str(row["file_url"]), output)
        record = {
            "analysis_index": int(row["analysis_index"]),
            "translage_dataset_id": dataset_id,
            "source_url": str(row["file_url"]),
            "downloaded_path": relative_path(output),
            "downloaded_bytes": output.stat().st_size,
            "status": status,
        }
        write_json(record, output.parent / "download_provenance.json")
        records.append(record)
    summary = {
        "snapshot_date": provenance["snapshot_date"],
        "manifest": relative_path(manifest_path),
        "n_downloaded_or_cached": len(records),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "records": records,
    }
    write_json(summary, data_root / "download_summary.json")
    return summary


def main() -> int:
    """Execute the requested download stage."""
    args = parse_args()
    summary = (
        fetch_catalog(args.output_dir)
        if args.stage == "catalog"
        else fetch_matrices(args.manifest, args.data_root, args.analysis_index)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
