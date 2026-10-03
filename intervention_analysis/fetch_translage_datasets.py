#!/usr/bin/env python3
"""Fetch the frozen TranslAGE catalog or its selected official cleaned matrices."""

from __future__ import annotations

import argparse
import hashlib
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


def parse_args() -> argparse.Namespace:
    """Parse the catalog and matrices download stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    catalog = commands.add_parser(
        "catalog", help="Fetch and verify the frozen catalog."
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


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    """Use the original analysis directory naming convention."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_frozen_inventory() -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the reviewed inventory and verify its recorded content hash."""
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if sha256(INVENTORY_PATH) != provenance["inventory_sha256"]:
        raise ValueError("Frozen TranslAGE inventory hash mismatch")
    inventory = pd.read_csv(INVENTORY_PATH)
    if len(inventory) != 23 or not inventory["TranslAGE Dataset ID"].is_unique:
        raise ValueError(
            "Frozen TranslAGE inventory must contain 23 unique dataset IDs"
        )
    return provenance, inventory


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write one JSON provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def download_verified(
    url: str, output: Path, expected_bytes: int, expected_sha256: str
) -> str:
    """Download atomically, or reuse a file only when its size and hash match."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "www.translage.io",
        "files.dborr.us",
    }:
        raise ValueError(f"Unexpected TranslAGE download URL: {url}")
    cached = output.is_file()
    temporary = output.with_suffix(output.suffix + ".part")
    if not cached:
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
    checked = output if cached else temporary
    if checked.stat().st_size != expected_bytes or sha256(checked) != expected_sha256:
        raise ValueError(
            f"Frozen size/SHA256 mismatch for {relative_path(checked)}; review source drift"
        )
    if not cached:
        temporary.replace(output)
    return "cached" if cached else "downloaded"


def fetch_catalog(output_dir: Path) -> dict[str, Any]:
    """Fetch the exact catalog snapshot used to define the analysis family."""
    provenance, _ = load_frozen_inventory()
    output = output_dir / "dataset_metadata.csv"
    status = download_verified(
        provenance["catalog_url"],
        output,
        provenance["catalog_bytes"],
        provenance["catalog_sha256"],
    )
    summary = {
        "source_url": provenance["catalog_url"],
        "snapshot_date": provenance["snapshot_date"],
        "path": relative_path(output),
        "sha256": provenance["catalog_sha256"],
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
    from intervention_analysis.screen_translage_datasets import EXCLUDED_DATASETS

    provenance, inventory = load_frozen_inventory()
    manifest = pd.read_csv(manifest_path)
    expected_ids = sorted(set(inventory["TranslAGE Dataset ID"]) - EXCLUDED_DATASETS)
    if manifest["TranslAGE Dataset ID"].tolist() != expected_ids or manifest[
        "analysis_index"
    ].tolist() != list(range(19)):
        raise ValueError(
            "Manifest must contain the frozen 19 analyses in their registered order"
        )
    checked_columns = ["Public ID", "file_url", "file_name", "cleaned_sha256"]
    expected = inventory.set_index("TranslAGE Dataset ID").loc[expected_ids]
    for column in checked_columns:
        if manifest[column].tolist() != expected[column].tolist():
            raise ValueError(f"Manifest differs from frozen inventory: {column}")
    if (
        manifest["expected_cleaned_bytes"].tolist()
        != expected["content_length_bytes"].tolist()
    ):
        raise ValueError(
            "Manifest differs from frozen inventory: expected_cleaned_bytes"
        )
    if indices is not None:
        unknown = sorted(set(indices) - set(manifest["analysis_index"]))
        if unknown:
            raise ValueError(f"Unknown analysis indices: {unknown}")
        manifest = manifest.loc[manifest["analysis_index"].isin(indices)]
    records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        dataset_id = str(row["TranslAGE Dataset ID"])
        output = repo_child(data_root / safe_id(dataset_id), str(row["file_name"]))
        status = download_verified(
            str(row["file_url"]),
            output,
            int(row["expected_cleaned_bytes"]),
            str(row["cleaned_sha256"]),
        )
        record = {
            "analysis_index": int(row["analysis_index"]),
            "translage_dataset_id": dataset_id,
            "source_url": str(row["file_url"]),
            "source_expected_bytes": int(row["expected_cleaned_bytes"]),
            "downloaded_path": relative_path(output),
            "downloaded_bytes": output.stat().st_size,
            "sha256": str(row["cleaned_sha256"]),
            "status": status,
        }
        write_json(record, output.parent / "download_provenance.json")
        records.append(record)
    summary = {
        "snapshot_date": provenance["snapshot_date"],
        "manifest": relative_path(manifest_path),
        "manifest_sha256": sha256(manifest_path),
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
