#!/usr/bin/env python3
"""Fetch GEO metadata and DNAm matrices for the reviewed 77 comparisons."""

from __future__ import annotations

import argparse
import json
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from generalization.geo_data import RemoteFile, download_file
from intervention_analysis.geo_fetch import RateLimiter, fetch_gse
from intervention_analysis.paths import (
    DEFAULT_FETCH_DIR,
    relative_path,
    repo_child,
    repo_path,
)

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"


ACQUISITION_MANIFEST = CONFIG_DIR / "matrix_acquisition_manifest_64.csv"
GSE_LIST = CONFIG_DIR / "gse_ids_64.txt"
VALID_RESULTS = CONFIG_DIR / "valid_interventions_77.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GEO metadata and matrices for the reviewed 64 GEO datasets "
            "in the 77-intervention analysis. Existing nonempty files are reused."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=repo_path,
        default=DEFAULT_FETCH_DIR,
        metavar="REPO_RELATIVE_PATH",
        help="Repository-relative output directory (default: data/intervention/geo_fetch_77).",
    )
    parser.add_argument("--metadata-workers", type=int, default=3)
    parser.add_argument("--matrix-workers", type=int, default=2)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch metadata but do not download matrix files.",
    )
    parser.add_argument(
        "--only-gse",
        nargs="+",
        help="Fetch a subset of frozen GSE accessions for debugging.",
    )
    return parser.parse_args()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_frozen_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Cross-check the frozen result, accession, and acquisition manifests."""
    acquisition = pd.read_csv(ACQUISITION_MANIFEST, dtype=str)
    valid = pd.read_csv(VALID_RESULTS, dtype=str)
    gse_ids = [
        line.strip() for line in GSE_LIST.read_text().splitlines() if line.strip()
    ]
    required = {
        "gse_id",
        "analysis_branch",
        "matrix_file",
        "matrix_format",
        "url",
    }
    missing = required.difference(acquisition.columns)
    if missing:
        raise KeyError(f"Acquisition manifest is missing columns: {sorted(missing)}")
    if len(valid) != 77 or valid["gse_id"].nunique() != 64:
        raise ValueError("Frozen result anchor is not 77 comparisons / 64 GSE")
    if len(acquisition) != 64 or not acquisition["gse_id"].is_unique:
        raise ValueError("Matrix acquisition manifest must have one row per 64 GSE")
    if len(gse_ids) != 64 or len(set(gse_ids)) != 64:
        raise ValueError("Frozen GSE list must contain 64 unique accessions")
    if set(gse_ids) != set(valid["gse_id"]) or set(gse_ids) != set(
        acquisition["gse_id"]
    ):
        raise ValueError("GSE membership differs across frozen inputs")
    if acquisition[list(required)].isna().any().any() or any(
        acquisition[column].str.strip().eq("").any() for column in required
    ):
        raise ValueError("Acquisition dataset IDs, matrix locations and formats must be nonempty")
    return acquisition, gse_ids


def fetch_metadata(
    gse_ids: list[str],
    output_dir: Path,
    workers: int,
    requests_per_second: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Fetch GEO metadata concurrently and return separate success/failure records."""
    limiter = RateLimiter(requests_per_second)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_gse, gse, output_dir, limiter): gse for gse in gse_ids
        }
        for future in as_completed(futures):
            gse = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({"gse_id": gse, "error": str(error)})
    return successes, failures


def fetch_matrix(row: Any, matrix_root: Path) -> dict[str, Any]:
    """Download or reuse a nonempty matrix from the reviewed source location."""
    output = repo_child(matrix_root, str(row.matrix_file))
    cached = output.is_file()
    downloaded = download_file(RemoteFile(output.name, str(row.url)), output)
    if not downloaded.is_file():
        raise FileNotFoundError(f"Missing downloaded matrix for {row.gse_id}")
    size = downloaded.stat().st_size
    if size == 0:
        raise ValueError(f"Empty matrix for {row.gse_id}: {relative_path(downloaded)}")
    return {
        "gse_id": str(row.gse_id),
        "matrix_file": str(row.matrix_file),
        "matrix_format": str(row.matrix_format),
        "url": str(row.url),
        "path": relative_path(output),
        "size_bytes": size,
        "status": "cached" if cached else "downloaded",
    }


def main() -> int:
    args = parse_args()
    if args.metadata_workers < 1 or args.matrix_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.requests_per_second <= 0:
        raise ValueError("--requests-per-second must be positive")
    acquisition, gse_ids = validate_frozen_inputs()
    if args.only_gse:
        requested = set(args.only_gse)
        unknown = sorted(requested.difference(gse_ids))
        if unknown:
            raise ValueError(f"Unknown frozen GSE accessions: {unknown}")
        gse_ids = [gse for gse in gse_ids if gse in requested]
        acquisition = acquisition.loc[acquisition["gse_id"].isin(requested)].copy()
    plan = {
        "n_interventions": int(
            pd.to_numeric(acquisition["n_final_comparisons"], errors="raise").sum()
        ),
        "n_gse": len(gse_ids),
        "n_matrix_files": len(acquisition),
        "metadata_only": args.metadata_only,
        "output_dir": relative_path(args.output_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_dir / "metadata"
    matrix_root = args.output_dir / "matrices"
    metadata_results, metadata_failures = fetch_metadata(
        gse_ids,
        metadata_dir,
        workers=args.metadata_workers,
        requests_per_second=args.requests_per_second,
    )

    matrix_results: list[dict[str, Any]] = []
    matrix_failures: list[dict[str, str]] = []
    if not args.metadata_only:
        with ThreadPoolExecutor(max_workers=args.matrix_workers) as executor:
            futures = {
                executor.submit(fetch_matrix, row, matrix_root): str(row.gse_id)
                for row in acquisition.itertuples(index=False)
            }
            for future in as_completed(futures):
                gse = futures[future]
                try:
                    matrix_results.append(future.result())
                except Exception as error:
                    matrix_failures.append({"gse_id": gse, "error": str(error)})

    if matrix_results:
        pd.DataFrame(matrix_results).sort_values("gse_id").to_csv(
            args.output_dir / "matrix_download_inventory.csv", index=False
        )
    summary = {
        **plan,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "metadata_success": len(metadata_results),
        "metadata_failures": sorted(metadata_failures, key=lambda row: row["gse_id"]),
        "matrix_success": len(matrix_results),
        "matrix_failures": sorted(matrix_failures, key=lambda row: row["gse_id"]),
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
        "frozen_matrix_manifest": relative_path(ACQUISITION_MANIFEST),
        "frozen_gse_list": relative_path(GSE_LIST),
    }
    write_json_atomic(summary, args.output_dir / "fetch_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if metadata_failures or matrix_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
