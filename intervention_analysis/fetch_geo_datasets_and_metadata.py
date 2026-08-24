#!/usr/bin/env python3
"""Fetch the exact GEO metadata and DNAm matrices used by the frozen 77 results."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"
DEFAULT_PIPELINE_ROOT = Path("/orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam")


ACQUISITION_MANIFEST = CONFIG_DIR / "matrix_acquisition_manifest_64.csv"
GSE_LIST = CONFIG_DIR / "gse_ids_64.txt"
VALID_RESULTS = CONFIG_DIR / "valid_interventions_77.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GEO metadata plus the exact 64 matrix artifacts used by the "
            "frozen 77-intervention analysis. Existing valid files are reused."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT,
        help="Directory containing the established GEO fetch/download helpers.",
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
        "--dry-run",
        action="store_true",
        help="Validate the frozen manifest and print the acquisition plan without writing.",
    )
    return parser.parse_args()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_frozen_inputs() -> tuple[pd.DataFrame, list[str]]:
    acquisition = pd.read_csv(ACQUISITION_MANIFEST, dtype=str)
    valid = pd.read_csv(VALID_RESULTS, dtype=str)
    gse_ids = [line.strip() for line in GSE_LIST.read_text().splitlines() if line.strip()]
    required = {
        "gse_id",
        "analysis_branch",
        "matrix_file",
        "matrix_format",
        "url",
        "expected_size_bytes",
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
    if set(gse_ids) != set(valid["gse_id"]) or set(gse_ids) != set(acquisition["gse_id"]):
        raise ValueError("GSE membership differs across frozen inputs")
    acquisition["expected_size_bytes"] = pd.to_numeric(
        acquisition["expected_size_bytes"], errors="raise"
    ).astype("int64")
    if acquisition["expected_size_bytes"].le(0).any():
        raise ValueError("Every matrix must have a positive expected size")
    return acquisition, gse_ids


def fetch_metadata(
    gse_ids: list[str],
    output_dir: Path,
    workers: int,
    requests_per_second: float,
    rate_limiter_class: Any,
    fetch_gse_function: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    limiter = rate_limiter_class(requests_per_second)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_gse_function, gse, output_dir, limiter): gse for gse in gse_ids
        }
        for future in as_completed(futures):
            gse = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({"gse_id": gse, "error": str(error)})
    return successes, failures


def fetch_matrix(row: Any, matrix_root: Path, download_function: Any) -> dict[str, Any]:
    output = matrix_root / str(row.matrix_file)
    if output.exists() and output.stat().st_size != int(row.expected_size_bytes):
        raise ValueError(
            f"{row.gse_id}: cached size {output.stat().st_size} differs from "
            f"expected {row.expected_size_bytes}; move the invalid file before retrying"
        )
    size, download_status = download_function(str(row.url), output)
    if size != int(row.expected_size_bytes):
        raise ValueError(
            f"{row.gse_id}: downloaded size {size} differs from {row.expected_size_bytes}"
        )
    return {
        "gse_id": str(row.gse_id),
        "matrix_file": str(row.matrix_file),
        "matrix_format": str(row.matrix_format),
        "url": str(row.url),
        "path": str(output.resolve()),
        "size_bytes": size,
        "status": ("cached" if download_status == "cached_not_rehashed" else "downloaded"),
    }


def main() -> int:
    args = parse_args()
    if args.metadata_workers < 1 or args.matrix_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.requests_per_second <= 0:
        raise ValueError("--requests-per-second must be positive")
    helper_paths = [
        args.pipeline_root / "download_geo_biometaclock_series.py",
        args.pipeline_root / "fetch_geo_review_metadata.py",
    ]
    missing_helpers = [str(path) for path in helper_paths if not path.is_file()]
    if missing_helpers:
        raise FileNotFoundError(f"Missing GEO pipeline helpers: {missing_helpers}")
    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from download_geo_biometaclock_series import download
    from fetch_geo_review_metadata import RateLimiter, fetch_gse

    acquisition, gse_ids = validate_frozen_inputs()
    plan = {
        "n_interventions": 77,
        "n_gse": len(gse_ids),
        "n_matrix_files": len(acquisition),
        "matrix_total_expected_bytes": int(acquisition["expected_size_bytes"].sum()),
        "matrix_total_expected_gib": round(
            float(acquisition["expected_size_bytes"].sum()) / 1024**3, 3
        ),
        "metadata_only": args.metadata_only,
        "output_dir": str(args.output_dir.resolve()),
        "pipeline_root": str(args.pipeline_root.resolve()),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **plan}, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_dir / "metadata"
    matrix_root = args.output_dir / "matrices"
    metadata_results, metadata_failures = fetch_metadata(
        gse_ids,
        metadata_dir,
        workers=args.metadata_workers,
        requests_per_second=args.requests_per_second,
        rate_limiter_class=RateLimiter,
        fetch_gse_function=fetch_gse,
    )

    matrix_results: list[dict[str, Any]] = []
    matrix_failures: list[dict[str, str]] = []
    if not args.metadata_only:
        with ThreadPoolExecutor(max_workers=args.matrix_workers) as executor:
            futures = {
                executor.submit(fetch_matrix, row, matrix_root, download): str(row.gse_id)
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
        "frozen_matrix_manifest": str(ACQUISITION_MANIFEST),
        "frozen_gse_list": str(GSE_LIST),
    }
    write_json_atomic(summary, args.output_dir / "fetch_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if metadata_failures or matrix_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
