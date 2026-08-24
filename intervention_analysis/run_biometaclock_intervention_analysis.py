#!/usr/bin/env python3
"""Predict, test, BH-adjust, and plot the frozen 77 GEO interventions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biological_age_prediction_and_mortality_analysis.published_biometaclock import (
    load_published_biometaclock,
)
from intervention_analysis.paths import (
    DEFAULT_MATRIX_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_PYAGING_DIR,
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    relative_path,
    repo_child,
    repo_path,
)
from intervention_analysis.matplotlib_runtime import MPLCONFIGDIR

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"
REFERENCE_RESULTS = CONFIG_DIR / "valid_interventions_77.csv"
SELECTION_AUDIT = CONFIG_DIR / "selection_audit_77.csv"
ACQUISITION_MANIFEST = CONFIG_DIR / "matrix_acquisition_manifest_64.csv"
BRANCH_CONFIG = {
    "age_corrected": {
        "sample_manifest": CONFIG_DIR / "age_corrected_sample_manifest.csv",
        "matrix_manifest": CONFIG_DIR / "age_corrected_matrix_manifest.csv",
        "result_source": "chronological_age_available_age_corrected",
    },
    "paired_no_age": {
        "sample_manifest": CONFIG_DIR / "paired_no_age_sample_manifest.csv",
        "matrix_manifest": CONFIG_DIR / "paired_no_age_matrix_manifest.csv",
        "result_source": "fully_paired_no_chronological_age_no_correction",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the frozen BioMetaClock weights, predict each of the 64 GEO "
            "datasets, run design-aware two-sided tests, recompute BH across "
            "the 77 interventions, and create the FDR volcano plot."
        )
    )
    parser.add_argument(
        "--matrix-root",
        type=repo_path,
        default=DEFAULT_MATRIX_ROOT,
        metavar="REPO_RELATIVE_PATH",
        help=(
            "Repository-relative matrix directory produced by the fetch entry point "
            "(default: data/intervention/geo_fetch_77/matrices)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=repo_path,
        default=DEFAULT_RESULTS_DIR,
        metavar="REPO_RELATIVE_PATH",
        help=(
            "Repository-relative results directory "
            "(default: results/intervention/biometaclock_77)."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=repo_path,
        default=DEFAULT_MODEL_DIR,
        metavar="REPO_RELATIVE_PATH",
        help="Repository-relative directory containing the tracked BioMetaClock weights.",
    )
    parser.add_argument(
        "--pyaging-dir",
        type=repo_path,
        default=DEFAULT_PYAGING_DIR,
        metavar="REPO_RELATIVE_PATH",
        help="Repository-relative cache for PyAging component clocks.",
    )
    parser.add_argument("--pyaging-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--only-gse",
        nargs="+",
        help="Run a subset for debugging; final BH/volcano are produced only for all 64 GSE.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed GSE shards in this output directory.",
    )
    return parser.parse_args()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    model, _ = load_published_biometaclock(args.model_dir)
    reference = pd.read_csv(REFERENCE_RESULTS)
    acquisition = pd.read_csv(ACQUISITION_MANIFEST, dtype=str)
    if len(reference) != 77 or reference["gse_id"].nunique() != 64:
        raise ValueError("Frozen reference must contain 77 comparisons / 64 GSE")
    if not reference["comparison_id"].is_unique:
        raise ValueError("Frozen reference comparison IDs are not unique")
    if len(acquisition) != 64 or not acquisition["gse_id"].is_unique:
        raise ValueError("Acquisition manifest must contain exactly 64 unique GSE")
    if set(reference["gse_id"]) != set(acquisition["gse_id"]):
        raise ValueError("Reference and matrix acquisition GSE sets differ")
    selected_gses = (
        set(args.only_gse) if args.only_gse else set(acquisition["gse_id"].astype(str))
    )
    unknown = sorted(selected_gses.difference(acquisition["gse_id"].astype(str)))
    if unknown:
        raise ValueError(f"Unknown frozen GSE accessions: {unknown}")
    for row in acquisition.itertuples(index=False):
        if str(row.gse_id) not in selected_gses:
            continue
        path = repo_child(args.matrix_root, str(row.matrix_file))
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing matrix for {row.gse_id}: {relative_path(path)}"
            )
        if path.stat().st_size != int(row.expected_size_bytes):
            raise ValueError(
                f"{row.gse_id}: matrix size {path.stat().st_size} differs from "
                f"expected {row.expected_size_bytes}"
            )
    for config in BRANCH_CONFIG.values():
        for key in ("sample_manifest", "matrix_manifest"):
            if not config[key].is_file():
                raise FileNotFoundError(relative_path(config[key]))
    return reference, acquisition, model.weights_path


def expected_ids_by_gse(reference: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(gse): set(group["comparison_id"].astype(str))
        for gse, group in reference.groupby("gse_id", sort=False)
    }


def completed_shard_is_valid(
    shard_dir: Path, expected_ids: set[str], model_path: Path
) -> bool:
    summary_path = shard_dir / "run_summary.json"
    results_path = shard_dir / "comparison_results.csv"
    if not summary_path.is_file() or not results_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        results = pd.read_csv(results_path, dtype=str)
    except Exception:
        return False
    metadata_path = model_path.parent / "biometaclock_672_participants_metadata.json"
    return (
        str(summary.get("model_weights", "")) == relative_path(model_path)
        and str(summary.get("model_metadata", "")) == relative_path(metadata_path)
        and str(summary.get("model_weights_sha256", "")) == sha256(model_path)
        and str(summary.get("model_metadata_sha256", "")) == sha256(metadata_path)
        and expected_ids.issubset(set(results["comparison_id"]))
        and results["comparison_id"].is_unique
        and bool(summary.get("beta_distribution_qc_applied"))
    )


def run_one_gse(
    gse: str,
    branch: str,
    args: argparse.Namespace,
    expected_ids: set[str],
    model_path: Path,
) -> dict[str, Any]:
    config = BRANCH_CONFIG[branch]
    shard_dir = args.output_dir / "shards" / branch / gse
    if not args.force and completed_shard_is_valid(shard_dir, expected_ids, model_path):
        return {"gse_id": gse, "branch": branch, "status": "cached"}
    shard_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "intervention_analysis.run_geo_biometaclock_gse",
        "--sample-manifest",
        relative_path(config["sample_manifest"]),
        "--matrix-manifest",
        relative_path(config["matrix_manifest"]),
        "--gse",
        gse,
        "--matrix-root",
        relative_path(args.matrix_root),
        "--output-dir",
        relative_path(shard_dir),
        "--model-dir",
        relative_path(args.model_dir),
        "--pyaging-dir",
        relative_path(args.pyaging_dir),
        "--pyaging-batch-size",
        str(args.pyaging_batch_size),
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    (shard_dir / "runner.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (shard_dir / "runner.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return {
            "gse_id": gse,
            "branch": branch,
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
        }
    if not completed_shard_is_valid(shard_dir, expected_ids, model_path):
        return {
            "gse_id": gse,
            "branch": branch,
            "status": "invalid_completed_shard",
            "command": command,
        }
    return {"gse_id": gse, "branch": branch, "status": "completed", "command": command}


def finalize(
    args: argparse.Namespace,
    reference: pd.DataFrame,
    acquisition: pd.DataFrame,
    model_path: Path,
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    context_result_parts = []
    for row in acquisition.itertuples(index=False):
        path = (
            args.output_dir
            / "shards"
            / str(row.analysis_branch)
            / str(row.gse_id)
            / "comparison_results.csv"
        )
        part = pd.read_csv(path)
        part["result_source"] = BRANCH_CONFIG[str(row.analysis_branch)]["result_source"]
        context_result_parts.append(part)
    context_results = pd.concat(context_result_parts, ignore_index=True)
    if not context_results["comparison_id"].is_unique:
        raise ValueError("Full GSE-context comparison IDs are not unique")
    context_results.to_csv(
        args.output_dir / "all_gse_context_comparison_results.csv", index=False
    )
    results = context_results.loc[
        context_results["comparison_id"].isin(set(reference["comparison_id"]))
    ].copy()
    if len(results) != 77 or results["gse_id"].nunique() != 64:
        raise ValueError("Fresh results are not 77 comparisons / 64 GSE")
    if set(results["comparison_id"]) != set(reference["comparison_id"]):
        raise ValueError("Fresh comparison IDs differ from the frozen 77")
    if not results["comparison_id"].is_unique:
        raise ValueError("Fresh comparison IDs are not unique")

    p_values = pd.to_numeric(results["two_sided_p_value"], errors="raise")
    q_standard = multipletests(p_values.to_numpy(), method="fdr_bh")[1]
    q_pipeline = scipy.stats.false_discovery_control(p_values.to_numpy(), method="bh")
    max_bh_error = float(np.max(np.abs(q_standard - q_pipeline)))
    if max_bh_error > 1e-12:
        raise ValueError(f"BH implementation mismatch: {max_bh_error}")
    results["bh_q_value_after_dataset_exclusion"] = q_standard
    results = results.sort_values(
        ["two_sided_p_value", "mean_diff_experiment_minus_control"],
        ascending=[True, False],
    ).reset_index(drop=True)
    results.insert(0, "rank_after_dataset_exclusion", range(1, len(results) + 1))

    comparison = results[
        [
            "comparison_id",
            "gse_id",
            "mean_diff_experiment_minus_control",
            "two_sided_p_value",
            "bh_q_value_after_dataset_exclusion",
        ]
    ].merge(
        reference[
            [
                "comparison_id",
                "mean_diff_experiment_minus_control",
                "two_sided_p_value",
                "bh_q_value_after_dataset_exclusion",
            ]
        ],
        on="comparison_id",
        how="left",
        validate="one_to_one",
        suffixes=("_fresh", "_frozen"),
    )
    for metric in (
        "mean_diff_experiment_minus_control",
        "two_sided_p_value",
        "bh_q_value_after_dataset_exclusion",
    ):
        comparison[f"{metric}_absolute_difference"] = (
            comparison[f"{metric}_fresh"] - comparison[f"{metric}_frozen"]
        ).abs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results_77_interventions.csv"
    results.to_csv(results_path, index=False)
    comparison.to_csv(
        args.output_dir / "frozen_reproduction_comparison.csv", index=False
    )

    volcano_stem = args.output_dir / "volcano" / "geo_biometaclock_volcano_fdr_5_20"
    plot_command = [
        sys.executable,
        "-m",
        "intervention_analysis.plot_geo_biometaclock_volcano_fdr",
        "--results",
        relative_path(results_path),
        "--selection-audit",
        relative_path(SELECTION_AUDIT),
        "--output-stem",
        relative_path(volcano_stem),
    ]
    plotted = subprocess.run(
        plot_command,
        text=True,
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    (args.output_dir / "volcano.stdout.log").write_text(
        plotted.stdout, encoding="utf-8"
    )
    (args.output_dir / "volcano.stderr.log").write_text(
        plotted.stderr, encoding="utf-8"
    )
    if plotted.returncode != 0:
        raise RuntimeError(
            "Volcano plotting failed; inspect "
            f"{relative_path(args.output_dir / 'volcano.stderr.log')}"
        )

    tests = results["test"].value_counts().to_dict()
    q_values = results["bh_q_value_after_dataset_exclusion"]
    return {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_interventions": len(results),
        "n_gse": int(results["gse_id"].nunique()),
        "n_gse_context_comparisons_computed": int(len(context_results)),
        "test_counts": {str(key): int(value) for key, value in tests.items()},
        "raw_p_lt_0_05": int(results["two_sided_p_value"].lt(0.05).sum()),
        "bh_q_le_0_05": int(q_values.le(0.05).sum()),
        "bh_q_le_0_20": int(q_values.le(0.20).sum()),
        "model_weights": relative_path(model_path),
        "model_metadata": relative_path(
            model_path.parent / "biometaclock_672_participants_metadata.json"
        ),
        "model_weights_sha256": sha256(model_path),
        "model_metadata_sha256": sha256(
            model_path.parent / "biometaclock_672_participants_metadata.json"
        ),
        "beta_distribution_qc": "strict_original_thresholds",
        "bh_method": "statsmodels.stats.multitest.multipletests(method='fdr_bh')",
        "bh_crosscheck": "scipy.stats.false_discovery_control(method='bh')",
        "bh_max_absolute_crosscheck_error": max_bh_error,
        "max_absolute_reproduction_differences": {
            metric: float(comparison[f"{metric}_absolute_difference"].max())
            for metric in (
                "mean_diff_experiment_minus_control",
                "two_sided_p_value",
                "bh_q_value_after_dataset_exclusion",
            )
        },
        "results": relative_path(results_path),
        "volcano_pdf": relative_path(volcano_stem.with_suffix(".pdf")),
        "volcano_svg": relative_path(volcano_stem.with_suffix(".svg")),
        "run_records": run_records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
            "anndata": importlib.metadata.version("anndata"),
            "pyaging": importlib.metadata.version("pyaging"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "torch": importlib.metadata.version("torch"),
        },
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.pyaging_batch_size < 1:
        raise ValueError("Worker count and batch size must be positive")
    reference, acquisition, model_path = validate_inputs(args)
    all_gses = list(acquisition["gse_id"])
    selected_gses = all_gses
    if args.only_gse:
        requested = set(args.only_gse)
        unknown = sorted(requested - set(all_gses))
        if unknown:
            raise ValueError(f"Unknown frozen GSE accessions: {unknown}")
        selected_gses = [gse for gse in all_gses if gse in requested]
    plan = {
        "n_frozen_interventions": 77,
        "n_frozen_gse": 64,
        "n_selected_gse": len(selected_gses),
        "matrix_root": relative_path(args.matrix_root),
        "output_dir": relative_path(args.output_dir),
        "model_weights": relative_path(model_path),
        "model_metadata": relative_path(
            args.model_dir / "biometaclock_672_participants_metadata.json"
        ),
        "model_weights_sha256": sha256(model_path),
        "model_metadata_sha256": sha256(
            args.model_dir / "biometaclock_672_participants_metadata.json"
        ),
        "pyaging_dir": relative_path(args.pyaging_dir),
        "matplotlib_cache": relative_path(MPLCONFIGDIR),
        "workers": args.workers,
    }
    ids_by_gse = expected_ids_by_gse(reference)
    branch_by_gse = dict(zip(acquisition["gse_id"], acquisition["analysis_branch"]))
    run_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one_gse,
                gse,
                str(branch_by_gse[gse]),
                args,
                ids_by_gse[gse],
                model_path,
            ): gse
            for gse in selected_gses
        }
        for future in as_completed(futures):
            run_records.append(future.result())
    run_records.sort(key=lambda row: int(str(row["gse_id"])[3:]))
    failures = [
        row for row in run_records if row["status"] not in {"completed", "cached"}
    ]
    if failures:
        summary = {
            **plan,
            "status": "failed",
            "failures": failures,
            "run_records": run_records,
        }
        write_json_atomic(summary, args.output_dir / "analysis_summary.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    if set(selected_gses) != set(all_gses):
        summary = {
            **plan,
            "status": "subset_complete_no_bh_or_volcano",
            "run_records": run_records,
        }
        write_json_atomic(summary, args.output_dir / "analysis_summary.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    summary = finalize(args, reference, acquisition, model_path, run_records)
    write_json_atomic(summary, args.output_dir / "analysis_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
