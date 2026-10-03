#!/usr/bin/env python3
"""Predict GEO and TranslAGE interventions, deduplicate, and BH-adjust 94 tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
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
    DEFAULT_TRANSLAGE_MANIFEST,
    DEFAULT_TRANSLAGE_DATA_ROOT,
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
            "datasets and 19 TranslAGE analyses, retain one source per overlapping "
            "accession, recompute BH across 94 tests, and create the FDR volcano."
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
            "(default: results/intervention/biometaclock_94)."
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
    parser.add_argument(
        "--translage-manifest", type=repo_path, default=DEFAULT_TRANSLAGE_MANIFEST
    )
    parser.add_argument(
        "--translage-data-root", type=repo_path, default=DEFAULT_TRANSLAGE_DATA_ROOT
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--only-gse",
        nargs="+",
        help="Run a subset for debugging; final BH/volcano are produced only for all 64 GSE.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed GEO and TranslAGE prediction shards.",
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
    """Validate frozen manifests, selected matrices, and published model files."""
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
    """Map each GEO accession to its required frozen comparison identifiers."""
    return {
        str(gse): set(group["comparison_id"].astype(str))
        for gse, group in reference.groupby("gse_id", sort=False)
    }


def translage_shard(output_dir: Path, row: pd.Series) -> Path:
    """Locate the shard for one frozen TranslAGE analysis."""
    dataset_id = re.sub(
        r"[^A-Za-z0-9._-]+", "_", str(row["TranslAGE Dataset ID"])
    ).strip("_")
    return (
        output_dir
        / "shards"
        / "translage"
        / f"{int(row['analysis_index']):03d}__{dataset_id}"
    )


def validate_translage_inputs(args: argparse.Namespace) -> pd.DataFrame:
    """Require the screened 19-test manifest and its verified cleaned payloads."""
    manifest = pd.read_csv(args.translage_manifest)
    if len(manifest) != 19 or manifest["Public ID"].nunique() != 13:
        raise ValueError("TranslAGE manifest must contain 19 analyses / 13 accessions")
    if not manifest["TranslAGE Dataset ID"].is_unique or set(
        manifest["analysis_index"]
    ) != set(range(19)):
        raise ValueError(
            "TranslAGE analysis identifiers must be unique and indexed 0..18"
        )
    for _, row in manifest.iterrows():
        dataset_id = re.sub(
            r"[^A-Za-z0-9._-]+", "_", str(row["TranslAGE Dataset ID"])
        ).strip("_")
        path = repo_child(
            args.translage_data_root, dataset_id + "/" + str(row["file_name"])
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing TranslAGE matrix: {relative_path(path)}")
        if path.stat().st_size != int(row["expected_cleaned_bytes"]):
            raise ValueError(f"TranslAGE matrix size mismatch: {relative_path(path)}")
        if sha256(path) != str(row["cleaned_sha256"]):
            raise ValueError(f"TranslAGE matrix SHA256 mismatch: {relative_path(path)}")
    return manifest.sort_values("analysis_index").reset_index(drop=True)


def run_one_translage(
    row: pd.Series, args: argparse.Namespace, model_path: Path
) -> dict[str, Any]:
    """Run a curated TranslAGE design with the same model and component clocks."""
    dataset_id = str(row["TranslAGE Dataset ID"])
    shard_dir = translage_shard(args.output_dir, row)
    expected = {str(row["concept_id"])}
    if (
        not args.force
        and (shard_dir / "analysis_sample_predictions.csv").is_file()
        and completed_shard_is_valid(shard_dir, expected, model_path)
    ):
        summary = json.loads((shard_dir / "run_summary.json").read_text())
        if summary["manifest_sha256"] == sha256(args.translage_manifest):
            return {"translage_dataset_id": dataset_id, "status": "cached"}
    shard_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "intervention_analysis.run_translage_biometaclock",
        "--manifest",
        relative_path(args.translage_manifest),
        "--task-index",
        str(int(row["analysis_index"])),
        "--data-root",
        relative_path(args.translage_data_root),
        "--output-root",
        relative_path(args.output_dir / "shards" / "translage"),
        "--model-dir",
        relative_path(args.model_dir),
        "--pyaging-dir",
        relative_path(args.pyaging_dir),
        "--pyaging-batch-size",
        str(args.pyaging_batch_size),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=single_thread_environment(),
    )
    (shard_dir / "runner.stdout.log").write_text(completed.stdout)
    (shard_dir / "runner.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        return {
            "translage_dataset_id": dataset_id,
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
        }
    if not completed_shard_is_valid(shard_dir, expected, model_path):
        return {
            "translage_dataset_id": dataset_id,
            "status": "invalid_completed_shard",
            "command": command,
        }
    return {
        "translage_dataset_id": dataset_id,
        "status": "completed",
        "command": command,
    }


def single_thread_environment() -> dict[str, str]:
    """Keep worker subprocesses within the requested process-level parallelism."""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def bh_adjust(p_values: pd.Series) -> tuple[np.ndarray, float]:
    """Apply statsmodels BH and independently cross-check with SciPy."""
    values = pd.to_numeric(p_values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("All raw p-values must be finite and in [0, 1]")
    adjusted = multipletests(values, method="fdr_bh")[1]
    check = scipy.stats.false_discovery_control(values, method="bh")
    error = float(np.max(np.abs(adjusted - check)))
    if error > 1e-12:
        raise ValueError(f"BH implementation mismatch: {error}")
    return adjusted, error


def combine_results(
    geo: pd.DataFrame, translage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Retain TranslAGE for the two shared accessions, then adjust all 94 tests."""
    if len(geo) != 77 or geo["gse_id"].nunique() != 64:
        raise ValueError("Expected 77 GEO tests from 64 accessions")
    if len(translage) != 19 or translage["gse_id"].nunique() != 13:
        raise ValueError("Expected 19 TranslAGE tests from 13 accessions")
    geo = geo.copy()
    translage = translage.copy()
    geo["family_test_id"] = "geo::" + geo["comparison_id"].astype(str)
    geo["source_dataset_id"] = geo["gse_id"]
    geo["source"] = "previous_geo"
    geo["intervention"] = geo["comparison_name"]
    geo["original_source_bh_q_value"] = geo["bh_q_value_after_dataset_exclusion"]
    translage["family_test_id"] = "translage::" + translage[
        "translage_dataset_id"
    ].astype(str)
    translage["source_dataset_id"] = translage["translage_dataset_id"]
    translage["source"] = "translage_current"
    translage["original_source_bh_q_value"], _ = bh_adjust(
        translage["two_sided_p_value"]
    )
    overlap = set(geo["gse_id"]) & set(translage["gse_id"])
    if overlap != {"GSE201532", "GSE74548"}:
        raise ValueError(f"Unexpected cross-source accessions: {sorted(overlap)}")
    audit = pd.DataFrame(
        [
            {
                "public_id": accession,
                "excluded_previous_geo_test_ids": ";".join(
                    geo.loc[geo["gse_id"].eq(accession), "family_test_id"]
                ),
                "retained_translage_test_ids": ";".join(
                    translage.loc[translage["gse_id"].eq(accession), "family_test_id"]
                ),
                "resolution": "retain_current_translage_audited_design",
            }
            for accession in sorted(overlap)
        ]
    )
    combined = pd.concat(
        [geo.loc[~geo["gse_id"].isin(overlap)], translage], ignore_index=True
    )
    if len(combined) != 94 or not combined["family_test_id"].is_unique:
        raise ValueError("Combined family must contain 94 unique tests")
    combined["public_id"] = combined["gse_id"]
    # Source-family q-values are provenance only; inference uses the combined family.
    combined = combined.drop(
        columns=["bh_q_value_after_dataset_exclusion", "rank_after_dataset_exclusion"]
    )
    combined["combined_bh_q_value"], error = bh_adjust(combined["two_sided_p_value"])
    combined["combined_significance_class"] = np.select(
        [
            combined["combined_bh_q_value"].le(0.05),
            combined["combined_bh_q_value"].le(0.20),
        ],
        ["strict_fdr_q_le_0.05", "exploratory_q_le_0.20"],
        default="not_significant",
    )
    combined = combined.sort_values(
        ["combined_bh_q_value", "two_sided_p_value", "family_test_id"]
    ).reset_index(drop=True)
    combined.insert(0, "combined_rank", range(1, len(combined) + 1))
    return combined, audit, error


def audit_sample_overlap(
    geo_results: pd.DataFrame, translage_manifest: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Check selected sample IDs across all 77-by-19 source comparisons."""
    geo_samples = pd.concat(
        [pd.read_csv(config["sample_manifest"]) for config in BRANCH_CONFIG.values()]
    )
    geo_samples = geo_samples.loc[
        geo_samples["comparison_id"].isin(geo_results["comparison_id"])
    ]
    records = []
    for _, row in translage_manifest.iterrows():
        samples = pd.read_csv(
            translage_shard(output_dir, row) / "analysis_sample_predictions.csv"
        )
        sample_ids = set(samples["sample_id"].astype(str))
        for (accession, comparison_id), group in geo_samples.groupby(
            ["gse_id", "comparison_id"]
        ):
            old_ids = set(group["sample_id"].astype(str))
            shared = sorted(sample_ids & old_ids)
            if shared:
                records.append(
                    {
                        "translage_dataset_id": str(row["TranslAGE Dataset ID"]),
                        "geo_gse_id": str(accession),
                        "geo_comparison_id": str(comparison_id),
                        "n_shared_sample_ids": len(shared),
                        "n_translage_selected_sample_ids": len(sample_ids),
                        "n_geo_selected_sample_ids": len(old_ids),
                        "all_geo_samples_contained_in_translage": old_ids.issubset(
                            sample_ids
                        ),
                        "shared_sample_ids": ";".join(shared),
                    }
                )
    audit = pd.DataFrame(records)
    observed = {(row["translage_dataset_id"], row["geo_gse_id"]) for row in records}
    if observed != {("GSE201532", "GSE201532"), ("GSE74548", "GSE74548")}:
        raise ValueError(f"Unexpected exact sample-ID overlaps: {sorted(observed)}")
    if not audit["all_geo_samples_contained_in_translage"].all():
        raise ValueError("The known GEO sample sets must be contained in TranslAGE")
    return audit


def completed_shard_is_valid(
    shard_dir: Path, expected_ids: set[str], model_path: Path
) -> bool:
    """Reuse matching caches; malformed files or missing required metadata raise."""
    summary_path = shard_dir / "run_summary.json"
    results_path = shard_dir / "comparison_results.csv"
    if not summary_path.is_file() or not results_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    results = pd.read_csv(results_path, dtype=str)
    metadata_path = model_path.parent / "biometaclock_672_participants_metadata.json"
    return (
        str(summary["model_weights"]) == relative_path(model_path)
        and str(summary["model_metadata"]) == relative_path(metadata_path)
        and str(summary["model_weights_sha256"]) == sha256(model_path)
        and str(summary["model_metadata_sha256"]) == sha256(metadata_path)
        and expected_ids.issubset(set(results["comparison_id"]))
        and results["comparison_id"].is_unique
        and bool(summary["beta_distribution_qc_applied"])
    )


def run_one_gse(
    gse: str,
    branch: str,
    args: argparse.Namespace,
    expected_ids: set[str],
    model_path: Path,
) -> dict[str, Any]:
    """Run or reuse one GEO shard and return its orchestration status record."""
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
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=single_thread_environment(),
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
    translage_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Audit source results, remove overlap, and write the 94-test BH family."""
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

    results["bh_q_value_after_dataset_exclusion"], _ = bh_adjust(
        results["two_sided_p_value"]
    )
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
    geo_results_path = args.output_dir / "results_77_interventions.csv"
    results.to_csv(geo_results_path, index=False)
    comparison.to_csv(
        args.output_dir / "frozen_reproduction_comparison.csv", index=False
    )

    translage = pd.concat(
        [
            pd.read_csv(
                translage_shard(args.output_dir, row) / "comparison_results.csv"
            )
            for _, row in translage_manifest.iterrows()
        ],
        ignore_index=True,
    )
    if set(translage["translage_dataset_id"]) != set(
        translage_manifest["TranslAGE Dataset ID"]
    ):
        raise ValueError(
            "TranslAGE result membership differs from the screened manifest"
        )
    translage.to_csv(
        args.output_dir / "results_19_translage_interventions.csv", index=False
    )
    sample_overlap = audit_sample_overlap(results, translage_manifest, args.output_dir)
    sample_overlap.to_csv(
        args.output_dir / "cross_source_sample_overlap_audit.csv", index=False
    )
    combined, overlap, max_bh_error = combine_results(results, translage)
    results_path = args.output_dir / "combined_geo_translage_results_bh.csv"
    combined.to_csv(results_path, index=False)
    overlap.to_csv(args.output_dir / "overlap_resolution_audit.csv", index=False)
    combined.loc[combined["combined_bh_q_value"].le(0.05)].to_csv(
        args.output_dir / "combined_geo_translage_strict_significant.csv", index=False
    )
    combined.loc[
        combined["combined_bh_q_value"].gt(0.05)
        & combined["combined_bh_q_value"].le(0.20)
    ].to_csv(args.output_dir / "combined_geo_translage_exploratory.csv", index=False)
    volcano_stem = (
        args.output_dir
        / "volcano"
        / "combined_geo_translage_biometaclock_volcano_fdr_5_20"
    )
    plot_command = [
        sys.executable,
        "-m",
        "intervention_analysis.plot_geo_biometaclock_volcano_fdr",
        "--results",
        relative_path(results_path),
        "--combined-geo-translage",
        "--axis-reference-results",
        relative_path(geo_results_path),
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
        env=single_thread_environment(),
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

    tests = combined["test"].value_counts().to_dict()
    q_values = combined["combined_bh_q_value"]
    return {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_interventions": len(combined),
        "n_public_accessions": int(combined["public_id"].nunique()),
        "n_geo_input_tests": len(results),
        "n_translage_input_tests": len(translage),
        "n_previous_geo_tests_excluded_as_overlap": len(results)
        + len(translage)
        - len(combined),
        "overlapping_public_accessions": overlap["public_id"].tolist(),
        "n_cross_source_test_pairs_checked": len(results) * len(translage),
        "n_gse_context_comparisons_computed": int(len(context_results)),
        "test_counts": {str(key): int(value) for key, value in tests.items()},
        "raw_p_lt_0_05": int(combined["two_sided_p_value"].lt(0.05).sum()),
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
        "translage_manifest": relative_path(args.translage_manifest),
        "translage_manifest_sha256": sha256(args.translage_manifest),
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
    translage_manifest = (
        validate_translage_inputs(args) if set(selected_gses) == set(all_gses) else None
    )
    plan = {
        "n_frozen_interventions": 94,
        "n_geo_interventions": 77,
        "n_translage_interventions": 19,
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

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one_translage, row, args, model_path)
            for _, row in translage_manifest.iterrows()
        ]
        for future in as_completed(futures):
            run_records.append(future.result())
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
    summary = finalize(
        args, reference, acquisition, model_path, run_records, translage_manifest
    )
    write_json_atomic(summary, args.output_dir / "analysis_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
