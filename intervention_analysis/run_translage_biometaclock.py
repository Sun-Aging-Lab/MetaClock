#!/usr/bin/env python3
"""Predict and test one curated TranslAGE comparison with published BioMetaClock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biological_age_prediction_and_mortality_analysis.published_biometaclock import (
    load_published_biometaclock,
)
from intervention_analysis.paths import relative_path, repo_child, repo_path
from intervention_analysis.run_geo_biometaclock_gse import (
    MAX_NEAR_ZERO_BETA_FRACTION,
    MIN_BETA_MEDIAN,
    MIN_FINITE_BETA_FRACTION,
    MIN_IN_RANGE_BETA_FRACTION,
    age_correct_strata,
    sha256,
    summarize_comparisons,
    write_csv_atomic,
    write_json_atomic,
)
from intervention_analysis.runtime import (
    MIN_CPG_COVERAGE,
    load_biometa_clock_features,
    run_pyaging_clocks,
)


def parse_args() -> argparse.Namespace:
    """Parse one manifest task; paths are relative to the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=repo_path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--data-root", type=repo_path, required=True)
    parser.add_argument("--output-root", type=repo_path, required=True)
    parser.add_argument("--model-dir", type=repo_path, required=True)
    parser.add_argument("--pyaging-dir", type=repo_path, required=True)
    parser.add_argument("--pyaging-batch-size", type=int, default=512)
    return parser.parse_args()


def safe_id(value: str) -> str:
    """Match the directory identifier used by the TranslAGE downloader."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def dataframe_payload(obj: Any, *, label: str) -> pd.DataFrame:
    """Adapt R data.frame columns, including single-column matrix covariates."""
    if not isinstance(obj, dict) or "data" not in obj or "attrs" not in obj:
        raise TypeError(f"{label}: expected custom-converted R data.frame")
    columns: dict[str, Any] = {}
    lengths: set[int] = set()
    for key, value in obj["data"].items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"{label}: unsupported shape {array.shape} for column {key}")
        columns[str(key)] = array
        lengths.add(len(array))
    if len(lengths) != 1:
        raise ValueError(
            f"{label}: inconsistent one-dimensional column lengths {lengths}"
        )
    index = np.asarray(obj["attrs"].get("row.names"), dtype=str)
    frame = pd.DataFrame(columns)
    if len(index) == len(frame):
        frame.index = index
    return frame


def load_cleaned(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read official RData and average consistent technical replicate beta rows."""
    import rdata
    from rdata.conversion import DEFAULT_CLASS_MAP

    constructors = dict(DEFAULT_CLASS_MAP)
    constructors["data.frame"] = lambda obj, attrs: {"data": obj, "attrs": attrs}
    objects = rdata.read_rda(path, constructor_dict=constructors)
    array_matrices = [
        (name, value)
        for name, value in objects.items()
        if getattr(value, "ndim", None) == 2
        and max(int(item) for item in value.shape) > 10000
    ]
    dataframe_matrices = [
        (name, value)
        for name, value in objects.items()
        if isinstance(value, dict) and "data" in value and len(value["data"]) > 10000
    ]
    matrices = [*array_matrices, *dataframe_matrices]
    phenotypes = [
        (name, value)
        for name, value in objects.items()
        if isinstance(value, dict)
        and "data" in value
        and {"cSAMPLEID", "cSUBJECTID", "cGROUP", "cTIME"}.issubset(
            {str(item) for item in value["data"]}
        )
    ]
    if len(matrices) != 1 or len(phenotypes) != 1:
        raise ValueError(
            f"Expected one methylation matrix and one phenotype frame; "
            f"found matrices={[x[0] for x in matrices]}, phenotypes={[x[0] for x in phenotypes]}"
        )
    matrix_name, matrix_object = matrices[0]
    pheno_name, pheno_obj = phenotypes[0]
    required = {"cSAMPLEID", "cSUBJECTID", "cGROUP", "cTIME"}
    # Only curated design fields enter the analysis; unrelated R metadata may
    # contain nested objects that are not tabular covariates.
    phenotype = dataframe_payload(
        {
            "data": {
                key: value
                for key, value in pheno_obj["data"].items()
                if str(key) in required | {"cAGE"}
            },
            "attrs": pheno_obj["attrs"],
        },
        label=pheno_name,
    )
    missing = sorted(required.difference(phenotype.columns))
    if missing:
        raise KeyError(f"Phenotype is missing curated TranslAGE fields: {missing}")
    sample_ids = phenotype["cSAMPLEID"].astype(str).to_numpy()
    if isinstance(matrix_object, dict):
        methylation = dataframe_payload(matrix_object, label=str(matrix_name))
        methylation.index = methylation.index.astype(str)
        if set(methylation.index) != set(sample_ids):
            raise ValueError(
                "R data.frame methylation row names do not match cSAMPLEID"
            )
        matrix_samples = methylation.index.to_numpy(dtype=str)
    else:
        matrix = matrix_object
        dimensions = list(matrix.dims)
        coordinates = {
            dimension: np.asarray(matrix.coords[dimension].values, dtype=str)
            for dimension in dimensions
        }
        sample_axes = [
            dimension
            for dimension, values in coordinates.items()
            if len(values) == len(sample_ids) and set(values) == set(sample_ids)
        ]
        probe_axes = [
            dimension
            for dimension, values in coordinates.items()
            if np.mean(np.char.startswith(values.astype(str), "cg")) > 0.5
        ]
        if len(sample_axes) != 1 or len(probe_axes) != 1:
            raise ValueError(
                "Could not identify unique sample and CpG axes in R matrix"
            )
        sample_axis = sample_axes[0]
        probe_axis = probe_axes[0]
        if list(matrix.dims) != [sample_axis, probe_axis]:
            matrix = matrix.transpose(sample_axis, probe_axis)
        matrix_samples = np.asarray(matrix.coords[sample_axis].values, dtype=str)
        probes = np.asarray(matrix.coords[probe_axis].values, dtype=str)
        methylation = pd.DataFrame(
            np.asarray(matrix.values), index=matrix_samples, columns=probes
        )
    source_matrix_shape = [int(item) for item in methylation.shape]
    # GSE240184 repeats curated sample IDs for technical replicates. Require
    # identical design metadata before averaging their beta measurements.
    phenotype["cSAMPLEID"] = phenotype["cSAMPLEID"].astype(str)
    duplicate_sample_mask = phenotype["cSAMPLEID"].duplicated(keep=False)
    duplicate_sample_ids = sorted(
        phenotype.loc[duplicate_sample_mask, "cSAMPLEID"].unique().tolist()
    )
    if duplicate_sample_ids:
        consistency_columns = [
            item
            for item in ["cSAMPLEID", "cSUBJECTID", "cGROUP", "cTIME", "cAGE"]
            if item in phenotype.columns
        ]
        consistency = (
            phenotype.loc[duplicate_sample_mask, consistency_columns]
            .groupby("cSAMPLEID", dropna=False)
            .nunique(dropna=False)
        )
        if consistency.gt(1).any().any():
            raise ValueError(
                "Duplicate cSAMPLEID rows have conflicting curated analysis metadata"
            )
        phenotype = phenotype.drop_duplicates("cSAMPLEID", keep="first").copy()
        methylation = methylation.groupby(level=0, sort=False).mean()
        matrix_samples = methylation.index.to_numpy(dtype=str)
    phenotype = phenotype.set_index(phenotype["cSAMPLEID"], drop=False)
    phenotype = phenotype.loc[matrix_samples].copy()
    metadata = {
        "r_objects": sorted(str(item) for item in objects),
        "matrix_object": str(matrix_name),
        "phenotype_object": str(pheno_name),
        "matrix_shape_full": source_matrix_shape,
        "matrix_shape_after_technical_replicate_collapse": [
            int(item) for item in methylation.shape
        ],
        "n_technical_replicate_sample_ids_collapsed": len(duplicate_sample_ids),
        "technical_replicate_sample_ids_collapsed": duplicate_sample_ids,
    }
    return methylation, phenotype, metadata


def prepare_design(
    phenotype: pd.DataFrame,
    row: pd.Series,
) -> tuple[pd.DataFrame, str, str, float, float]:
    """Select complete donor pairs at the baseline and prespecified endpoint."""
    design = phenotype[["cSAMPLEID", "cSUBJECTID", "cGROUP", "cTIME"]].copy()
    if design[["cSAMPLEID", "cSUBJECTID"]].isna().any().any():
        raise ValueError("Curated sample and donor fields must be complete")
    design["cSAMPLEID"] = design["cSAMPLEID"].astype(str)
    design["cSUBJECTID"] = design["cSUBJECTID"].astype(str)
    design["cGROUP"] = design["cGROUP"].astype(str).str.strip()
    design["cTIME"] = pd.to_numeric(design["cTIME"], errors="coerce")
    # GSE217633 includes reference samples without an intervention time.
    # Exclude them before selecting paired donors or predicting clock age.
    design = design.loc[design["cTIME"].notna()].copy()
    if design.empty:
        raise ValueError("No samples have a curated intervention time")
    if design["cSAMPLEID"].duplicated().any():
        raise ValueError("cSAMPLEID must be unique")
    override = pd.to_numeric(
        pd.Series([row["target_time_days"]]), errors="raise"
    ).iloc[0]
    donor_keys = ["cGROUP", "cSUBJECTID"]
    bounds = design.groupby(donor_keys, as_index=False)["cTIME"].agg(
        donor_baseline_time="min", donor_latest_time="max"
    )
    design = design.merge(bounds, on=donor_keys, how="left", validate="many_to_one")
    # GSE191297 uses the manifest's 42-day endpoint. Otherwise retain each
    # donor's latest visit; GSE210245 donors have different follow-up times.
    if pd.notna(override):
        target_time = float(override)
        is_endpoint = np.isclose(design["cTIME"], target_time)
    else:
        is_endpoint = np.isclose(design["cTIME"], design["donor_latest_time"])
        target_time = float(design.loc[is_endpoint, "cTIME"].max())
    is_baseline = np.isclose(design["cTIME"], design["donor_baseline_time"])
    selected = design.loc[is_baseline | is_endpoint].copy()
    selected["timepoint"] = np.where(
        np.isclose(selected["cTIME"], selected["donor_baseline_time"]),
        "pre",
        "post",
    )
    selected = selected.loc[
        selected["donor_latest_time"].gt(selected["donor_baseline_time"])
    ].copy()
    baseline_time = float(selected["donor_baseline_time"].min())
    if not target_time > baseline_time:
        raise ValueError(
            f"Endpoint {target_time} must be after baseline {baseline_time}"
        )
    selected["chronological_age"] = pd.to_numeric(
        phenotype.loc[selected["cSAMPLEID"], "cAGE"].to_numpy()
        if "cAGE" in phenotype
        else np.nan,
        errors="coerce",
    )
    group_values = {value.lower(): value for value in selected["cGROUP"].unique()}
    has_parallel_groups = {"control", "subject"}.issubset(group_values)
    concept_id = str(row["concept_id"])
    if has_parallel_groups:
        selected = selected.loc[
            selected["cGROUP"].str.lower().isin({"control", "subject"})
        ].copy()
        selected["group"] = np.where(
            selected["cGROUP"].str.lower().eq("control"), "control", "experiment"
        )
        complete_donors = (
            selected.groupby(["group", "cSUBJECTID"])["timepoint"]
            .nunique()
            .loc[lambda x: x.eq(2)]
            .reset_index()[["group", "cSUBJECTID"]]
        )
        selected = selected.merge(
            complete_donors,
            on=["group", "cSUBJECTID"],
            how="inner",
            validate="many_to_one",
        )
        counts = selected.groupby("group")["cSUBJECTID"].nunique()
        if (
            not {"control", "experiment"}.issubset(counts.index)
            or int(counts.min()) < 4
        ):
            raise ValueError(
                f"Parallel change design has fewer than 4 complete donors per arm: {counts.to_dict()}"
            )
        analysis_method = "donor_pre_post_change_welch"
        estimand = "between_arm_endpoint_minus_baseline_change"
    else:
        donor_time_counts = selected.groupby("cSUBJECTID")["timepoint"].nunique()
        paired = set(donor_time_counts.loc[donor_time_counts.eq(2)].index.astype(str))
        selected = selected.loc[selected["cSUBJECTID"].isin(paired)].copy()
        if len(paired) < 4:
            raise ValueError(
                f"Paired baseline/endpoint design has only {len(paired)} donors"
            )
        selected["group"] = np.where(
            selected["timepoint"].eq("pre"), "control", "experiment"
        )
        analysis_method = (
            "donor_level_paired"
            if selected["chronological_age"].notna().all()
            else "donor_level_paired_no_age_correction"
        )
        estimand = "paired_endpoint_minus_baseline"
    selected = selected.rename(
        columns={"cSAMPLEID": "sample_id", "cSUBJECTID": "donor_id"}
    )
    selected["comparison_id"] = concept_id
    selected["gse_id"] = str(row["Public ID"])
    selected["comparison_name"] = str(row["Name"])
    selected["analysis_method"] = analysis_method
    selected["age_correction_stratum"] = concept_id
    selected["canonical_comparison_id"] = concept_id
    signature = hashlib.sha256(
        "\n".join(
            selected.sort_values(["sample_id", "group", "timepoint"])[
                ["sample_id", "donor_id", "group", "timepoint"]
            ]
            .astype(str)
            .agg("|".join, axis=1)
        ).encode()
    ).hexdigest()
    selected["sample_signature_id"] = signature
    selected["estimand"] = estimand
    return selected, analysis_method, estimand, baseline_time, target_time


def main() -> int:
    """Run beta QC, component clocks, donor tests, and reproducibility outputs."""
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    selected_row = manifest.loc[manifest["analysis_index"].eq(args.task_index)]
    if len(selected_row) != 1:
        raise ValueError(f"Expected one manifest row for task {args.task_index}")
    row = selected_row.iloc[0]
    dataset_id = str(row["TranslAGE Dataset ID"])
    output_dir = args.output_root / f"{args.task_index:03d}__{safe_id(dataset_id)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = repo_child(
        args.data_root / safe_id(dataset_id), str(row["file_name"])
    )
    if not cleaned_path.exists():
        raise FileNotFoundError(cleaned_path)

    methylation, phenotype, rdata_meta = load_cleaned(cleaned_path)
    n_missing_time_samples_excluded = int(
        pd.to_numeric(phenotype["cTIME"], errors="coerce").isna().sum()
    )
    design, analysis_method, estimand, baseline_time, target_time = prepare_design(
        phenotype, row
    )
    sample_ids = design["sample_id"].drop_duplicates().astype(str).tolist()
    methylation = methylation.loc[sample_ids]

    model, clock_names = load_published_biometaclock(args.model_dir)
    required_features = load_biometa_clock_features(clock_names, args.pyaging_dir)
    required_cpgs = [
        item
        for item in required_features
        if str(item).lower().startswith(("cg", "ch."))
    ]
    observed = [item for item in required_cpgs if item in methylation.columns]
    coverage = len(observed) / len(required_cpgs)
    if coverage < MIN_CPG_COVERAGE:
        raise ValueError(
            f"Required-CpG coverage {coverage:.4f} below {MIN_CPG_COVERAGE:.4f}"
        )
    methylation = methylation.loc[:, observed].astype(np.float32)
    values = methylation.to_numpy(dtype=float)
    finite = np.isfinite(values)
    finite_fraction = float(np.mean(finite))
    minimum_sample_finite_fraction = float(np.min(np.mean(finite, axis=1)))
    finite_values = values[finite]
    beta_min = float(np.min(finite_values)) if len(finite_values) else float("nan")
    beta_max = float(np.max(finite_values)) if len(finite_values) else float("nan")
    in_range_fraction = float(np.mean((finite_values >= 0) & (finite_values <= 1)))
    beta_median = float(np.median(finite_values))
    near_zero_fraction = float(np.mean(finite_values <= 0.01))
    if finite_fraction < MIN_FINITE_BETA_FRACTION:
        raise ValueError(f"Finite beta fraction {finite_fraction:.4f} below threshold")
    if minimum_sample_finite_fraction < MIN_FINITE_BETA_FRACTION:
        raise ValueError(
            f"Minimum sample finite beta fraction {minimum_sample_finite_fraction:.4f} below threshold"
        )
    if beta_min < 0 or beta_max > 1:
        raise ValueError(f"Finite beta range {beta_min}..{beta_max} is outside [0,1]")
    if in_range_fraction < MIN_IN_RANGE_BETA_FRACTION:
        raise ValueError(
            f"In-range beta fraction {in_range_fraction:.4f} below threshold"
        )
    if (
        beta_median < MIN_BETA_MEDIAN
        or near_zero_fraction > MAX_NEAR_ZERO_BETA_FRACTION
    ):
        raise ValueError(
            "Matrix resembles a detection-p matrix rather than beta values"
        )

    age_complete = design["chronological_age"].notna().all()
    clock_input = methylation.copy()
    if age_complete:
        sample_age = (
            design[["sample_id", "chronological_age"]]
            .drop_duplicates()
            .set_index("sample_id")
            .loc[methylation.index, "chronological_age"]
        )
        clock_input["age"] = sample_age.to_numpy(dtype=float)
    clock_outputs = run_pyaging_clocks(
        frame=clock_input,
        sample_id=pd.Series(methylation.index, dtype=str),
        clocks=clock_names,
        pyaging_dir=args.pyaging_dir,
        batch_size=args.pyaging_batch_size,
        required_features=required_features,
    ).set_index("sample_id")
    inputs = clock_outputs.loc[methylation.index, clock_names].replace(
        [np.inf, -np.inf], np.nan
    )
    # Preserve the frozen adapter convention for missing component predictions.
    model_matrix = inputs.fillna(0.0).to_numpy(dtype=np.float64)
    predictions = pd.DataFrame(
        {
            "sample_id": methylation.index,
            "predicted_bioage": np.asarray(model.predict(model_matrix), dtype=float),
        }
    )
    analysis_samples = design.merge(
        predictions, on="sample_id", how="left", validate="many_to_one"
    )
    if age_complete:
        analysis_samples, age_fits = age_correct_strata(analysis_samples)
        analysis_samples["analysis_predicted_age"] = analysis_samples[
            "age_corrected_predicted_age"
        ]
    else:
        analysis_samples["analysis_predicted_age"] = analysis_samples[
            "predicted_bioage"
        ]
        age_fits = pd.DataFrame(
            [
                {
                    "gse_id": str(row["Public ID"]),
                    "age_correction_stratum": dataset_id,
                    "age_adjustment": "none_missing_chronological_age",
                    "age_fit_n_unique_samples": 0,
                    "age_fit_n_unique_ages": 0,
                    "age_reference_age": np.nan,
                    "age_fit_slope": np.nan,
                    "age_fit_intercept": np.nan,
                    "age_fit_r2": np.nan,
                }
            ]
        )
    results, donor = summarize_comparisons(analysis_samples)
    if not age_complete and analysis_method == "donor_pre_post_change_welch":
        results["age_adjustment"] = "none_missing_chronological_age"
        results["control_mean_age_corrected_predicted_age"] = np.nan
        results["experiment_mean_age_corrected_predicted_age"] = np.nan
    results.insert(0, "translage_dataset_id", dataset_id)
    results["intervention"] = str(row["Name"])
    results["estimand"] = estimand
    results["baseline_time_days"] = baseline_time
    results["endpoint_time_days"] = target_time

    group_summary = analysis_samples.groupby(
        ["group", "timepoint"], as_index=False
    ).agg(
        n_samples=("sample_id", "nunique"),
        n_donors=("donor_id", "nunique"),
        mean_predicted_bioage=("predicted_bioage", "mean"),
        sd_predicted_bioage=("predicted_bioage", "std"),
        mean_analysis_predicted_age=("analysis_predicted_age", "mean"),
    )
    group_summary.insert(0, "translage_dataset_id", dataset_id)
    qc = pd.DataFrame(
        [
            {
                "translage_dataset_id": dataset_id,
                "n_selected_samples": len(methylation),
                "n_full_matrix_samples": rdata_meta["matrix_shape_full"][0],
                "n_full_matrix_cpgs": rdata_meta["matrix_shape_full"][1],
                "n_required_cpgs": len(required_cpgs),
                "n_observed_required_cpgs": len(observed),
                "required_cpg_coverage": coverage,
                "finite_beta_fraction": finite_fraction,
                "minimum_sample_finite_beta_fraction": minimum_sample_finite_fraction,
                "in_range_beta_fraction": in_range_fraction,
                "finite_beta_min": beta_min,
                "finite_beta_max": beta_max,
                "finite_beta_median": beta_median,
                "near_zero_beta_fraction": near_zero_fraction,
                "n_missing_time_samples_excluded": n_missing_time_samples_excluded,
                "n_technical_replicate_sample_ids_collapsed": rdata_meta[
                    "n_technical_replicate_sample_ids_collapsed"
                ],
            }
        ]
    )

    write_csv_atomic(
        inputs.reset_index().rename(columns={"index": "sample_id"}),
        output_dir / "biometa_clock_inputs.csv",
    )
    write_csv_atomic(predictions, output_dir / "sample_predictions.csv")
    write_csv_atomic(analysis_samples, output_dir / "analysis_sample_predictions.csv")
    write_csv_atomic(donor, output_dir / "donor_predictions.csv")
    write_csv_atomic(results, output_dir / "comparison_results.csv")
    write_csv_atomic(
        group_summary, output_dir / "control_experimental_prediction_summary.csv"
    )
    write_csv_atomic(age_fits, output_dir / "dataset_age_fit.csv")
    write_csv_atomic(qc, output_dir / "matrix_qc.csv")
    summary = {
        "status": "completed",
        "analysis_index": args.task_index,
        "translage_dataset_id": dataset_id,
        "public_id": str(row["Public ID"]),
        "intervention": str(row["Name"]),
        "analysis_method": analysis_method,
        "estimand": estimand,
        "baseline_time_days": baseline_time,
        "endpoint_time_days": target_time,
        "age_complete": bool(age_complete),
        "n_selected_samples": len(methylation),
        "n_selected_donors": int(design["donor_id"].nunique()),
        "n_missing_time_samples_excluded": n_missing_time_samples_excluded,
        "cleaned_rdata": relative_path(cleaned_path),
        "cleaned_rdata_sha256": sha256(cleaned_path),
        "model_weights": relative_path(model.weights_path),
        "model_weights_sha256": sha256(model.weights_path),
        "model_metadata": relative_path(model.metadata_path),
        "model_metadata_sha256": sha256(model.metadata_path),
        "manifest": relative_path(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "pyaging_batch_size": args.pyaging_batch_size,
        "minimum_cpg_coverage": MIN_CPG_COVERAGE,
        "beta_distribution_qc_applied": True,
        "beta_qc_thresholds": {
            "minimum_finite_beta_fraction": MIN_FINITE_BETA_FRACTION,
            "minimum_in_range_beta_fraction": MIN_IN_RANGE_BETA_FRACTION,
            "minimum_beta_median": MIN_BETA_MEDIAN,
            "maximum_near_zero_beta_fraction": MAX_NEAR_ZERO_BETA_FRACTION,
        },
        "clock_names": clock_names,
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "anndata",
                "numpy",
                "pandas",
                "pyaging",
                "rdata",
                "scikit-learn",
                "scipy",
                "torch",
            )
        },
        "python": platform.python_version(),
        **rdata_meta,
    }
    write_json_atomic(summary, output_dir / "run_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
