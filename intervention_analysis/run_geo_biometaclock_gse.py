#!/usr/bin/env python3
"""Predict and test the frozen intervention comparisons for one GEO dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from biological_age_prediction_and_mortality_analysis.published_biometaclock import (
    load_published_biometaclock,
)
from intervention_analysis.paths import (
    DEFAULT_MODEL_DIR,
    DEFAULT_PYAGING_DIR,
    relative_path,
    repo_child,
    repo_path,
)
from intervention_analysis.runtime import (
    DatasetSpec,
    MIN_CPG_COVERAGE,
    dataset_spec,
    load_biometa_clock_features,
    parse_beta,
    run_pyaging_clocks,
)


MIN_FINITE_BETA_FRACTION = 0.90
MIN_IN_RANGE_BETA_FRACTION = 0.99
MIN_BETA_MEDIAN = 0.02
MAX_NEAR_ZERO_BETA_FRACTION = 0.95


def normalized_probe_id(value: str) -> str:
    """Strip quoting and platform suffixes from a CpG probe identifier."""
    probe = value.strip('"')
    if match := re.match(r"^(cg\d+)(?:_.+)?$", probe, flags=re.IGNORECASE):
        return match.group(1)
    return probe


def parse_beta_value(value: str) -> float:
    """Parse one beta value, mapping the supported missing markers to NaN."""
    text = value.strip().strip('"')
    if text.upper() in {"", "NA", "NAN", "N/A", "#N/D"}:
        return float("nan")
    return float(text)


def welch_df(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate Welch-Satterthwaite degrees of freedom for two samples."""
    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    left_term = left_var / len(left)
    right_term = right_var / len(right)
    denominator = left_term**2 / (len(left) - 1) + right_term**2 / (len(right) - 1)
    return (
        float((left_term + right_term) ** 2 / denominator)
        if denominator > 0
        else float("nan")
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a sibling temporary file and atomic rename."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write JSON through a sibling temporary file and atomic rename."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_normalized_delimited(
    path: Path,
    spec: DatasetSpec,
    required_cpgs: set[str],
    sample_rows: bool,
    no_probe_header: bool = False,
    probe_column_index: int = 0,
) -> pd.DataFrame:
    """Read required CpGs and assigned samples from a gzipped delimited matrix.

    ``sample_rows`` selects whether samples occupy rows or columns. The returned
    float32 frame always has repository sample IDs as its index and CpGs as
    columns; only ``required_cpgs`` are retained.
    """
    requested = [str(item["matrix_column"]) for item in spec.samples]
    requested_sample_ids = [str(item["sample_id"]) for item in spec.samples]
    with gzip.open(
        path, "rt", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        first = handle.readline()
        while first and (not first.strip() or first.lstrip().startswith("#VALUE")):
            first = handle.readline()
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        header = [
            value.strip().strip('"')
            for value in next(csv.reader([first], delimiter=delimiter))
        ]
        if sample_rows:
            selected = [
                index for index, value in enumerate(header) if value in required_cpgs
            ]
            if not selected:
                raise ValueError(
                    f"{spec.accession}: no required CpGs in normalized header"
                )
            rows: dict[str, list[float]] = {}
            for line in handle:
                parsed = next(csv.reader([line], delimiter=delimiter))
                if parsed and parsed[0].strip('"') in requested:
                    rows[parsed[0].strip('"')] = [
                        parse_beta_value(parsed[index]) for index in selected
                    ]
            missing = sorted(set(requested).difference(rows))
            if missing:
                raise KeyError(
                    f"{spec.accession}: normalized rows missing samples {missing[:10]}"
                )
            beta = pd.DataFrame.from_dict(
                rows, orient="index", columns=[header[index] for index in selected]
            )
        else:
            missing = sorted(
                str(item["matrix_column"])
                for item in spec.samples
                if "matrix_column_index_actual" not in item
                and str(item["matrix_column"]) not in header
            )
            if missing:
                raise KeyError(
                    f"{spec.accession}: normalized columns missing samples {missing[:10]}"
                )
            indices: list[int] = []
            for item in spec.samples:
                value = str(item["matrix_column"])
                if "matrix_column_index_actual" in item:
                    index = int(item["matrix_column_index_actual"])
                    if not 0 <= index < len(header) or header[index] != value:
                        raise ValueError(
                            f"{spec.accession}: verified column index {index} does not "
                            f"match expected header {value!r}"
                        )
                else:
                    index = header.index(value)
                indices.append(index + int(no_probe_header))
            rows = {}
            for line in handle:
                parsed = next(csv.reader([line], delimiter=delimiter))
                raw_probe = (
                    parsed[probe_column_index].strip('"')
                    if len(parsed) > probe_column_index
                    else ""
                )
                probe = normalized_probe_id(raw_probe)
                if probe not in required_cpgs:
                    continue
                rows[probe] = [parse_beta_value(parsed[index]) for index in indices]
            beta = pd.DataFrame.from_dict(
                rows, orient="index", columns=requested_sample_ids
            ).T
    if sample_rows:
        matrix_to_sample = {
            str(item["matrix_column"]): str(item["sample_id"]) for item in spec.samples
        }
        beta.index = [matrix_to_sample[str(value)] for value in beta.index]
    beta.index.name = "sample_id"
    return beta.astype(np.float32)


def parse_assigned_beta(
    path: Path, spec: DatasetSpec, required_cpgs: set[str]
) -> pd.DataFrame:
    """Dispatch one frozen matrix assignment to its format-specific parser.

    Methylated/unmethylated signal inputs are converted to beta values with the
    manifest's assigned samples. Other normalized layouts are delegated to the
    delimited or GEO series-matrix parsers.
    """
    if spec.matrix_format == "normalized_delimited_meth_unmeth":
        requested = [str(item["matrix_column"]) for item in spec.samples]
        with gzip.open(
            path, "rt", encoding="utf-8", errors="replace", newline=""
        ) as handle:
            first = handle.readline()
            while first and not first.strip():
                first = handle.readline()
            delimiter = "\t" if first.count("\t") > first.count(",") else ","
            header = [
                value.strip('"')
                for value in next(csv.reader([first], delimiter=delimiter))
            ]
            methylated = [
                header.index(f"{value}_methylated_signal") for value in requested
            ]
            unmethylated = [
                header.index(f"{value}_unmethylated_signal") for value in requested
            ]
            rows: dict[str, list[float]] = {}
            for line in handle:
                parsed = next(csv.reader([line], delimiter=delimiter))
                probe = normalized_probe_id(parsed[0]) if parsed else ""
                if probe not in required_cpgs:
                    continue
                m_values = np.asarray([float(parsed[index]) for index in methylated])
                u_values = np.asarray([float(parsed[index]) for index in unmethylated])
                rows[probe] = (m_values / (m_values + u_values + 100.0)).tolist()
        beta = pd.DataFrame.from_dict(rows, orient="index", columns=requested).T
        matrix_to_sample = {
            str(item["matrix_column"]): str(item["sample_id"]) for item in spec.samples
        }
        beta.index = [matrix_to_sample[str(value)] for value in beta.index]
        beta.index.name = "sample_id"
        return beta.astype(np.float32)
    if spec.matrix_format == "normalized_delimited_cpg_rows":
        return read_normalized_delimited(path, spec, required_cpgs, sample_rows=False)
    if spec.matrix_format == "normalized_delimited_sample_rows":
        return read_normalized_delimited(path, spec, required_cpgs, sample_rows=True)
    if spec.matrix_format == "normalized_delimited_cpg_rows_no_probe_header":
        return read_normalized_delimited(
            path,
            spec,
            required_cpgs,
            sample_rows=False,
            no_probe_header=True,
        )
    if spec.matrix_format == "normalized_delimited_cpg_rows_indexed":
        return read_normalized_delimited(
            path,
            spec,
            required_cpgs,
            sample_rows=False,
            probe_column_index=1,
        )
    return parse_beta(path, spec, required_cpgs)


def age_correct_strata(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize BioMetaClock predictions on age within each declared stratum.

    Returns the sample rows with raw and residualized predictions together with
    one audit row per stratum describing the fitted linear or intercept-only
    adjustment.
    """
    corrected_parts: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    for stratum, rows in samples.groupby("age_correction_stratum", sort=False):
        unique = rows[
            ["sample_id", "chronological_age", "predicted_bioage"]
        ].drop_duplicates()
        if unique["sample_id"].duplicated().any():
            raise ValueError(f"{stratum}: conflicting sample ages or predictions")
        age = unique["chronological_age"].to_numpy(dtype=float)
        prediction = unique["predicted_bioage"].to_numpy(dtype=float)
        if not np.isfinite(age).all() or not np.isfinite(prediction).all():
            raise ValueError(f"{stratum}: non-finite age or BioMeta prediction")
        reference_age = float(np.mean(age))
        if len(np.unique(age)) >= 2:
            model = LinearRegression().fit(age.reshape(-1, 1), prediction)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            r2 = float(model.score(age.reshape(-1, 1), prediction))
            method = "linear_regression_residual_within_gse_stratum"
        else:
            slope = 0.0
            intercept = float(np.mean(prediction))
            r2 = float("nan")
            method = "intercept_only_residual_constant_age"
        unique["age_corrected_predicted_age"] = prediction - (intercept + slope * age)
        corrected_parts.append(
            rows.drop(columns=["predicted_bioage"]).merge(
                unique[
                    ["sample_id", "predicted_bioage", "age_corrected_predicted_age"]
                ],
                on="sample_id",
                how="left",
                validate="many_to_one",
            )
        )
        fits.append(
            {
                "gse_id": str(rows["gse_id"].iloc[0]),
                "age_correction_stratum": str(stratum),
                "age_adjustment": method,
                "age_fit_n_unique_samples": len(unique),
                "age_fit_n_unique_ages": int(len(np.unique(age))),
                "age_reference_age": reference_age,
                "age_fit_slope": slope,
                "age_fit_intercept": intercept,
                "age_fit_r2": r2,
            }
        )
    return pd.concat(corrected_parts, ignore_index=True), pd.DataFrame(fits)


def summarize_comparisons(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse replicate samples by donor and run each declared comparison.

    The manifest's analysis method determines paired versus Welch testing and
    whether pre/post values are differenced first. Effects and test statistics
    use experiment minus control. The returned frames contain comparison-level
    results and the donor-level values used by those tests, respectively.
    """
    donor_frames: list[pd.DataFrame] = []
    for comparison_id, sub in samples.groupby("comparison_id", sort=False):
        methods = sub["analysis_method"].drop_duplicates().tolist()
        if len(methods) != 1:
            raise ValueError(f"{comparison_id}: analysis_method must be constant")
        if methods[0] == "donor_level_paired_no_age_correction":
            donor_sets = {
                group: set(rows["donor_id"])
                for group, rows in sub.groupby("group", sort=False)
            }
            if (
                set(donor_sets) != {"control", "experiment"}
                or donor_sets["control"] != donor_sets["experiment"]
            ):
                raise ValueError(
                    f"{comparison_id}: no-age analysis requires identical donor sets"
                )
        keys = [
            "comparison_id",
            "gse_id",
            "comparison_name",
            "canonical_comparison_id",
            "group",
            "donor_id",
        ]
        if methods[0] in {
            "donor_level_welch",
            "donor_level_paired",
            "donor_level_partial_paired",
            "donor_level_paired_no_age_correction",
        }:
            collapsed = sub.groupby(keys, as_index=False).agg(
                chronological_age=("chronological_age", "mean"),
                predicted_bioage=("predicted_bioage", "mean"),
                analysis_predicted_age=(
                    "analysis_predicted_age",
                    "mean",
                ),
                n_samples_collapsed=("sample_id", "nunique"),
            )
        elif methods[0] == "donor_pre_post_change_welch":
            if set(sub["timepoint"]) != {"pre", "post"}:
                raise ValueError(
                    f"{comparison_id}: change analysis requires pre and post labels"
                )
            by_time = sub.groupby([*keys, "timepoint"], as_index=False).agg(
                chronological_age=("chronological_age", "mean"),
                predicted_bioage=("predicted_bioage", "mean"),
                analysis_predicted_age=(
                    "analysis_predicted_age",
                    "mean",
                ),
                n_samples_collapsed=("sample_id", "nunique"),
            )
            counts = by_time.groupby(keys)["timepoint"].nunique()
            if not bool(counts.eq(2).all()):
                raise ValueError(
                    f"{comparison_id}: every donor requires both pre and post"
                )
            index = by_time.set_index([*keys, "timepoint"])
            post = index.xs("post", level="timepoint")
            pre = index.xs("pre", level="timepoint")
            if not post.index.equals(pre.index):
                raise ValueError(f"{comparison_id}: pre/post donor sets differ")
            collapsed = post.reset_index()[keys].copy()
            collapsed["chronological_age"] = (
                post["chronological_age"].to_numpy()
                + pre["chronological_age"].to_numpy()
            ) / 2.0
            collapsed["predicted_bioage"] = (
                post["predicted_bioage"].to_numpy() - pre["predicted_bioage"].to_numpy()
            )
            collapsed["analysis_predicted_age"] = (
                post["analysis_predicted_age"].to_numpy()
                - pre["analysis_predicted_age"].to_numpy()
            )
            collapsed["n_samples_collapsed"] = (
                post["n_samples_collapsed"].to_numpy()
                + pre["n_samples_collapsed"].to_numpy()
            )
        else:
            raise ValueError(
                f"{comparison_id}: unsupported analysis_method={methods[0]}"
            )
        if methods[0] in {"donor_level_paired", "donor_level_partial_paired"}:
            paired = set(
                collapsed.loc[collapsed["group"].eq("control"), "donor_id"]
            ) & set(collapsed.loc[collapsed["group"].eq("experiment"), "donor_id"])
            collapsed = collapsed.loc[collapsed["donor_id"].isin(paired)].copy()
        collapsed["analysis_method"] = methods[0]
        donor_frames.append(collapsed)
    donor = pd.concat(donor_frames, ignore_index=True)
    results: list[dict[str, Any]] = []
    for comparison_id, sub in donor.groupby("comparison_id", sort=False):
        control_rows = sub.loc[sub["group"].eq("control")].sort_values("donor_id")
        experiment_rows = sub.loc[sub["group"].eq("experiment")].sort_values("donor_id")
        control = control_rows["analysis_predicted_age"].to_numpy(dtype=float)
        experiment = experiment_rows["analysis_predicted_age"].to_numpy(dtype=float)
        if min(len(control), len(experiment)) < 4:
            raise ValueError(
                f"{comparison_id}: fewer than 4 donor observations in one arm"
            )
        method = str(sub["analysis_method"].iloc[0])
        if method in {
            "donor_level_paired",
            "donor_level_partial_paired",
            "donor_level_paired_no_age_correction",
        }:
            if (
                control_rows["donor_id"].tolist()
                != experiment_rows["donor_id"].tolist()
            ):
                raise ValueError(f"{comparison_id}: paired donor IDs do not align")
            test = stats.ttest_rel(experiment, control)
            test_name = "two_sided_paired_t_test"
            test_df = float(len(control) - 1)
            n_paired = len(control)
        else:
            test = stats.ttest_ind(experiment, control, equal_var=False)
            test_name = "two_sided_welch_t_test"
            test_df = welch_df(experiment, control)
            n_paired = 0
        test_statistic = float(test.statistic)
        test_p_value = float(test.pvalue)
        if not np.isfinite(test_p_value):
            mean_difference = float(np.mean(experiment) - np.mean(control))
            degenerate = (
                np.var(experiment - control) == 0
                if n_paired
                else np.var(experiment) == 0 and np.var(control) == 0
            )
            if degenerate:
                test_statistic = (
                    0.0
                    if mean_difference == 0
                    else float(np.copysign(np.inf, mean_difference))
                )
                test_p_value = 1.0 if mean_difference == 0 else 0.0
            else:
                raise ValueError(
                    f"{comparison_id}: t-test returned a non-finite p-value"
                )
        first = sub.iloc[0]
        no_age_correction = method == "donor_level_paired_no_age_correction"
        results.append(
            {
                "comparison_id": comparison_id,
                "gse_id": first["gse_id"],
                "comparison_name": first["comparison_name"],
                "analysis_method": first["analysis_method"],
                "test": test_name,
                "control_n_donors": len(control),
                "experiment_n_donors": len(experiment),
                "n_paired_donors_analyzed": n_paired,
                "control_mean_raw_bioage": float(
                    sub.loc[sub["group"].eq("control"), "predicted_bioage"].mean()
                ),
                "experiment_mean_raw_bioage": float(
                    sub.loc[sub["group"].eq("experiment"), "predicted_bioage"].mean()
                ),
                "control_mean_age_corrected_predicted_age": (
                    float("nan") if no_age_correction else float(np.mean(control))
                ),
                "experiment_mean_age_corrected_predicted_age": (
                    float("nan") if no_age_correction else float(np.mean(experiment))
                ),
                "control_mean_analysis_predicted_age": float(np.mean(control)),
                "experiment_mean_analysis_predicted_age": float(np.mean(experiment)),
                "mean_diff_experiment_minus_control": float(
                    np.mean(experiment) - np.mean(control)
                ),
                "delta_control_minus_experiment": float(
                    np.mean(control) - np.mean(experiment)
                ),
                "test_statistic_experiment_minus_control": test_statistic,
                "test_df": test_df,
                "two_sided_p_value": test_p_value,
                "age_adjustment": (
                    "none_fully_paired_no_chronological_age"
                    if no_age_correction
                    else "within_gse_stratum_prediction_scale"
                ),
                "canonical_comparison_id": first["canonical_comparison_id"],
            }
        )
    return pd.DataFrame(results), donor


def main() -> int:
    started_at_ns = time.time_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=repo_path, required=True)
    parser.add_argument("--matrix-manifest", type=repo_path, required=True)
    parser.add_argument("--gse", required=True)
    parser.add_argument("--matrix-root", type=repo_path, required=True)
    parser.add_argument("--output-dir", type=repo_path, required=True)
    parser.add_argument("--model-dir", type=repo_path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pyaging-dir", type=repo_path, default=DEFAULT_PYAGING_DIR)
    parser.add_argument("--pyaging-batch-size", type=int, default=2048)
    parser.add_argument("--min-cpg-coverage", type=float, default=MIN_CPG_COVERAGE)
    parser.add_argument(
        "--skip-beta-distribution-qc",
        action="store_true",
        help=(
            "Sensitivity-analysis mode: do not reject matrices using the 99%% "
            "in-range fraction, finite-beta median, or near-zero-beta fraction "
            "thresholds. Finite-value, per-sample finite-value, CpG-coverage, and "
            "strict [0, 1] beta-range checks remain active."
        ),
    )
    args = parser.parse_args()
    for label, path in (
        ("sample manifest", args.sample_manifest),
        ("matrix manifest", args.matrix_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {relative_path(path)}")
    for stale_name in (
        "failure_summary.json",
        "run_summary.json",
        "comparison_results.csv",
        "sample_predictions.csv",
        "analysis_sample_predictions.csv",
        "donor_age_corrected_predictions.csv",
        "donor_paired_raw_predictions.csv",
        "dataset_age_fit.csv",
        "biometa_clock_inputs.csv",
        "matrix_qc.csv",
    ):
        (args.output_dir / stale_name).unlink(missing_ok=True)

    manifest = pd.read_csv(args.sample_manifest, dtype=str)
    manifest = manifest.loc[manifest["gse_id"].eq(args.gse)].copy()
    if manifest.empty:
        raise ValueError(f"No eligible manifest rows for {args.gse}")
    methods = set(manifest["analysis_method"].dropna())
    no_age_correction = methods == {"donor_level_paired_no_age_correction"}
    if "donor_level_paired_no_age_correction" in methods and not no_age_correction:
        raise ValueError(f"{args.gse}: cannot mix age-corrected and no-age methods")
    manifest["chronological_age"] = pd.to_numeric(
        manifest["chronological_age"],
        errors="coerce" if no_age_correction else "raise",
    )
    matrices = pd.read_csv(args.matrix_manifest, dtype=str)
    matrices = matrices.loc[matrices["gse_id"].eq(args.gse)].copy()
    if matrices.empty:
        raise ValueError(f"No matrix assignments for {args.gse}")
    assignments = (
        manifest[["gse_id", "sample_id", "matrix_column"]]
        .drop_duplicates()
        .merge(
            matrices,
            on=["gse_id", "sample_id", "matrix_column"],
            how="left",
            validate="one_to_one",
        )
    )
    if assignments["matrix_file"].isna().any():
        missing = assignments.loc[
            assignments["matrix_file"].isna(), "sample_id"
        ].tolist()
        raise ValueError(f"{args.gse}: missing matrix assignment for {missing[:10]}")

    model, clock_names = load_published_biometaclock(args.model_dir)
    required_features = load_biometa_clock_features(clock_names, args.pyaging_dir)
    if no_age_correction:
        non_cpg_features = [
            feature
            for feature in required_features
            if not feature.lower().startswith(("cg", "ch."))
        ]
        if non_cpg_features:
            raise ValueError(
                "No-age analysis requested, but a selected clock requires non-CpG "
                f"features: {non_cpg_features}"
            )
    required_cpgs = {
        feature
        for feature in required_features
        if feature.lower().startswith("cg") or feature.lower().startswith("ch.")
    }
    beta_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    for (matrix_file, matrix_format), rows in assignments.groupby(
        ["matrix_file", "matrix_format"], sort=False
    ):
        spec_rows = rows.copy()
        if "matrix_column_actual" in spec_rows:
            spec_rows["matrix_column"] = spec_rows["matrix_column_actual"]
        spec_rows = spec_rows.assign(
            analysis_id=args.gse,
            dataset=args.gse,
            intervention="",
            group="control",
            chronological_age=18.0,
            female=0.0,
        )
        spec = dataset_spec(args.gse, spec_rows)
        path = repo_child(args.matrix_root, str(matrix_file))
        if not path.is_file():
            raise FileNotFoundError(f"{args.gse}: missing matrix {relative_path(path)}")
        beta = parse_assigned_beta(path, spec, required_cpgs)
        coverage = len(set(beta.columns).intersection(required_cpgs)) / len(
            required_cpgs
        )
        if coverage < args.min_cpg_coverage:
            raise ValueError(
                f"{args.gse}/{matrix_file}: CpG coverage {coverage:.4f} below "
                f"{args.min_cpg_coverage:.4f}"
            )
        beta_values = beta.to_numpy(dtype=float)
        finite_mask = np.isfinite(beta_values)
        finite_fraction = float(np.mean(finite_mask))
        minimum_sample_finite_fraction = float(np.min(np.mean(finite_mask, axis=1)))
        finite_values = beta_values[finite_mask]
        in_range_fraction = (
            float(np.mean((finite_values >= 0.0) & (finite_values <= 1.0)))
            if len(finite_values)
            else 0.0
        )
        beta_min = float(np.min(finite_values)) if len(finite_values) else float("nan")
        beta_max = float(np.max(finite_values)) if len(finite_values) else float("nan")
        beta_median = (
            float(np.median(finite_values)) if len(finite_values) else float("nan")
        )
        near_zero_fraction = (
            float(np.mean(finite_values <= 0.01)) if len(finite_values) else 1.0
        )
        if finite_fraction < MIN_FINITE_BETA_FRACTION:
            raise ValueError(
                f"{args.gse}/{matrix_file}: finite beta fraction {finite_fraction:.4f} "
                f"below {MIN_FINITE_BETA_FRACTION:.4f}"
            )
        if minimum_sample_finite_fraction < MIN_FINITE_BETA_FRACTION:
            raise ValueError(
                f"{args.gse}/{matrix_file}: minimum per-sample finite beta fraction "
                f"{minimum_sample_finite_fraction:.4f} below "
                f"{MIN_FINITE_BETA_FRACTION:.4f}"
            )
        if beta_min < 0.0 or beta_max > 1.0:
            raise ValueError(
                f"{args.gse}/{matrix_file}: finite beta values must all be within "
                f"[0, 1] (observed range {beta_min:.6g} to {beta_max:.6g})"
            )
        if (
            not args.skip_beta_distribution_qc
            and in_range_fraction < MIN_IN_RANGE_BETA_FRACTION
        ):
            raise ValueError(
                f"{args.gse}/{matrix_file}: only {in_range_fraction:.4f} of finite values "
                "are within the expected beta range [0, 1]"
            )
        if not args.skip_beta_distribution_qc and (
            beta_median < MIN_BETA_MEDIAN
            or near_zero_fraction > MAX_NEAR_ZERO_BETA_FRACTION
        ):
            raise ValueError(
                f"{args.gse}/{matrix_file}: values resemble a detection-p matrix, "
                f"not methylation beta values (median={beta_median:.6g}, "
                f"fraction<=0.01={near_zero_fraction:.4f})"
            )
        beta_parts.append(beta)
        qc_rows.append(
            {
                "gse_id": args.gse,
                "matrix_file": matrix_file,
                "matrix_format": matrix_format,
                "n_samples": len(beta),
                "required_cpg_coverage": coverage,
                "finite_beta_fraction": finite_fraction,
                "minimum_sample_finite_beta_fraction": minimum_sample_finite_fraction,
                "in_range_beta_fraction": in_range_fraction,
                "finite_beta_min": beta_min,
                "finite_beta_max": beta_max,
                "finite_beta_median": beta_median,
                "near_zero_beta_fraction": near_zero_fraction,
                "beta_distribution_qc_applied": not args.skip_beta_distribution_qc,
            }
        )
    methylation = pd.concat(beta_parts, axis=0, sort=False)
    if methylation.index.duplicated().any():
        raise ValueError(f"{args.gse}: duplicate samples across matrix files")
    sample_age = manifest[["sample_id", "chronological_age"]].drop_duplicates()
    if sample_age["sample_id"].duplicated().any():
        raise ValueError(f"{args.gse}: one sample has conflicting chronological ages")
    sample_meta = sample_age.set_index("sample_id").loc[methylation.index]
    clock_input = methylation.copy()
    if not no_age_correction:
        clock_input["age"] = sample_meta["chronological_age"].to_numpy(dtype=float)
    clock_outputs = run_pyaging_clocks(
        frame=clock_input,
        sample_id=pd.Series(methylation.index.astype(str)),
        clocks=clock_names,
        pyaging_dir=args.pyaging_dir,
        batch_size=args.pyaging_batch_size,
        required_features=required_features,
    ).set_index("sample_id")
    X = (
        clock_outputs.loc[methylation.index, clock_names]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    predictions = pd.DataFrame(
        {
            "gse_id": args.gse,
            "sample_id": methylation.index,
            "chronological_age": sample_meta["chronological_age"].to_numpy(dtype=float),
            "predicted_bioage": np.asarray(model.predict(X), dtype=float),
        }
    )
    analysis_samples = manifest.merge(
        predictions[["sample_id", "predicted_bioage"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if no_age_correction:
        analysis_samples["analysis_predicted_age"] = analysis_samples[
            "predicted_bioage"
        ]
        age_fits = pd.DataFrame(
            [
                {
                    "gse_id": args.gse,
                    "age_correction_stratum": "none",
                    "age_adjustment": "none_fully_paired_no_chronological_age",
                    "age_fit_n_unique_samples": 0,
                    "age_fit_n_unique_ages": 0,
                    "age_reference_age": np.nan,
                    "age_fit_slope": np.nan,
                    "age_fit_intercept": np.nan,
                    "age_fit_r2": np.nan,
                }
            ]
        )
    else:
        analysis_samples, age_fits = age_correct_strata(analysis_samples)
        analysis_samples["analysis_predicted_age"] = analysis_samples[
            "age_corrected_predicted_age"
        ]
    results, donor = summarize_comparisons(analysis_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(predictions, args.output_dir / "sample_predictions.csv")
    write_csv_atomic(
        analysis_samples, args.output_dir / "analysis_sample_predictions.csv"
    )
    donor_output = (
        "donor_paired_raw_predictions.csv"
        if no_age_correction
        else "donor_age_corrected_predictions.csv"
    )
    write_csv_atomic(donor, args.output_dir / donor_output)
    write_csv_atomic(results, args.output_dir / "comparison_results.csv")
    write_csv_atomic(pd.DataFrame(qc_rows), args.output_dir / "matrix_qc.csv")
    write_csv_atomic(age_fits, args.output_dir / "dataset_age_fit.csv")
    write_csv_atomic(
        clock_outputs.reset_index(), args.output_dir / "biometa_clock_inputs.csv"
    )
    summary = {
        "cache_format_version": 1,
        "started_at_ns": started_at_ns,
        "input_files": sorted(
            relative_path(path)
            for path in {args.sample_manifest, args.matrix_manifest}
            | {repo_child(args.matrix_root, str(name)) for name in assignments["matrix_file"].unique()}
        ),
        "model_parameters": model.parameters,
        "gse_id": args.gse,
        "n_comparisons": len(results),
        "n_unique_samples": len(predictions),
        "model_weights": relative_path(model.weights_path),
        "model_metadata": relative_path(model.metadata_path),
        "training_n_participants": 672,
        "beta_distribution_qc_applied": not args.skip_beta_distribution_qc,
        "skipped_beta_distribution_qc_thresholds": (
            []
            if not args.skip_beta_distribution_qc
            else [
                "minimum_in_range_beta_fraction",
                "minimum_finite_beta_median",
                "maximum_near_zero_beta_fraction",
            ]
        ),
        "delta_definition": (
            "experiment minus control on raw predicted BioMeta age within paired donors"
            if no_age_correction
            else "experiment minus control on age-corrected predicted age"
        ),
        "test": (
            "two-sided paired t-test on identical donor sets"
            if no_age_correction
            else "paired t-test for overlapping donor designs; Welch t-test otherwise"
        ),
        "age_fit_scope": (
            "none"
            if no_age_correction
            else "once per GSE and curated correction stratum across unique selected GSMs"
        ),
        "age_adjustment": (
            "none_fully_paired_no_chronological_age"
            if no_age_correction
            else "within_gse_stratum_prediction_scale"
        ),
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "anndata",
                "numpy",
                "pandas",
                "pyaging",
                "scikit-learn",
                "scipy",
                "torch",
            )
        },
    }
    write_json_atomic(summary, args.output_dir / "run_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            failure_dir = repo_path(sys.argv[sys.argv.index("--output-dir") + 1])
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure_summary.json").write_text(
                json.dumps(
                    {
                        "gse_id": (
                            sys.argv[sys.argv.index("--gse") + 1]
                            if "--gse" in sys.argv
                            else ""
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
