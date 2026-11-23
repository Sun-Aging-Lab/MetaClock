"""Portable MetaClock-19 re-scoring bridge for the external intervention panel.

The bridge starts from vetted 19-component predictions.  It deliberately does
not parse methylation matrices or require PyAging.  A public linearized export
of the frozen ``SimpleImputer(0) -> StandardScaler -> ElasticNet`` pipeline is
used to reproduce sample scores, after which the repository's established
age-correction and donor-level inference functions are called unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from intervention_analysis.run_geo_biometaclock_gse import (
    age_correct_strata,
    summarize_comparisons,
)
from biological_age_prediction_and_mortality_analysis.published_biometaclock import (
    MODEL_METADATA_FILE,
    MODEL_WEIGHTS_FILE,
    PublishedBioMetaClock,
    load_published_biometaclock,
)


HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHT_DIR = HERE / "config" / "external_full19"
WEIGHT_FILE = MODEL_WEIGHTS_FILE
METADATA_FILE = MODEL_METADATA_FILE
EXPECTED_SOURCE_MODEL_SHA256 = "1207762c578c793e9959a0b9590101734ad328ddb946fa96d2b16cb00b818621"
EXPECTED_WEIGHT_SHA256 = "4a9b3267fe84391295d4588e554943fc6f907aac6f4b88188254fc0472431759"
TASK_MANIFESTS = ("tasks_initial.json", "tasks_recovered.json")
SAFE_TASK_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
PAIRED_METHODS = {
    "donor_level_paired",
    "donor_level_partial_paired",
    "donor_level_paired_no_age_correction",
}
PANEL_REQUIRED = {
    "family_test_id",
    "comparison_id",
    "public_id",
    "source",
    "source_dataset_id",
    "intervention",
    "display_intervention",
    "analysis_method",
    "test",
}
DESIGN_REQUIRED = {
    "comparison_id",
    "gse_id",
    "comparison_name",
    "sample_id",
    "group",
    "donor_id",
    "chronological_age",
    "analysis_method",
    "age_correction_stratum",
    "canonical_comparison_id",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frozen_linear_model(
    weight_dir: Path = DEFAULT_WEIGHT_DIR,
) -> PublishedBioMetaClock:
    """Load the standard published model export and pin its manuscript identity."""
    weight_dir = Path(weight_dir)
    weight_path = weight_dir / WEIGHT_FILE
    metadata_path = weight_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("weights_sha256") != _sha256(weight_path):
        raise ValueError("Frozen weight CSV checksum differs from metadata")
    if metadata.get("weights_sha256") != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Frozen weight CSV is not the manuscript model export")
    if metadata.get("source_model_sha256") != EXPECTED_SOURCE_MODEL_SHA256:
        raise ValueError("Frozen weights do not identify the manuscript source model")
    if metadata.get("model_variant") != "external_intervention_metaclock19_full":
        raise ValueError("Frozen weights are not the external full19 model variant")
    if metadata.get("imputation", {}).get("fill_value") != 0.0:
        raise ValueError("This bridge supports the manuscript's zero imputation only")
    model, features = load_published_biometaclock(weight_dir)
    if len(features) != 19:
        raise ValueError("Frozen external model must have exactly 19 components")
    return model


def score_components(
    components: pd.DataFrame,
    model: PublishedBioMetaClock,
) -> pd.DataFrame:
    """Score a unique sample-by-component table with the frozen linear model."""
    _require_columns(components, {"sample_id", *model.features}, "component table")
    if components["sample_id"].astype(str).eq("").any() or components["sample_id"].duplicated().any():
        raise ValueError("component table sample_id values must be nonempty and unique")
    numeric = components.loc[:, model.features].apply(pd.to_numeric, errors="raise")
    matrix = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    predictions = model.predict(matrix.to_numpy(dtype=float))
    if not np.isfinite(predictions).all():
        raise ValueError("Frozen full19 model returned non-finite predictions")
    return pd.DataFrame(
        {
            "sample_id": components["sample_id"].astype(str).to_numpy(),
            "predicted_bioage": predictions,
        }
    )


def _normalized_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if "family_test_id" not in panel and panel.index.name == "family_test_id":
        panel = panel.reset_index()
    else:
        panel = panel.copy()
    _require_columns(panel, PANEL_REQUIRED, "panel")
    if panel["family_test_id"].duplicated().any() or panel["family_test_id"].astype(str).eq("").any():
        raise ValueError("panel family_test_id values must be nonempty and unique")
    if "gse_id" in panel and not panel["gse_id"].astype(str).eq(panel["public_id"].astype(str)).all():
        raise ValueError("panel gse_id and public_id must agree")
    return panel.set_index("family_test_id", verify_integrity=True, drop=False)


def _safe_shard(run_dir: Path, task_key: str) -> Path:
    if not SAFE_TASK_KEY.fullmatch(task_key) or Path(task_key).name != task_key:
        raise ValueError(f"Unsafe task_key: {task_key!r}")
    shard_root = (run_dir / "shards").resolve()
    shard = (shard_root / task_key).resolve()
    try:
        shard.relative_to(shard_root)
    except ValueError as error:
        raise ValueError(f"task_key escapes the shard directory: {task_key!r}") from error
    return shard


def _task_inventory(run_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in TASK_MANIFESTS:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing task manifest: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{name} must contain a JSON list")
        for task in payload:
            required = {"task_key", "kind", "public_id", "retained_families"}
            if not isinstance(task, dict) or not required.issubset(task):
                raise ValueError(f"Invalid task in {name}")
            key = str(task["task_key"])
            _safe_shard(run_dir, key)
            if key in seen:
                raise ValueError(f"Repeated task_key across manifests: {key}")
            if task["kind"] not in {"geo", "translage"}:
                raise ValueError(f"Unsupported task kind for {key}: {task['kind']}")
            seen.add(key)
            tasks.append(task)
    return tasks


def _design_from_shard(shard: Path) -> pd.DataFrame:
    """Read design columns only; historical scores are never recomputation inputs."""
    path = shard / "analysis_sample_predictions.csv"
    design = pd.read_csv(path, dtype={"sample_id": str, "donor_id": str})
    _require_columns(design, DESIGN_REQUIRED, "analysis design shard")
    drop = {
        "predicted_bioage",
        "age_corrected_predicted_age",
        "analysis_predicted_age",
    }.intersection(design.columns)
    design = design.drop(columns=sorted(drop))
    design["chronological_age"] = pd.to_numeric(design["chronological_age"], errors="raise")
    if not np.isfinite(design["chronological_age"].to_numpy(float)).all():
        raise ValueError(f"{shard.name}: chronological ages must be finite")
    if not set(design["group"].astype(str)).issubset({"control", "experiment"}):
        raise ValueError(f"{shard.name}: invalid comparison group")
    return design


def _score_complete_task(
    task: dict[str, Any],
    run_dir: Path,
    model: PublishedBioMetaClock,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score and analyze the complete task design before selecting panel families."""
    shard = _safe_shard(run_dir, str(task["task_key"]))
    component_path = shard / "component_predictions_19.csv"
    components = pd.read_csv(component_path, dtype={"sample_id": str})
    scores = score_components(components, model)
    design = _design_from_shard(shard)
    expected_samples = set(design["sample_id"].astype(str))
    if expected_samples != set(scores["sample_id"]):
        raise ValueError(
            f"{shard.name}: component/design sample sets differ; "
            f"missing scores={len(expected_samples - set(scores['sample_id']))}, "
            f"extra scores={len(set(scores['sample_id']) - expected_samples)}"
        )
    analysis = design.merge(scores, on="sample_id", how="left", validate="many_to_one")
    analysis, _ = age_correct_strata(analysis)
    analysis["analysis_predicted_age"] = analysis["age_corrected_predicted_age"]
    results, donors = summarize_comparisons(analysis)
    prefix = "geo::" if task["kind"] == "geo" else "translage::"
    results["family_test_id"] = prefix + results["comparison_id"].astype(str)
    donors["family_test_id"] = prefix + donors["comparison_id"].astype(str)
    for signature in ("sample_signature_sha256", "sample_signature_id"):
        if signature not in analysis:
            continue
        signatures = analysis.groupby("comparison_id", sort=False)[signature].agg(
            lambda values: tuple(pd.unique(values.astype(str)))
        )
        if any(len(values) != 1 for values in signatures):
            raise ValueError(f"{task['task_key']}: conflicting {signature} values")
        results[signature] = results["comparison_id"].map(
            signatures.map(lambda values: values[0])
        )
    return results, donors


def _family_moments(donors: pd.DataFrame, panel_row: pd.Series) -> dict[str, Any]:
    control_rows = donors.loc[donors["group"].eq("control")].sort_values("donor_id")
    experiment_rows = donors.loc[donors["group"].eq("experiment")].sort_values("donor_id")
    control = control_rows["analysis_predicted_age"].to_numpy(dtype=float)
    experiment = experiment_rows["analysis_predicted_age"].to_numpy(dtype=float)
    method = str(panel_row["analysis_method"])
    paired_sd = float("nan")
    if method in PAIRED_METHODS:
        if control_rows["donor_id"].astype(str).tolist() != experiment_rows["donor_id"].astype(str).tolist():
            raise ValueError(f"{panel_row.name}: paired donor IDs do not align")
        paired_sd = float(np.std(experiment - control, ddof=1))
    row = panel_row.to_dict()
    row.update(
        {
            "family_test_id": str(panel_row.name),
            "gse_id": str(panel_row["public_id"]),
            "control_n_donors": int(len(control)),
            "experiment_n_donors": int(len(experiment)),
            "control_mean": float(np.mean(control)),
            "experiment_mean": float(np.mean(experiment)),
            "control_sd": float(np.std(control, ddof=1)),
            "experiment_sd": float(np.std(experiment, ddof=1)),
            "paired_difference_sd": paired_sd,
        }
    )
    return row


def compute_moments(
    panel: pd.DataFrame,
    run_dir: Path,
    weight_dir: Path = DEFAULT_WEIGHT_DIR,
) -> pd.DataFrame:
    """Return standardized per-family sufficient statistics for inference.

    Each represented task is scored and age-corrected using its complete design;
    panel family selection happens only after repository inference finishes.
    """
    panel = _normalized_panel(panel)
    run_dir = Path(run_dir)
    model = load_frozen_linear_model(weight_dir)
    represented: set[str] = set()
    moment_rows: list[dict[str, Any]] = []
    for task in _task_inventory(run_dir):
        families = [value for value in task["retained_families"] if value in panel.index]
        if not families:
            continue
        overlap = represented.intersection(families)
        if overlap:
            raise ValueError(f"Multiple shards represent panel families: {sorted(overlap)}")
        if set(panel.loc[families, "public_id"].astype(str)) != {str(task["public_id"])}:
            raise ValueError(f"Panel accession differs from shard {task['task_key']}")
        results, donors = _score_complete_task(task, run_dir, model)
        by_family = results.set_index("family_test_id", verify_integrity=True)
        for family in families:
            if family not in by_family.index:
                raise ValueError(f"{task['task_key']}: missing recomputed family {family}")
            panel_row = panel.loc[family]
            result_row = by_family.loc[family]
            if str(result_row["comparison_id"]) != str(panel_row["comparison_id"]):
                raise ValueError(f"{family}: comparison_id differs from recomputed design")
            if str(result_row["gse_id"]) != str(panel_row["public_id"]):
                raise ValueError(f"{family}: public accession differs from recomputed design")
            for column in ("analysis_method", "test"):
                if str(result_row[column]) != str(panel_row[column]):
                    raise ValueError(f"{family}: {column} differs from recomputed design")
            for column in (
                "control_n_donors",
                "experiment_n_donors",
                "n_paired_donors_analyzed",
            ):
                if column in panel_row and str(panel_row[column]) not in {"", "nan"}:
                    if int(result_row[column]) != int(panel_row[column]):
                        raise ValueError(f"{family}: {column} differs from recomputed design")
            for signature in ("sample_signature_sha256", "sample_signature_id"):
                if signature in panel_row and str(panel_row[signature]) not in {"", "nan"}:
                    if signature not in result_row or str(result_row[signature]) != str(panel_row[signature]):
                        raise ValueError(f"{family}: {signature} differs from recomputed design")
            family_donors = donors.loc[donors["family_test_id"].eq(family)]
            moment_rows.append(_family_moments(family_donors, panel_row))
        represented.update(families)
    missing = set(panel.index).difference(represented)
    if missing:
        raise ValueError(f"Missing panel comparisons in task manifests: {sorted(missing)}")
    return (
        pd.DataFrame(moment_rows)
        .set_index("family_test_id", drop=False)
        .loc[panel.index]
        .reset_index(drop=True)
    )
