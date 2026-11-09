"""Validated primary-participant predictions shared by Figure 4 analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRODUCTION_CONFIG_ID, PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH
from .data import _canonical_participant_ids, sort_figure4_samples
from .provenance import software_versions

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "biological_age_prediction"
MODEL_NAMES = ("bio_metaclock", "baseline")
PREDICTION_COLUMNS = (
    "ID", "sampleID", "age_years", "sex_male", "clinical_phenoage", "fold", "model", "pred_bioage"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_assignment() -> pd.DataFrame:
    assignment = pd.read_csv(PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH, dtype={"ID": str, "sampleID": str})
    assignment["ID"] = _canonical_participant_ids(assignment["ID"])
    if (set(assignment.columns) != {"ID", "sampleID", "fold"}
            or len(assignment) != 672 or assignment.ID.duplicated().any()
            or assignment.sampleID.duplicated().any()
            or not assignment.fold.isin(range(5)).all()
            or assignment.fold.value_counts().sort_index().tolist() != [135, 135, 134, 134, 134]):
        raise ValueError("Invalid primary-participant OOF assignment.")
    return assignment


def production_oof_folds(frame: pd.DataFrame) -> np.ndarray:
    assignment = production_assignment()
    ids = _canonical_participant_ids(frame.ID)
    if set(ids) != set(assignment.ID):
        raise ValueError("Figure 4 participants do not match the production OOF assignment.")
    return ids.map(assignment.set_index("ID").fold).to_numpy(int)


def primary_samples(frame: pd.DataFrame) -> pd.DataFrame:
    primary = sort_figure4_samples(frame).drop_duplicates("ID", keep="first").reset_index(drop=True)
    expected = production_assignment().set_index("ID").sampleID
    if set(primary.ID) != set(expected.index) or not primary.sampleID.eq(primary.ID.map(expected)).all():
        raise ValueError("Primary sample IDs differ from the Figure 4 cohort.")
    return primary


def _validate_predictions(frame: pd.DataFrame, expected: pd.DataFrame, models: tuple[str, ...], *, kind: str) -> pd.DataFrame:
    if missing := set(PREDICTION_COLUMNS).difference(frame.columns):
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    result = frame.copy()
    result["ID"] = _canonical_participant_ids(result.ID)
    result["sampleID"] = result.sampleID.astype(str)
    expected = expected.copy()
    expected["ID"] = _canonical_participant_ids(expected.ID)
    expected["sampleID"] = expected.sampleID.astype(str)
    if expected.sampleID.duplicated().any():
        raise ValueError("Expected sample IDs are not unique.")
    if set(result.model) != set(models) or not set(models).issubset(MODEL_NAMES):
        raise ValueError("Unexpected model coverage in the prediction table.")
    wanted = {(sample, model) for sample in expected.sampleID for model in models}
    if result.duplicated(["sampleID", "model"]).any() or set(zip(result.sampleID, result.model)) != wanted:
        raise ValueError("Predictions do not cover each expected sample and model exactly once.")
    expected_by_sample = expected.set_index("sampleID")
    if not result.ID.eq(result.sampleID.map(expected_by_sample.ID)).all():
        raise ValueError("Predictions have inconsistent participant/sample IDs.")
    numeric = ["age_years", "sex_male", "clinical_phenoage", "pred_bioage"]
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("Predictions or clinical covariates are nonfinite.")
    for column in numeric[:-1]:
        if column in expected:
            reference = result.sampleID.map(expected_by_sample[column]).to_numpy(float)
            if not np.allclose(result[column].to_numpy(float), reference, rtol=1e-12, atol=1e-10):
                raise ValueError(f"Prediction {column} differs from the analysis cohort.")
    if kind == "oof":
        by_id = production_assignment().set_index("ID").fold
        if not result.fold.eq(result.ID.map(by_id)).all():
            raise ValueError("Predictions use a different OOF assignment.")
    elif not result.fold.eq(-1).all():
        raise ValueError("Duplicate holdout predictions must use the common holdout marker -1.")
    return result


def _save_predictions(predictions: pd.DataFrame, output_dir: Path, kind: str, expected: pd.DataFrame) -> None:
    models = tuple(model for model in MODEL_NAMES if model in set(predictions.model))
    frame = _validate_predictions(predictions, expected, models, kind=kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "oof" if kind == "oof" else "duplicate_holdout"
    path = output_dir / f"{stem}_predictions.csv"
    frame.to_csv(path, index=False)
    manifest = {
        "config_id": PRODUCTION_CONFIG_ID,
        "assignment_sha256": file_sha256(PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH),
        "predictions_sha256": file_sha256(path),
        "models": list(models),
        "samples_per_model": len(expected),
        "prediction_kind": kind,
        "software": software_versions(["numpy", "pandas", "scikit-learn", "joblib"]),
    }
    (output_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def save_oof_predictions(predictions: pd.DataFrame, *, output_dir: Path | None = None) -> None:
    _save_predictions(predictions, output_dir or DEFAULT_OUTPUT_DIR, "oof", production_assignment())


def save_duplicate_predictions(predictions: pd.DataFrame, valid: pd.DataFrame, *, output_dir: Path | None = None) -> None:
    counts = valid.groupby("ID").sampleID.transform("size")
    expected = valid.loc[counts.eq(2)].copy()
    if len(expected) != 88 or expected.ID.nunique() != 44:
        raise ValueError("Duplicate holdout requires 88 profiles from 44 participants.")
    _save_predictions(predictions, output_dir or DEFAULT_OUTPUT_DIR, "duplicate", expected)


def _load_predictions(output_dir: Path, kind: str, expected: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    stem = "oof" if kind == "oof" else "duplicate_holdout"
    path, manifest_path = output_dir / f"{stem}_predictions.csv", output_dir / f"{stem}_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run biological-age prediction before the downstream Figure 4 analysis.")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("config_id") != PRODUCTION_CONFIG_ID
            or manifest.get("assignment_sha256") != file_sha256(PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH)
            or manifest.get("predictions_sha256") != file_sha256(path)
            or manifest.get("prediction_kind") != kind
            or not set(models).issubset(manifest.get("models", []))):
        raise ValueError("Prediction configuration, content checksum, or model coverage differs.")
    frame = pd.read_csv(path, dtype={"ID": str, "sampleID": str}, float_precision="round_trip")
    frame = frame.loc[frame.model.isin(models)].copy()
    return _validate_predictions(frame, expected, models, kind=kind)


def load_primary_oof_predictions(
    cohort: pd.DataFrame, output_dir: Path | None = None,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> pd.DataFrame:
    primary = primary_samples(cohort)
    predictions = _load_predictions(output_dir or DEFAULT_OUTPUT_DIR, "oof", primary, model_names)
    score = predictions[["ID", "sampleID", "fold", "model", "pred_bioage"]]
    primary = primary.drop(columns=["fold", "model", "pred_bioage"], errors="ignore")
    return primary.merge(score, on=["ID", "sampleID"], how="inner", validate="one_to_many", sort=False)


def load_duplicate_predictions(
    valid: pd.DataFrame, output_dir: Path | None = None,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> pd.DataFrame:
    counts = valid.groupby("ID").sampleID.transform("size")
    expected = valid.loc[counts.eq(2)].copy()
    if len(expected) != 88 or expected.ID.nunique() != 44:
        raise ValueError("Duplicate holdout requires 88 profiles from 44 participants.")
    return _load_predictions(output_dir or DEFAULT_OUTPUT_DIR, "duplicate", expected, model_names)
