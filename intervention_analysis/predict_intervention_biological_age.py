#!/usr/bin/env python3
"""Apply a frozen biological MetaClock to intervention component predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "results/intervention/model/biometaclock.joblib"  # fill in your path
DEFAULT_FEATURES = (
    PROJECT_ROOT / "results/intervention/model/biometaclock_features.csv"
)  # fill in your path
DEFAULT_COMPONENT_PREDICTIONS = (
    PROJECT_ROOT / "datasets/intervention/component_predictions.csv"
)  # fill in your path
DEFAULT_SAMPLE_MANIFEST = (
    PROJECT_ROOT / "datasets/intervention/sample_manifest.csv"
)  # fill in your path
DEFAULT_OUTPUT = PROJECT_ROOT / "results/intervention/sample_predictions.csv"
KEY = "sample_id"


def load_inputs(
    component_path: Path,
    manifest_path: Path,
    feature_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Join sample metadata to the ordered, complete component-clock matrix."""
    features = pd.read_csv(feature_path)["feature"].astype(str).str.lower().tolist()
    if "dnamfitage" in features:
        raise ValueError("DNAmFitAge is globally excluded from biological MetaClock inputs.")
    components = pd.read_csv(component_path)
    manifest = pd.read_csv(manifest_path)
    required_manifest = {KEY, "comparison_id", "group", "donor_id"}
    missing = sorted(required_manifest.difference(manifest.columns))
    if missing:
        raise KeyError(f"Sample manifest is missing columns: {missing}")
    missing_features = sorted({KEY, *features}.difference(components.columns))
    if missing_features:
        raise KeyError(f"Component table is missing columns: {missing_features}")
    if manifest[KEY].duplicated().any() or components[KEY].duplicated().any():
        raise ValueError("sample_id must be unique in both input tables.")
    joined = manifest.merge(components[[KEY, *features]], on=KEY, how="left", validate="one_to_one")
    matrix = joined[features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Every sample must have finite predictions for every component clock.")
    return joined, features


def predict(model_path: Path, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Append the biological-age prediction while preserving manifest columns."""
    model = joblib.load(model_path)
    if int(getattr(model, "n_features_in_", len(features))) != len(features):
        raise ValueError("Model feature count does not match the feature manifest.")
    output = frame.drop(columns=features).copy()
    output["predicted_age"] = np.asarray(
        model.predict(frame[features].to_numpy(float)), dtype=float
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--component-predictions", type=Path, default=DEFAULT_COMPONENT_PREDICTIONS)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, features = load_inputs(args.component_predictions, args.sample_manifest, args.features)
    output = predict(args.model, frame, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
