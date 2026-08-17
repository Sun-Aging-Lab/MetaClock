#!/usr/bin/env python3
"""Apply the published 672-participant BioMetaClock weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXTERNAL_COMPONENTS, require_exact_sequence

WEIGHT_DIR = Path(__file__).with_name("model_weights")
DEFAULT_WEIGHTS = WEIGHT_DIR / "biometaclock_672_participants_weights.csv"
DEFAULT_METADATA = WEIGHT_DIR / "biometaclock_672_participants_metadata.json"
DEFAULT_INPUT = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT = Path("results/biological_age/biometaclock_predictions.csv")


def load_frozen_weights(
    weights_path: Path = DEFAULT_WEIGHTS,
    metadata_path: Path = DEFAULT_METADATA,
) -> tuple[pd.DataFrame, float]:
    """Load and validate the ordered raw-scale coefficient representation."""
    weights = pd.read_csv(weights_path)
    required = {
        "feature",
        "scaler_mean",
        "scaler_scale",
        "standardized_coefficient",
        "raw_coefficient",
    }
    if set(weights.columns) != required or weights["feature"].duplicated().any():
        raise ValueError(
            "The BioMetaClock weight table has an invalid schema or duplicate feature."
        )
    require_exact_sequence(
        weights["feature"].astype(str).tolist(),
        EXTERNAL_COMPONENTS,
        "published BioMetaClock weights",
    )
    numeric = weights[list(required - {"feature"})].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all() or (numeric["scaler_scale"] <= 0).any():
        raise ValueError("The BioMetaClock weight table contains invalid numeric values.")
    expected_raw = numeric["standardized_coefficient"] / numeric["scaler_scale"]
    if not np.allclose(expected_raw, numeric["raw_coefficient"], rtol=1e-12, atol=1e-14):
        raise ValueError(
            "Raw BioMetaClock coefficients disagree with scaler-adjusted coefficients."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("training_participants") != 672
        or metadata.get("feature_count") != len(weights)
    ):
        raise ValueError("BioMetaClock metadata does not describe the published 672-person model.")
    intercept = float(metadata["raw_intercept"])
    if not np.isfinite(intercept):
        raise ValueError("BioMetaClock raw intercept is not finite.")
    return weights, intercept


def predict_biometaclock(
    components: pd.DataFrame,
    weights_path: Path = DEFAULT_WEIGHTS,
    metadata_path: Path = DEFAULT_METADATA,
) -> np.ndarray:
    """Predict biological age from the exact ordered 17 component-clock columns."""
    weights, intercept = load_frozen_weights(weights_path, metadata_path)
    features = weights["feature"].astype(str).tolist()
    if missing := set(features).difference(components.columns):
        raise KeyError(f"Component table is missing BioMetaClock features: {sorted(missing)}")
    design = components[features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(design).all():
        raise ValueError("Component predictions contain non-finite values.")
    return intercept + design @ weights["raw_coefficient"].to_numpy(float)


def main() -> int:
    """Read component predictions and write one BioMetaClock prediction per row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    components = pd.read_csv(args.components)
    output = components.drop(columns=list(EXTERNAL_COMPONENTS)).copy()
    output["biometaclock_prediction"] = predict_biometaclock(components)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
