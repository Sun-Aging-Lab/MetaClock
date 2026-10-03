#!/usr/bin/env python3
"""Build leakage-free ENCV features from saved component pipelines."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .species_config import (
    POOLED_COMPONENT,
    SPECIES,
    direct_input_path,
    loo_feature_path,
)
from .species_submodels import COMPONENT_CONFIG_ID, predict_elasticnetcv

TARGET = "age"


def build_loo_features(target_species: str) -> pd.DataFrame:
    """Apply only non-self source models to one GSE-derived target species."""
    direct = pd.read_parquet(direct_input_path(target_species))
    direct.index = direct.index.astype(str)
    external = [species for species in SPECIES if species != target_species]
    output = pd.DataFrame({TARGET: direct[TARGET].to_numpy(dtype=float)}, index=direct.index)
    for source in [*external, POOLED_COMPONENT]:
        output[f"{source}_elasticnetcv"] = predict_elasticnetcv(direct, source)
    if output.isna().any(axis=None) or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite LOO features for {target_species}")
    output.attrs["component_config_id"] = COMPONENT_CONFIG_ID
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=SPECIES, required=True)
    args = parser.parse_args()
    output = build_loo_features(args.species)
    path = loo_feature_path(args.species)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, compression="zstd")
    path.with_suffix(".provenance.json").write_text(
        json.dumps(
            {"target_species": args.species,
             "component_config_id": COMPONENT_CONFIG_ID,
             "elasticnetcv_sources": [*[_ for _ in SPECIES if _ != args.species], POOLED_COMPONENT],
             "evolution_sources": [_ for _ in SPECIES if _ != args.species],
             "target_component_excluded": True, "samples": len(output)},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
