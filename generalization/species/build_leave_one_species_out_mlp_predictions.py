"""Apply trained MLP-ELU components to make ten-column LOO meta inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from generalization.modeling import atomic_write_parquet, project_path
from generalization.species.species_config import (
    MLP_LOO_ROOT,
    POOLED_COMPONENT,
    SPECIES,
    direct_path,
)
from generalization.species.train_species_mlp_elu_components import MLPClock, MLPConfig

TARGET = "age"


def predict_component(frame: pd.DataFrame, model_dir: Path, device: str) -> pd.Series:
    """Align CpGs, apply saved preprocessing, and predict one component clock."""
    bundle = joblib.load(model_dir / "preprocessing.joblib")
    features = bundle["features"]
    values = bundle["scaler"].transform(
        bundle["imputer"].transform(frame.reindex(columns=features))
    )
    model = MLPClock(len(features), MLPConfig(**bundle["config"])).to(device)
    model.load_state_dict(
        torch.load(model_dir / "weights.pt", map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        prediction = (
            model(torch.as_tensor(values, dtype=torch.float32, device=device)).cpu().numpy()
        )
    return pd.Series(prediction, index=frame.index)


def build_loo_predictions(
    target_species: str,
    component_root: Path,
    output_root: Path,
    *,
    device: str,
) -> Path:
    """Generate nine held-out species components plus the pooled component."""
    component_root = project_path(component_root)
    output_root = project_path(output_root)
    frame = pd.read_parquet(project_path(direct_path(target_species))).dropna(subset=[TARGET])
    frame.index = frame.index.astype(str)
    components = [species for species in SPECIES if species != target_species] + [POOLED_COMPONENT]
    result = pd.DataFrame(index=frame.index)
    for component in components:
        result[f"{component}_mlp_elu"] = predict_component(
            frame, component_root / component, device
        )
    result[TARGET] = frame[TARGET].to_numpy(dtype=float)
    output = output_root / f"loo_per_species_predictions_{target_species}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {f"{component}_mlp_elu" for component in components} | {TARGET}
    if (
        set(result.columns) != expected
        or not np.isfinite(result.drop(columns=TARGET).to_numpy(float)).all()
    ):
        raise ValueError("LOO prediction table is incomplete or non-finite")
    atomic_write_parquet(result, output, index=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=SPECIES, required=True)
    parser.add_argument(
        "--component-root", type=Path, default=Path("results/figure3/species/mlp_elu_components")
    )
    parser.add_argument("--output-root", type=Path, default=MLP_LOO_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        build_loo_predictions(
            args.species, args.component_root, args.output_root, device=args.device
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
