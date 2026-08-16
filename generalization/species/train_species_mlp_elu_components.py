"""Train the ten species and one pooled MLP-ELU component clocks."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generalization.modeling import atomic_write_json, project_path
from generalization.species.species_config import (
    POOLED_COMPONENT,
    SPECIES,
    component_training_path,
)

TARGET = "age"


@dataclass(frozen=True)
class MLPConfig:
    """Recovered MLP-ELU component architecture and optimizer settings."""

    hidden_dims: tuple[int, int] = (100, 50)
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 105


class MLPClock(nn.Module):
    """Two-hidden-layer ELU network used for each species component clock."""

    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        h1, h2 = config.hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set NumPy, Python, and Torch seeds for reproducible component training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_batch_size(n_rows: int, preferred: int) -> int:
    """Avoid a final one-row batch because BatchNorm cannot train on it."""
    if n_rows < 2:
        raise ValueError("MLP-ELU training requires at least two samples for BatchNorm")
    size = min(n_rows, preferred)
    while size > 2 and n_rows % size == 1:
        size -= 1
    return size


def train_component(
    component: str,
    input_path: Path,
    output_root: Path,
    config: MLPConfig,
    *,
    device: str,
) -> Path:
    """Fit one component and save weights, preprocessing, and provenance."""
    input_path = project_path(input_path)
    output_root = project_path(output_root)
    frame = pd.read_parquet(input_path).dropna(subset=[TARGET])
    features = [
        column
        for column in frame.columns
        if str(column).startswith("cg") and not frame[column].isna().all()
    ]
    if not features:
        raise ValueError(f"{input_path}: no CpG columns")
    target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError("MLP target contains non-finite values")
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    scaler = StandardScaler()
    values = scaler.fit_transform(imputer.fit_transform(frame[features])).astype(np.float32)
    if values.shape[1] != len(features):
        raise RuntimeError("Imputation changed the MLP input dimension")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(target)),
        batch_size=safe_batch_size(len(frame), config.batch_size),
        shuffle=True,
    )

    set_seed(config.seed)
    model = MLPClock(values.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=30, threshold=1e-2
    )
    loss_function = nn.MSELoss()
    loss_history: list[float] = []
    for _ in range(config.epochs):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
        epoch_loss = total / len(frame)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)

    output_dir = output_root / component
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.pt"
    weights_temporary = output_dir / ".weights.pt.tmp"
    torch.save(model.state_dict(), weights_temporary)
    weights_temporary.replace(weights_path)
    preprocessing_path = output_dir / "preprocessing.joblib"
    preprocessing_temporary = output_dir / ".preprocessing.joblib.tmp"
    joblib.dump(
        {"features": features, "imputer": imputer, "scaler": scaler, "config": asdict(config)},
        preprocessing_temporary,
    )
    preprocessing_temporary.replace(preprocessing_path)
    atomic_write_json(
        {
            "component": component,
            "input": str(input_path),
            "samples": len(frame),
            "features": len(features),
            "final_training_mse": loss_history[-1],
            "architecture": "StandardScaler + MLP(100, 50; ELU; BatchNorm; dropout=0.5)",
            "parameters": asdict(config),
        },
        output_dir / "training_summary.json",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=(*SPECIES, POOLED_COMPONENT), required=True)
    parser.add_argument("--input", type=Path, help="Override the configured relative input")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/figure3/species/mlp_elu_components")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = train_component(
        args.component,
        args.input or component_training_path(args.component),
        args.output_root,
        MLPConfig(),
        device=args.device,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
