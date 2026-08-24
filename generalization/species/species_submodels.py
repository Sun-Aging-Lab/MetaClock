"""Train, save, load, and apply Figure 3 species component models."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .species_config import direct_input_path, mlp_input_path, model_dir

TARGET = "age"
SEED = 105


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, int] = (100, 50)
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = SEED


class MLPClock(nn.Module):
    """Two-hidden-layer BatchNorm/ELU component clock."""

    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        first, second = config.hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.BatchNorm1d(first),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(first, second),
            nn.BatchNorm1d(second),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(second, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_batch_size(rows: int, preferred: int) -> int:
    size = min(rows, preferred)
    while size > 2 and rows % size == 1:
        size -= 1
    return size


def train_mlp_elu(source: str, *, device: str) -> Path:
    """Train one source MLP-ELU clock and save weights plus preprocessing."""
    frame = pd.read_parquet(mlp_input_path(source)).dropna(subset=[TARGET])
    features = [str(column) for column in frame if column != TARGET]
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    values = scaler.fit_transform(imputer.fit_transform(frame[features])).astype(np.float32)
    target = frame[TARGET].to_numpy(dtype=np.float32)
    config = MLPConfig()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(target)),
        batch_size=safe_batch_size(len(frame), config.batch_size),
        shuffle=True,
    )
    set_seed(config.seed)
    model = MLPClock(len(features), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=30, threshold=1e-2
    )
    loss_function = nn.MSELoss()
    history: list[float] = []
    for epoch in range(config.epochs):
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
        if not np.isfinite(epoch_loss):
            raise RuntimeError(f"Non-finite MLP loss for {source} at epoch {epoch + 1}")
        history.append(epoch_loss)
        scheduler.step(epoch_loss)
    output = model_dir("mlp_elu", source)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output / "weights.pt",
    )
    joblib.dump(
        {"features": features, "imputer": imputer, "scaler": scaler, "config": asdict(config)},
        output / "preprocessing.joblib",
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {"source": source, "model_type": "mlp_elu", "samples": len(frame),
             "features": len(features),
             "final_training_mse": history[-1], "parameters": asdict(config),
             "runtime": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__, "torch": torch.__version__}},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def train_elasticnetcv(source: str, *, n_jobs: int) -> Path:
    """Train one source ElasticNetCV clock and save its complete pipeline."""
    frame = pd.read_parquet(direct_input_path(source)).dropna(subset=[TARGET])
    features = [str(column) for column in frame if str(column).lower().startswith("cg")]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(cv=5, alphas=10, max_iter=10_000,
                                   random_state=SEED, n_jobs=n_jobs)),
        ]
    )
    model.fit(frame[features], frame[TARGET].to_numpy(dtype=float))
    output = model_dir("elasticnetcv", source)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"features": features, "model": model},
        output / "model.joblib",
    )
    fitted = model.named_steps["model"]
    (output / "training_summary.json").write_text(
        json.dumps(
            {"source": source, "model_type": "elasticnetcv", "samples": len(frame),
             "features": len(features),
             "selected_alpha": float(fitted.alpha_), "cv_folds": 5, "alpha_candidates": 10,
             "max_iter": 10_000, "seed": SEED,
             "runtime": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__}},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def _strict_values(frame: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise ValueError(f"{label} input is missing {len(missing)} trained features")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} input contains infinite values")
    return values


def predict_mlp_elu(frame: pd.DataFrame, source: str, *, device: str) -> np.ndarray:
    """Apply one saved MLP-ELU artifact with strict feature identity."""
    root = model_dir("mlp_elu", source)
    bundle = joblib.load(root / "preprocessing.joblib")
    features = list(bundle["features"])
    values = bundle["scaler"].transform(bundle["imputer"].transform(
        _strict_values(frame, features, f"MLP {source}")
    ))
    config = MLPConfig(**bundle["config"])
    model = MLPClock(len(features), config).to(device)
    model.load_state_dict(torch.load(root / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(values, dtype=torch.float32, device=device)).cpu().numpy()


def predict_elasticnetcv(frame: pd.DataFrame, source: str) -> np.ndarray:
    """Apply one saved ElasticNetCV component pipeline."""
    bundle = joblib.load(model_dir("elasticnetcv", source) / "model.joblib")
    features = list(bundle["features"])
    return np.asarray(
        bundle["model"].predict(_strict_values(frame, features, f"ENCV {source}")),
        dtype=float,
    )
