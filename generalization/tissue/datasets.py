"""Healthy-tissue Figure 3 datasets and generated-input locations."""

from pathlib import Path

from generalization.datasets import CHRONOLOGICAL_COMPONENTS
from generalization.modeling import DatasetSpec


def _spec(gse: str, title: str) -> DatasetSpec:
    root = Path("data/generalization/tissue") / gse
    return DatasetSpec(
        gse,
        title,
        "tissue",
        root / "intermediate" / "beta_age.parquet",
        root / "prepared" / "component_predictions_age.parquet",
        CHRONOLOGICAL_COMPONENTS,
    )


DATASETS = {
    "GSE175458": _spec("GSE175458", "Lung"),
    "GSE232891": _spec("GSE232891", "Saliva"),
    "GSE287331": _spec("GSE287331", "Breast"),
}
DEFAULT_RESULTS = Path("results/generalization/tissue")
