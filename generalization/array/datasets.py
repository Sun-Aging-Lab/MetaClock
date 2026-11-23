"""Array-platform Figure 3 datasets and generated-input locations."""

from pathlib import Path

from generalization.datasets import CHRONOLOGICAL_COMPONENTS
from generalization.modeling import DatasetSpec


def _spec(gse: str, title: str) -> DatasetSpec:
    root = Path("data/generalization/array") / gse
    return DatasetSpec(
        gse,
        title,
        "array",
        root / "intermediate" / "beta_age.parquet",
        root / "prepared" / "component_predictions_age.parquet",
        CHRONOLOGICAL_COMPONENTS,
    )


DATASETS = {
    "GSE184221": _spec("GSE184221", "MethylChip40"),
    "GSE234461": _spec("GSE234461", "MethylChipEPIC v1.0"),
}
DEFAULT_RESULTS = Path("results/generalization/array")
