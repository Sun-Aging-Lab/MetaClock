"""Synthetic checks for formal scatter color and grid invariants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chronological_age_prediction_performance.plot_chronological_age_prediction_scatter import (
    read_complete_metrics,
)
from chronological_age_prediction_performance.plotting import COMPONENT_COLORS, META_COLOR


class ChronologicalPlottingTests(unittest.TestCase):
    """Protect color separation and complete metrics-grid validation."""

    def test_component_color_never_reuses_metaclock_color(self) -> None:
        self.assertTrue(
            all(
                color != META_COLOR
                for label, color in COMPONENT_COLORS.items()
                if label != "MetaClock"
            )
        )

    def test_metrics_grid_rejects_missing_dataset_label_cell(self) -> None:
        table = pd.DataFrame(
            {
                "dataset": ["GSE147221"],
                "model": ["ElasticNet"],
                "mae": [1.0],
                "pearson_r": [0.9],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
            path = Path(temporary.name)
        self.addCleanup(path.unlink)
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "dataset × model"):
            read_complete_metrics(path, label_column="model", labels=("ElasticNet",))
