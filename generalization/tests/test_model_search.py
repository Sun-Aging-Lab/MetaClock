from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from generalization.estimators import ElasticNetConfig, GeneralizationConfig, TrainingCpGs, fit_model


class FoldLocalSearchTests(TestCase):
    def test_mi_fits_only_inner_training_rows_and_full_draw(self) -> None:
        rng = np.random.default_rng(105)
        matrix = rng.random((23, 8), dtype=np.float32)
        matrix[6:, 0] = np.nan
        matrix[:18, 1] = np.nan
        rows = pd.DataFrame({"row": np.arange(len(matrix)), "age": rng.normal(size=len(matrix))})
        config = GeneralizationConfig(
            elastic_net=ElasticNetConfig(alphas=(0.1, 0.01), n_jobs=1),
            mi_features=6, svr_c_values=(1.0, 0.01), svr_epsilon_values=(1.0, 0.001),
        )
        original_fit = TrainingCpGs.fit
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cpgs.npy"
            np.save(path, matrix)
            for model, folds in (("EN", 5), ("SVM", 3)):
                events = {}

                def record_fit(selector: TrainingCpGs, X: np.ndarray, y: np.ndarray) -> TrainingCpGs:
                    original_fit(selector, X, y)
                    indices = tuple(X[:, 0])
                    self.assertNotIn(indices, events)
                    events[indices] = selector.selected_indices_
                    return selector

                with self.subTest(model=model), patch.object(TrainingCpGs, "fit", record_fit):
                    fit_model(model, rows.iloc[:18], rows.iloc[18:], config, matrix_path=path)
                    expected = {tuple(i) for i, _ in KFold(folds).split(rows.iloc[:18])}
                    self.assertEqual(set(events), expected | {tuple(range(18))})
                    for indices, selected in events.items():
                        valid = ~np.isnan(matrix[list(indices)]).all(axis=0)
                        self.assertTrue(valid[selected].all())
                        self.assertEqual(len(selected), config.mi_features)

    def test_constant_age_accepts_zero_support_vectors(self) -> None:
        data = pd.DataFrame({"clock": np.arange(18), "age": 5.0})
        config = GeneralizationConfig(elastic_net=ElasticNetConfig(n_jobs=1),
                                      svr_c_values=(1.0,), svr_epsilon_values=(0.001, 0.1))
        result = fit_model("Meta SVM", data.iloc[:12], data.iloc[12:], config, domain="species")
        self.assertEqual(result["support_vectors"], 0)
        self.assertEqual(result["mae"], 0.0)
