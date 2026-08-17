"""Structural smoke tests for the formal intervention volcano vector output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from intervention_analysis.plot_intervention_fdr_volcano import (
    FIGURE_MM,
    OUTLINE_WIDTH_PT,
    configure_style,
    plot_volcano,
)
from intervention_analysis.validate_intervention_volcano import validate_pdf, validate_svg


class InterventionVolcanoTests(unittest.TestCase):
    """Check geometry, typography, and editable vector text without source data."""

    def test_formal_style_constants_are_applied(self) -> None:
        configure_style()
        import matplotlib.pyplot as plt

        self.assertEqual(OUTLINE_WIDTH_PT, 0.2)
        self.assertEqual(plt.rcParams["axes.linewidth"], OUTLINE_WIDTH_PT)
        self.assertEqual(plt.rcParams["xtick.major.width"], OUTLINE_WIDTH_PT)
        self.assertEqual(plt.rcParams["ytick.major.width"], OUTLINE_WIDTH_PT)
        self.assertEqual(plt.rcParams["font.family"], ["Liberation Sans"])

    def test_vector_smoke_has_formal_canvas_styles_and_text(self) -> None:
        rows = [
            ("curated_1fc35799c70a1f70", "Abdominal radiotherapy", 0.01, 2.0),
            ("curated_fe95adeead1b848b", "Dexamethasone", 0.10, 1.0),
            ("curated_a5aa77c6d7ffdccc", "Cigarette smoking", 0.15, 0.5),
            ("curated_6ac8603ac6e44a0e", "Cytarabine (AraC)", 0.20, 2.5),
            ("curated_1fa2b9d81d6ea894", "FAE", 0.05, 4.0),
            ("not_labelled", "", 0.80, 1.5),
        ]
        frame = pd.DataFrame(
            {
                "comparison_id": [row[0] for row in rows],
                "short_label": [row[1] for row in rows],
                "bh_q_value_after_dataset_exclusion": [row[2] for row in rows],
                "minus_log10_fdr": [-np.log10(row[2]) for row in rows],
                "fdr_category": [
                    "FDR ≤ 5%" if row[2] <= 0.05 else
                    "5% < FDR ≤ 20%" if row[2] <= 0.20 else "FDR > 20%"
                    for row in rows
                ],
                "mean_diff_experiment_minus_control": [row[3] for row in rows],
                "dx": [-5, -5, -5, 5, -5, 0],
                "dy": [-1, -1, 16, 8, 8, 0],
                "ha": ["right", "right", "right", "left", "right", "left"],
                "va": ["center", "center", "bottom", "bottom", "bottom", "center"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "intervention_volcano"
            pdf_path, svg_path = plot_volcano(frame, stem)
            self.assertTrue(pdf_path.is_file())
            self.assertTrue(svg_path.is_file())

            import pymupdf

            with pymupdf.open(pdf_path) as document:
                self.assertEqual(len(document), 1)
                page = document[0]
                observed_mm = (page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72)
                for observed, expected in zip(observed_mm, FIGURE_MM, strict=True):
                    self.assertAlmostEqual(observed, expected, places=2)
                text = page.get_text()
                for label in frame.loc[frame["short_label"].ne(""), "short_label"]:
                    self.assertIn(label, text)
                self.assertIn("Δ MetaClock Prediction", text)
                self.assertIn("-log10(FDR)", text)
                self.assertNotIn("-1.0", text)
                fonts = {font[3].replace("-", "") for font in page.get_fonts()}
                self.assertTrue(any("LiberationSans" in font for font in fonts), fonts)

            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("font-family: 'Liberation Sans'", svg_text)
            self.assertIn("stroke-width: 0.2", svg_text)
            self.assertIn("Δ MetaClock Prediction", svg_text)
            self.assertIn("-log10(FDR)", svg_text)
            labels = set(frame.loc[frame["short_label"].ne(""), "short_label"])
            self.assertEqual(validate_pdf(pdf_path, labels)["outline_width_pt"], 0.2)
            self.assertEqual(validate_svg(svg_path, labels)["outline_width_pt"], 0.2)


if __name__ == "__main__":
    unittest.main()
