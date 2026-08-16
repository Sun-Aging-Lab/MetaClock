#!/usr/bin/env python3
"""Render all biological-age and mortality panels from validated source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .plot_external_disease_panel import plot_external_disease_heatmap
    from .plot_mortality_panels import plot_kaplan_meier_survival, plot_standardized_cox_effects
    from .plot_prediction_reliability_panels import (
        plot_duplicate_reliability,
        plot_prediction_metrics,
    )
    from .plotting import configure_style
except ImportError:  # Direct-script fallback.
    from plot_external_disease_panel import plot_external_disease_heatmap
    from plot_mortality_panels import plot_kaplan_meier_survival, plot_standardized_cox_effects
    from plot_prediction_reliability_panels import (
        plot_duplicate_reliability,
        plot_prediction_metrics,
    )
    from plotting import configure_style

DEFAULT_SOURCE = Path("results/figure4/source_data")  # fill in your path
DEFAULT_EXTERNAL_TESTS = Path(
    "results/figure4/external_disease/external_disease_tests.csv"
)  # fill in your path
DEFAULT_OUTPUT = Path("results/figure4/figures")


def parse_args() -> argparse.Namespace:
    """Parse validated source-data and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--external-tests", type=Path, default=DEFAULT_EXTERNAL_TESTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Render all panels in formal dimensions with semantic filenames."""
    args = parse_args()
    configure_style()
    plot_prediction_metrics(args.source_dir, args.output_dir)
    plot_duplicate_reliability(args.source_dir, args.output_dir)
    plot_external_disease_heatmap(args.external_tests, args.output_dir)
    plot_standardized_cox_effects(args.source_dir, args.output_dir)
    plot_kaplan_meier_survival(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
