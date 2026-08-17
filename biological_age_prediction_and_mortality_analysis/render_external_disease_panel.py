#!/usr/bin/env python3
"""Render the unrestricted external-disease panel from its source table."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot_external_disease_panel import plot_external_disease_heatmap
from .plotting import configure_style

DEFAULT_EXTERNAL_TESTS = Path(
    "results/figure4/external_disease/03_external_disease_three_model_all_datasets.csv"
)
DEFAULT_OUTPUT = Path("results/figure4/figures")


def parse_args() -> argparse.Namespace:
    """Parse validated source-data and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-tests", type=Path, default=DEFAULT_EXTERNAL_TESTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Render the external-disease heatmap with its formal dimensions."""
    args = parse_args()
    configure_style()
    plot_external_disease_heatmap(args.external_tests, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
