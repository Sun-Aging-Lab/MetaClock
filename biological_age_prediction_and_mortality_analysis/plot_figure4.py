"""Render the prediction, reliability, external disease and external mortality panels."""

from __future__ import annotations

import argparse

import pandas as pd

from . import run_biological_age_prediction as prediction
from . import run_cox_analysis as cox
from . import run_duplicate_sample_reliability as reliability
from . import run_kaplan_meier_analysis as km
from .figure4_external_disease import render_figure4_external_disease


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    km.add_input_arguments(parser)
    args = parser.parse_args(argv)
    prediction.plot_metrics(pd.read_csv(prediction.OUTPUT_DIR / "oof_metrics.csv"))
    reliability.plot_reliability(pd.read_csv(reliability.OUTPUT_DIR / "duplicate_sample_reliability.csv"))
    render_figure4_external_disease()
    cox.main()
    km.render_or_use(args.private_curve_dir, args.private_cohort_dir)


if __name__ == "__main__":
    main()
