"""Render the prediction, reliability, external disease and external mortality panels."""

import pandas as pd

from . import run_biological_age_prediction as prediction
from . import run_cox_analysis as cox
from . import run_duplicate_sample_reliability as reliability
from . import run_kaplan_meier_analysis as km
from .figure4_external_disease import render_figure4_external_disease


def main() -> None:
    prediction.plot_metrics(pd.read_csv(prediction.OUTPUT_DIR / "oof_metrics.csv"))
    reliability.plot_reliability(pd.read_csv(reliability.OUTPUT_DIR / "duplicate_sample_reliability.csv"))
    render_figure4_external_disease()
    cox.main()
    km.main()


if __name__ == "__main__":
    main()
