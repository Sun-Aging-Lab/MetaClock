#!/usr/bin/env python3
"""Fit/test one tissue dataset and one parallel training-size point."""

from generalization.group_cli import training_main
from generalization.tissue.datasets import DATASETS, DEFAULT_RESULTS

if __name__ == "__main__":
    raise SystemExit(
        training_main(DATASETS, DEFAULT_RESULTS, "Train/test one Figure 3 tissue point")
    )
