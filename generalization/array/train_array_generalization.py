#!/usr/bin/env python3
"""Fit/test one array dataset and one parallel training-size point."""

from generalization.array.datasets import DATASETS, DEFAULT_RESULTS
from generalization.group_cli import training_main

if __name__ == "__main__":
    raise SystemExit(
        training_main(DATASETS, DEFAULT_RESULTS, "Train/test one Figure 3 array point")
    )
