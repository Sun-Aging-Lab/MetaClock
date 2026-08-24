#!/usr/bin/env python3
"""Parallel train/test entry point fixed to GSE175458 lung data."""

from generalization.group_cli import training_main
from generalization.tissue.datasets import DATASETS, DEFAULT_RESULTS

if __name__ == "__main__":
    raise SystemExit(
        training_main(
            DATASETS,
            DEFAULT_RESULTS,
            "Train/test GSE175458 lung",
            fixed_dataset="GSE175458",
        )
    )
