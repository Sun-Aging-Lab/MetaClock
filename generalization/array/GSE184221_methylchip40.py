#!/usr/bin/env python3
"""Parallel train/test entry point fixed to GSE184221 MethylChip40 data."""

from generalization.array.datasets import DATASETS, DEFAULT_RESULTS
from generalization.group_cli import training_main

if __name__ == "__main__":
    raise SystemExit(
        training_main(
            DATASETS,
            DEFAULT_RESULTS,
            "Train/test GSE184221 MethylChip40",
            fixed_dataset="GSE184221",
        )
    )
