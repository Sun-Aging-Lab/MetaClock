#!/usr/bin/env python3
"""Parallel train/test entry point fixed to GSE234461 EPIC v1 data."""

from generalization.array.datasets import DATASETS, DEFAULT_RESULTS
from generalization.group_cli import training_main

if __name__ == "__main__":
    raise SystemExit(
        training_main(
            DATASETS,
            DEFAULT_RESULTS,
            "Train/test GSE234461 MethylChipEPIC v1",
            fixed_dataset="GSE234461",
        )
    )
