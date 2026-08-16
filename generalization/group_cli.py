"""Shared thin CLI implementations for array and tissue entry points."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from .aggregation import aggregate_dataset
from .modeling import DatasetSpec, GeneralizationConfig, evaluate_training_size


def training_main(specs: Mapping[str, DatasetSpec], default_output: Path, description: str) -> int:
    """Parse and run one group-specific point-training command."""
    config = GeneralizationConfig()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=specs, required=True)
    parser.add_argument(
        "--size-index", type=int, choices=range(config.n_training_sizes), required=True
    )
    parser.add_argument("--direct-path", type=Path)
    parser.add_argument("--meta-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=default_output)
    args = parser.parse_args()
    spec = specs[args.dataset]
    if args.direct_path is not None:
        spec = replace(spec, direct_path=args.direct_path)
    if args.meta_path is not None:
        spec = replace(spec, meta_path=args.meta_path)
    print(evaluate_training_size(spec, args.size_index, args.output_root, config=config))
    return 0


def aggregation_main(
    specs: Mapping[str, DatasetSpec],
    default_shards: Path,
    default_summaries: Path,
    description: str,
) -> int:
    """Parse and run a group-specific strict aggregation command."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=specs, action="append")
    parser.add_argument("--shard-root", type=Path, default=default_shards)
    parser.add_argument("--summary-root", type=Path, default=default_summaries)
    args = parser.parse_args()
    selected = args.dataset or list(specs)
    for dataset in selected:
        print(aggregate_dataset(specs[dataset], args.shard_root, args.summary_root))
    return 0
