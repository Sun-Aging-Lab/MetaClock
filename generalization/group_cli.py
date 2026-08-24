"""Shared command-line plumbing for array and tissue reproduction scripts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from repository_paths import repo_path

from .feature_preparation import prepare_dataset_inputs
from .geo_data import DEFAULT_DATA_ROOT, download_dataset, select_datasets
from .modeling import DatasetSpec, GeneralizationConfig, evaluate_training_size
from .pyaging_components import predict_dataset_components

DEFAULT_CACHE = DEFAULT_DATA_ROOT / "pyaging_model_cache"


def _selection(parser: argparse.ArgumentParser, domain: str) -> None:
    choices = [spec.gse for spec in select_datasets(domain, None)]
    parser.add_argument(
        "--gse",
        action="append",
        choices=choices,
        help=f"GEO accession to process; repeat as needed (default: all {domain} datasets)",
    )
    parser.add_argument("--data-root", type=repo_path, default=DEFAULT_DATA_ROOT)


def download_main(domain: str) -> int:
    parser = argparse.ArgumentParser(
        description=f"Download and normalize Figure 3 {domain} GEO data"
    )
    _selection(parser, domain)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-normalize", action="store_true")
    args = parser.parse_args()
    from .geo_data import build_beta_age

    for spec in select_datasets(domain, args.gse):
        download_dataset(spec, args.data_root, force=args.force_download)
        print(build_beta_age(spec, args.data_root, force=args.force_normalize))
    return 0


def prediction_main(domain: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run 16 PyAging clocks for Figure 3 {domain}")
    _selection(parser, domain)
    parser.add_argument("--cache-dir", type=repo_path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for spec in select_datasets(domain, args.gse):
        print(
            predict_dataset_components(
                spec,
                args.data_root,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                force=args.force,
            )
        )
    return 0


def preparation_main(domain: str) -> int:
    parser = argparse.ArgumentParser(description=f"Prepare Figure 3 {domain} training inputs")
    _selection(parser, domain)
    parser.add_argument("--cache-dir", type=repo_path, default=DEFAULT_CACHE)
    parser.add_argument("--pyaging-batch-size", type=int, default=32)
    parser.add_argument("--mi-features", type=int, default=10_000)
    parser.add_argument("--mrmr-features", type=int, default=1_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for spec in select_datasets(domain, args.gse):
        direct, meta = prepare_dataset_inputs(
            spec,
            args.data_root,
            cache_dir=args.cache_dir,
            mi_features=args.mi_features,
            mrmr_features=args.mrmr_features,
            pyaging_batch_size=args.pyaging_batch_size,
            force=args.force,
        )
        print(direct)
        print(meta)
    return 0


def training_main(
    specs: Mapping[str, DatasetSpec],
    default_output: Path,
    description: str,
    *,
    fixed_dataset: str | None = None,
) -> int:
    config = GeneralizationConfig()
    parser = argparse.ArgumentParser(description=description)
    if fixed_dataset is None:
        parser.add_argument("--gse", choices=specs, required=True)
    parser.add_argument(
        "--size-index", type=int, choices=range(config.n_training_sizes), default=None
    )
    parser.add_argument("--direct-path", type=repo_path)
    parser.add_argument("--meta-path", type=repo_path)
    parser.add_argument("--output-root", type=repo_path, default=default_output)
    args = parser.parse_args()
    dataset = fixed_dataset or args.gse
    size_index = args.size_index
    if size_index is None:
        import os

        size_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    spec = specs[dataset]
    if args.direct_path is not None:
        spec = replace(spec, direct_path=args.direct_path)
    if args.meta_path is not None:
        spec = replace(spec, meta_path=args.meta_path)
    print(evaluate_training_size(spec, size_index, args.output_root, config=config))
    return 0
