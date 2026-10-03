#!/usr/bin/env python3
"""Prepare GSE223748 inputs and train saved species submodel artifacts."""

from __future__ import annotations

import argparse

from .prepare_species_gse223748_inputs import prepare_inputs
from .species_config import POOLED_COMPONENT, SPECIES
from .species_submodels import train_elasticnetcv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Prepare inputs before training")
    parser.add_argument("--model-type", choices=("elasticnetcv",), default="elasticnetcv")
    parser.add_argument("--source", choices=(*SPECIES, POOLED_COMPONENT))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.prepare:
        prepare_inputs()
    sources = (args.source,) if args.source else (*SPECIES, POOLED_COMPONENT)
    for source in sources:
        train_elasticnetcv(source, n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
