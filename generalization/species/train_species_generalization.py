"""Train one species point using direct CpGs and ten MLP-ELU meta inputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from generalization.modeling import (
    DatasetSpec,
    GeneralizationConfig,
    evaluate_training_size,
)
from generalization.species.species_config import (
    DISPLAY_NAMES,
    POOLED_COMPONENT,
    SPECIES,
    direct_path,
    mlp_loo_path,
)

DEFAULT_OUTPUT = Path("results/figure3/generalization/species/shards")


def expected_components(target: str) -> tuple[str, ...]:
    """Return nine held-out species MLP-ELUs plus the pooled MLP-ELU."""
    return tuple(
        [f"{species}_mlp_elu" for species in SPECIES if species != target]
        + [f"{POOLED_COMPONENT}_mlp_elu"]
    )


def species_spec(species: str) -> DatasetSpec:
    """Build the strict portable dataset declaration for one species."""
    return DatasetSpec(
        species,
        DISPLAY_NAMES[species],
        direct_path(species),
        mlp_loo_path(species),
        expected_components(species),
    )


def main() -> int:
    config = GeneralizationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=SPECIES, required=True)
    parser.add_argument(
        "--size-index", type=int, choices=range(config.n_training_sizes), required=True
    )
    parser.add_argument("--direct-path", type=Path)
    parser.add_argument("--meta-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    spec = species_spec(args.species)
    if args.direct_path is not None:
        spec = replace(spec, direct_path=args.direct_path)
    if args.meta_path is not None:
        spec = replace(spec, meta_path=args.meta_path)
    print(evaluate_training_size(spec, args.size_index, args.output_root, config=config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
