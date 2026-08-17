"""Load and validate the manifest shared by biological analyses and enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("analysis_manifest.json")


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the versioned manifest and enforce its global exclusion invariant."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported analysis manifest schema.")
    excluded = {str(value).lower() for value in manifest["globally_excluded_models"]}
    if excluded != {"dnamfitage"}:
        raise ValueError("The manifest must globally exclude exactly dnamfitage.")
    serialized = json.dumps(manifest).lower()
    # The exclusion declaration itself is the only permitted occurrence.
    if serialized.count("dnamfitage") != 1:
        raise ValueError("dnamfitage appears outside the global exclusion declaration.")
    enrichment = manifest["enrichment"]
    for prefix, expected_candidates, expected_affine in (
        ("chronological", 16, 8),
        ("biological", 19, 17),
    ):
        candidates = tuple(enrichment[f"{prefix}_candidates"])
        affine = tuple(enrichment[f"{prefix}_affine_components"])
        non_affine = tuple(enrichment[f"{prefix}_non_affine_components"])
        if len(candidates) != expected_candidates or len(set(candidates)) != len(candidates):
            raise ValueError(
                f"{prefix} candidates must contain {expected_candidates} unique clocks."
            )
        if len(affine) != expected_affine or len(set(affine)) != len(affine):
            raise ValueError(f"{prefix} affine components must contain {expected_affine} clocks.")
        if set(affine).intersection(non_affine) or set(affine).union(non_affine) != set(candidates):
            raise ValueError(f"{prefix} affine/non-affine manifests must partition candidates.")
    return manifest


MANIFEST = load_manifest()
BIOLOGICAL_ANALYSIS = MANIFEST["biological_analysis"]
ENRICHMENT = MANIFEST["enrichment"]
EXCLUDED_MODELS = frozenset(MANIFEST["globally_excluded_models"])
EXTERNAL_COMPONENTS = tuple(BIOLOGICAL_ANALYSIS["external_component_features"])
EXTERNAL_MODELS = tuple(BIOLOGICAL_ANALYSIS["external_models"])


def require_exact_sequence(observed: list[str], expected: tuple[str, ...], label: str) -> None:
    """Require an exact, ordered feature/model sequence."""
    if observed != list(expected):
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(
            f"{label} does not match the analysis manifest; missing={missing}, extra={extra}, "
            f"order_matches={set(observed) == set(expected)}."
        )
