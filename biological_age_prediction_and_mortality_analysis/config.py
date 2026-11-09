"""Load and validate the manifest shared by biological analyses and enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("analysis_manifest.json")


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read and validate the versioned analysis manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported analysis manifest schema.")
    biological = manifest["biological_analysis"]
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
    oof_components = tuple(biological["oof_component_features"])
    external_components = tuple(biological["external_component_features"])
    if oof_components != tuple(enrichment["biological_candidates"]):
        raise ValueError("OOF component features must match the 19 biological candidates.")
    expected_external = tuple(
        component for component in oof_components if component not in {"grimage", "grimage2"}
    )
    if external_components != expected_external:
        raise ValueError("External components must be the ordered OOF list without GrimAge1/2.")
    dnam = biological["inchianti_dnam_features"]
    prefix_counts = dnam["prefix_counts"]
    if int(dnam["total"]) != sum(int(value) for value in prefix_counts.values()):
        raise ValueError("InCHIANTI DNAm feature prefix counts do not sum to the total.")
    fold_assignment = biological["production_oof"]
    if int(fold_assignment["baseline_features"]) != int(dnam["total"]):
        raise ValueError("The biological-age baseline must use every InCHIANTI DNAm feature.")
    counts = {int(fold): int(count) for fold, count in fold_assignment["fold_counts"].items()}
    if (
        int(fold_assignment["rows"]) != int(biological["inchianti_participants"])
        or int(fold_assignment["folds"]) != 5
        or set(counts) != set(range(5))
        or any(count <= 0 for count in counts.values())
        or sum(counts.values()) != int(fold_assignment["rows"])
    ):
        raise ValueError("The primary-participant OOF fold manifest metadata is inconsistent.")
    return manifest


MANIFEST = load_manifest()
BIOLOGICAL_ANALYSIS = MANIFEST["biological_analysis"]
ENRICHMENT = MANIFEST["enrichment"]
OOF_COMPONENTS = tuple(BIOLOGICAL_ANALYSIS["oof_component_features"])
EXTERNAL_COMPONENTS = tuple(BIOLOGICAL_ANALYSIS["external_component_features"])
EXTERNAL_MODELS = tuple(BIOLOGICAL_ANALYSIS["external_models"])
INCHIANTI_DNAM_FEATURE_COUNT = int(BIOLOGICAL_ANALYSIS["inchianti_dnam_features"]["total"])
DNAM_FEATURE_PREFIX_COUNTS = {
    str(prefix): int(count)
    for prefix, count in BIOLOGICAL_ANALYSIS["inchianti_dnam_features"]["prefix_counts"].items()
}
DNAM_FEATURE_PREFIXES = tuple(DNAM_FEATURE_PREFIX_COUNTS)
INCHIANTI_COMPLETE_CASE_SAMPLES = int(BIOLOGICAL_ANALYSIS["inchianti_complete_case_samples"])
INCHIANTI_PARTICIPANTS = int(BIOLOGICAL_ANALYSIS["inchianti_participants"])
INCHIANTI_DUPLICATE_PAIRS = int(BIOLOGICAL_ANALYSIS["inchianti_duplicate_pairs"])
INCHIANTI_COX_EVENTS = int(BIOLOGICAL_ANALYSIS["inchianti_cox_events"])
PRODUCTION_OOF = BIOLOGICAL_ANALYSIS["production_oof"]
INCHIANTI_OOF_FOLDS = int(PRODUCTION_OOF["folds"])
PRODUCTION_CONFIG_ID = str(PRODUCTION_OOF["config_id"])
PRODUCTION_OOF_FOLD_ASSIGNMENT_PATH = MANIFEST_PATH.parent / str(PRODUCTION_OOF["path"])


def require_exact_sequence(observed: list[str], expected: tuple[str, ...], label: str) -> None:
    """Require an exact, ordered feature/model sequence."""
    if observed != list(expected):
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(
            f"{label} does not match the analysis manifest; missing={missing}, extra={extra}, "
            f"order_matches={set(observed) == set(expected)}."
        )
