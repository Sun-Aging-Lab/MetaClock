#!/usr/bin/env python3
"""Select Figure 3 species from GSE223748 and write model-ready beta tables."""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repository_paths import relative_path

from .species_config import (
    EXPECTED_SAMPLE_COUNTS,
    GESTATION_TABLE,
    GSE223748_BETA_PATH,
    GSE223748_SERIES_PATH,
    INPUT_ROOT,
    INPUT_CONFIG_ID,
    LATIN_TO_MODEL,
    POOLED_COMPONENT,
    SPECIES,
    direct_input_path,
    metadata_path,
)

ARRAY_ID = re.compile(r"(\d+_R\d+C\d+)")
TARGET = "age"
GESTATION_DECIMALS = 3


def parse_geo_metadata(path: Path) -> pd.DataFrame:
    """Parse GSE223748 sample fields without relying on metadata row order."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing downloaded input: {path}. "
            "Run python -m generalization.species.fetch_gse223748 first."
        )
    csv.field_size_limit(sys.maxsize)
    base: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    wanted = {
        "!Sample_geo_accession": "gsm",
        "!Sample_title": "title",
        "!Sample_source_name_ch1": "tissue",
        "!Sample_organism_ch1": "organism",
    }
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0] == "!series_matrix_table_begin":
                if row and row[0] == "!series_matrix_table_begin":
                    break
                continue
            if row[0] in wanted:
                base[wanted[row[0]]] = row[1:]
            elif row[0] == "!Sample_characteristics_ch1":
                characteristics.append(row[1:])
    if set(base) != set(wanted.values()):
        raise ValueError(f"Missing GEO metadata fields: {sorted(set(wanted.values()) - set(base))}")
    lengths = {len(values) for values in base.values()}
    if lengths != {15043}:
        raise ValueError(f"Expected 15,043 GSE223748 samples; observed field lengths {lengths}")
    records: list[dict[str, str]] = []
    for index in range(15043):
        record = {key: values[index] for key, values in base.items()}
        for values in characteristics:
            value = values[index].strip()
            if ":" in value:
                key, item = value.split(":", 1)
                if key.strip():
                    record[key.strip().lower()] = item.strip()
        records.append(record)
    metadata = pd.DataFrame(records)
    metadata["model_species"] = metadata["organism"].map(LATIN_TO_MODEL)
    metadata["array_id"] = metadata["title"].str.extract(ARRAY_ID, expand=False)
    metadata["age_years"] = pd.to_numeric(metadata["age"], errors="coerce")
    metadata["maximum_lifespan_years"] = pd.to_numeric(
        metadata["maximumlifespanyears"], errors="coerce"
    )
    if not metadata["gsm"].is_unique or not metadata["array_id"].is_unique:
        raise ValueError("GSE223748 GSM and array IDs must be unique")
    return metadata


def add_transformed_age(metadata: pd.DataFrame) -> pd.DataFrame:
    """Apply the Figure 3 gestation-adjusted relative-age transformation."""
    life_history = pd.read_csv(GESTATION_TABLE)
    if list(life_history.columns) != ["species", "gestation_years"]:
        raise ValueError(f"Unexpected gestation table schema: {GESTATION_TABLE}")
    if life_history["species"].duplicated().any():
        raise ValueError("Gestation table contains duplicated species")
    gestation = pd.to_numeric(
        life_history.set_index("species")["gestation_years"], errors="raise"
    )
    frame = metadata.copy()
    frame["gestation_years"] = frame["organism"].map(gestation)
    rounded_species = (set(LATIN_TO_MODEL) - {"Canis lupus familiaris"}) | {"Delphinapterus leucas"}
    rounded = frame["organism"].isin(rounded_species)
    frame.loc[rounded, "gestation_years"] = frame.loc[rounded, "gestation_years"].round(GESTATION_DECIMALS)
    frame["relative_age"] = (
        (frame["age_years"] + frame["gestation_years"])
        / (frame["maximum_lifespan_years"] + frame["gestation_years"])
    )
    valid = (
        frame["age_years"].notna()
        & frame["maximum_lifespan_years"].notna()
        & frame["gestation_years"].notna()
        & frame["array_id"].notna()
        & frame["relative_age"].gt(0)
        & frame["relative_age"].lt(1)
    )
    frame["exclusion_reason"] = np.where(valid, "", "missing_life_history_or_invalid_age")
    frame.loc[valid, TARGET] = -np.log(-np.log(frame.loc[valid, "relative_age"]))
    return frame


def read_beta_matrix(path: Path, sample_ids: set[str]) -> pd.DataFrame:
    """Read selected GSE223748 columns from the deposited normalized beta matrix."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing downloaded input: {path}. "
            "Run python -m generalization.species.fetch_gse223748 first."
        )
    with gzip.open(path, "rt", newline="") as handle:
        header = next(csv.reader(handle))
    probe_column = "Unnamed: 0" if header[0] == "" else header[0]
    missing = sample_ids.difference(header[1:])
    if missing:
        raise ValueError(f"GSE223748 beta matrix is missing {len(missing)} selected array IDs")
    selected = {probe_column, *sample_ids}
    matrix = pd.read_csv(
        path,
        compression="gzip",
        usecols=lambda column: column in selected,
        dtype={sample_id: np.float32 for sample_id in sample_ids},
        low_memory=False,
    ).set_index(probe_column)
    matrix.index = matrix.index.astype(str)
    if matrix.index.has_duplicates or matrix.columns.duplicated().any():
        raise ValueError("GSE223748 beta matrix has duplicated probes or sample columns")
    return matrix


def sample_by_probe(matrix: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Align matrix columns to GSM records and prepend the transformed target."""
    arrays = records["array_id"].tolist()
    output = matrix.loc[:, arrays].T
    array_to_gsm = records.set_index("array_id")["gsm"]
    output.index = output.index.map(array_to_gsm)
    output.index.name = "sample_id"
    age = records.set_index("gsm")[TARGET]
    output.insert(0, TARGET, age.loc[output.index].to_numpy(dtype=float))
    return output


def prepare_inputs() -> None:
    """Build CpG inputs for ten targets and one leakage-free pooled ENCV clock."""
    metadata = add_transformed_age(parse_geo_metadata(GSE223748_SERIES_PATH))
    valid = metadata["exclusion_reason"].eq("")
    targets = metadata.loc[valid & metadata["model_species"].notna()].copy()
    observed = targets["model_species"].value_counts().to_dict()
    if observed != EXPECTED_SAMPLE_COUNTS:
        raise ValueError(f"Unexpected selected target-species counts: {observed}")
    pooled = metadata.loc[valid & metadata["model_species"].isna()].copy()
    if pooled.empty:
        raise ValueError("No non-target GSE223748 samples are eligible for alltissueconcat")
    records = {
        species: targets.loc[targets["model_species"].eq(species)].copy()
        for species in SPECIES
    }
    records[POOLED_COMPONENT] = pooled
    matrix = read_beta_matrix(
        GSE223748_BETA_PATH,
        {array_id for table in records.values() for array_id in table["array_id"]},
    )
    cpgs = [probe for probe in matrix.index if probe.lower().startswith("cg")]
    if not cpgs:
        raise ValueError("GSE223748 beta matrix has no CpG probes named cg...")
    summaries: list[dict[str, object]] = []
    for source, source_records in records.items():
        full = sample_by_probe(matrix, source_records)
        full.attrs["input_config_id"] = INPUT_CONFIG_ID
        direct_path = direct_input_path(source)
        direct_path.parent.mkdir(parents=True, exist_ok=True)
        full.loc[:, [TARGET, *cpgs]].to_parquet(direct_path, compression="zstd")
        meta_path = metadata_path(source)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        source_records.to_csv(meta_path, sep="\t", index=False)
        summaries.append(
            {
                "source": source,
                "samples": len(full),
                "component_model_type": "elasticnetcv",
                "direct_cpg_features": len(cpgs),
            }
        )
    excluded = metadata.loc[metadata["model_species"].notna() & ~valid]
    excluded.to_csv(INPUT_ROOT / "excluded_target_species_samples.tsv", sep="\t", index=False)
    (INPUT_ROOT / "input_provenance.json").write_text(
        json.dumps(
            {
                "dataset": "GSE223748",
                "input_config_id": INPUT_CONFIG_ID,
                "beta_matrix": relative_path(GSE223748_BETA_PATH),
                "series_metadata": relative_path(GSE223748_SERIES_PATH),
                "target_transform": "-log(-log((age + gestation)/(maximum_lifespan + gestation)))",
                "gestation_precision": {
                    "rounded_decimals": GESTATION_DECIMALS,
                    "rounded_species": sorted((set(LATIN_TO_MODEL) - {"Canis lupus familiaris"}) | {"Delphinapterus leucas"}),
                    "other_species": "full table precision",
                },
                "pooled_rule": "valid non-target species only; all ten target species excluded",
                "sources": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    prepare_inputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
