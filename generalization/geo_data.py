"""Download and normalize the five public GEO inputs used in Figure 3.

The statistical workflow is shared, but GEO's processed matrices and sample
annotations are not standardized.  This module keeps those dataset-specific
adapters explicit and validates the published cohort counts before emitting a
common samples-by-CpGs ``beta_age.parquet`` contract.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "generalization"
ARRAY_ID_PATTERN = re.compile(r"(\d+_R\d+C\d+)")
THRIFT_CONTAINER_LIMIT = 512 * 1024 * 1024
THRIFT_STRING_LIMIT = 256 * 1024 * 1024


@dataclass(frozen=True)
class RemoteFile:
    """One immutable GEO source file."""

    filename: str
    url: str


@dataclass(frozen=True)
class GeoDatasetSpec:
    """Dataset-specific public inputs and expected normalized cohort."""

    domain: str
    gse: str
    title: str
    platform: str
    files: tuple[RemoteFile, ...]
    beta_filenames: tuple[str, ...]
    expected_geo_samples: int
    expected_selected_samples: int


def _geo_url(gse: str, location: str, filename: str) -> str:
    prefix = f"{gse[:6]}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/{location}/{filename}"


def _remote(gse: str, location: str, filename: str) -> RemoteFile:
    return RemoteFile(filename, _geo_url(gse, location, filename))


DATASETS: dict[str, GeoDatasetSpec] = {
    "GSE184221": GeoDatasetSpec(
        "array",
        "GSE184221",
        "MethylChip40",
        "HorvathMammalMethylChip40",
        (
            _remote("GSE184221", "matrix", "GSE184221_series_matrix.txt.gz"),
            _remote("GSE184221", "suppl", "GSE184221_datBetaNormalized.csv.gz"),
            _remote(
                "GSE184221", "suppl", "GSE184221_n240_datBetaNormalized.csv.gz"
            ),
        ),
        (
            "GSE184221_datBetaNormalized.csv.gz",
            "GSE184221_n240_datBetaNormalized.csv.gz",
        ),
        384,
        384,
    ),
    "GSE234461": GeoDatasetSpec(
        "array",
        "GSE234461",
        "MethylChipEPIC v1.0",
        "Illumina HumanMethylationEPIC",
        (
            _remote("GSE234461", "matrix", "GSE234461_series_matrix.txt.gz"),
            _remote("GSE234461", "suppl", "GSE234461_Matrix_processed.csv.gz"),
        ),
        ("GSE234461_Matrix_processed.csv.gz",),
        245,
        245,
    ),
    "GSE175458": GeoDatasetSpec(
        "tissue",
        "GSE175458",
        "Lung",
        "Illumina MethylationEPIC",
        (
            _remote("GSE175458", "matrix", "GSE175458_series_matrix.txt.gz"),
            _remote(
                "GSE175458", "suppl", "GSE175458_FinalBetas_sysbio_547samps.csv.gz"
            ),
        ),
        ("GSE175458_FinalBetas_sysbio_547samps.csv.gz",),
        547,
        142,
    ),
    "GSE232891": GeoDatasetSpec(
        "tissue",
        "GSE232891",
        "Saliva",
        "Illumina MethylationEPIC",
        (
            _remote("GSE232891", "matrix", "GSE232891_series_matrix.txt.gz"),
            _remote(
                "GSE232891", "suppl", "GSE232891_Processed_Beta_Values.csv.gz"
            ),
        ),
        ("GSE232891_Processed_Beta_Values.csv.gz",),
        552,
        248,
    ),
    "GSE287331": GeoDatasetSpec(
        "tissue",
        "GSE287331",
        "Breast",
        "Illumina MethylationEPIC v1.0",
        (
            _remote("GSE287331", "matrix", "GSE287331_series_matrix.txt.gz"),
            _remote("GSE287331", "suppl", "GSE287331_betas_processed.csv.gz"),
            _remote("GSE287331", "suppl", "GSE287331_meta.csv.gz"),
        ),
        ("GSE287331_betas_processed.csv.gz",),
        446,
        182,
    ),
}


def datasets_for_domain(domain: str) -> tuple[GeoDatasetSpec, ...]:
    """Return the stable manuscript order for a domain."""
    if domain not in {"array", "tissue"}:
        raise ValueError(f"Unknown human generalization domain: {domain}")
    return tuple(spec for spec in DATASETS.values() if spec.domain == domain)


def select_datasets(domain: str, requested: Iterable[str] | None) -> tuple[GeoDatasetSpec, ...]:
    """Resolve an optional list of accessions, defaulting to the whole domain."""
    available = {spec.gse: spec for spec in datasets_for_domain(domain)}
    names = list(dict.fromkeys(requested or available))
    invalid = sorted(set(names) - set(available))
    if invalid:
        raise ValueError(
            f"Invalid {domain} accessions {invalid}; choose from {sorted(available)}"
        )
    return tuple(available[name] for name in names)


def dataset_dir(data_root: Path, spec: GeoDatasetSpec) -> Path:
    """Return the deterministic generated-data directory for one dataset."""
    return data_root.expanduser().resolve() / spec.domain / spec.gse


def atomic_json(payload: dict[str, object], output: Path) -> None:
    """Write JSON atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    """Write a wide Parquet atomically while preserving its sample index."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.stem}.", suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def read_wide_parquet(path: Path) -> pd.DataFrame:
    """Read very wide methylation Parquets with explicit Arrow metadata limits."""
    import pyarrow.parquet as pq

    return pq.read_table(
        path,
        thrift_container_size_limit=THRIFT_CONTAINER_LIMIT,
        thrift_string_size_limit=THRIFT_STRING_LIMIT,
    ).to_pandas()


def download_file(source: RemoteFile, output: Path, *, force: bool = False) -> Path:
    """Download one GEO file atomically, resuming a server-supported partial file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.part")
    if output.is_file() and not force:
        return output
    if force:
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(
        source.url,
        headers={
            "User-Agent": "metaclock-reproduction/0.1",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed HTTPS GEO URLs
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    os.replace(partial, output)
    return output


def download_dataset(spec: GeoDatasetSpec, data_root: Path, *, force: bool = False) -> Path:
    """Fetch every raw file required by one public GEO adapter."""
    root = dataset_dir(data_root, spec)
    raw_dir = root / "raw"
    rows: list[dict[str, object]] = []
    for source in spec.files:
        output = download_file(source, raw_dir / source.filename, force=force)
        rows.append(
            {
                "filename": source.filename,
                "url": source.url,
                "bytes": output.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(root / "download_manifest.tsv", sep="\t", index=False)
    atomic_json(
        {
            "dataset": spec.gse,
            "domain": spec.domain,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "files": rows,
        },
        root / "download_provenance.json",
    )
    return raw_dir


def parse_series_metadata(path: Path) -> pd.DataFrame:
    """Parse named GEO sample fields without relying on characteristic row order."""
    fields: dict[str, list[str]] = {}
    characteristics: dict[str, list[str]] = {}
    field_names = {
        "!Sample_geo_accession": "gsm",
        "!Sample_title": "title",
        "!Sample_source_name_ch1": "source_name_ch1",
        "!Sample_description": "description",
    }
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if row[0] == "!series_matrix_table_begin":
                break
            if row[0] in field_names:
                fields[field_names[row[0]]] = row[1:]
            elif row[0] == "!Sample_characteristics_ch1":
                split_values = [value.split(":", 1) for value in row[1:]]
                if any(len(parts) != 2 for parts in split_values):
                    raise ValueError(f"Malformed GEO characteristic row in {path}")
                keys = {parts[0].strip().lower() for parts in split_values}
                if len(keys) != 1:
                    raise ValueError(f"Inconsistent GEO characteristic row in {path}")
                key = next(iter(keys))
                if key in characteristics:
                    raise ValueError(f"Duplicate GEO characteristic {key!r} in {path}")
                characteristics[key] = [parts[1].strip() for parts in split_values]
    required = {"gsm", "title", "source_name_ch1"}
    if not required.issubset(fields):
        raise ValueError(f"Missing GEO fields in {path}: {sorted(required - set(fields))}")
    lengths = {len(values) for values in [*fields.values(), *characteristics.values()]}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent GEO sample-field lengths in {path}: {lengths}")
    metadata = pd.DataFrame({**fields, **characteristics})
    if not metadata["gsm"].is_unique:
        raise ValueError(f"Duplicate GSM identifiers in {path}")
    return metadata


def _series_path(raw_dir: Path, spec: GeoDatasetSpec) -> Path:
    return raw_dir / f"{spec.gse}_series_matrix.txt.gz"


def _select_samples(
    spec: GeoDatasetSpec, raw_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Apply the audited Figure 3 cohort definition and return its beta sample key."""
    metadata = parse_series_metadata(_series_path(raw_dir, spec))
    if len(metadata) != spec.expected_geo_samples:
        raise ValueError(
            f"{spec.gse}: expected {spec.expected_geo_samples} GEO samples, "
            f"found {len(metadata)}"
        )

    if spec.gse in {"GSE184221", "GSE234461"}:
        metadata["array_id"] = metadata["title"].str.extract(ARRAY_ID_PATTERN, expand=False)
        metadata["age_numeric"] = pd.to_numeric(metadata["age"], errors="raise")
        if metadata[["array_id", "age_numeric"]].isna().any().any():
            raise ValueError(f"{spec.gse}: missing array ID or age")
        if not metadata["array_id"].is_unique:
            raise ValueError(f"{spec.gse}: duplicate array IDs")
        selected = metadata.copy()
        excluded = metadata.iloc[0:0].copy()
        excluded["exclusion_reason"] = pd.Series(dtype="string")
        sample_key = "array_id"
    elif spec.gse == "GSE175458":
        metadata["age_numeric"] = pd.to_numeric(metadata["age"], errors="coerce")
        status = metadata["case_control_status"].str.strip().str.casefold()
        diagnosis = metadata["dx"].str.strip().str.casefold()
        keep = status.eq("control") & diagnosis.eq("normal") & metadata["age_numeric"].ge(18)
        selected = metadata.loc[keep].copy()
        excluded = metadata.loc[~keep].copy()
        excluded["exclusion_reason"] = np.select(
            [
                status.loc[excluded.index].eq("case"),
                diagnosis.loc[excluded.index].eq("other"),
                diagnosis.loc[excluded.index].eq("control"),
                metadata.loc[excluded.index, "age_numeric"].lt(18),
            ],
            ["ipf_case", "control_dx_other", "control_dx_control", "minor_age_lt18"],
            default="not_selected",
        )
        sample_key = "methylationid"
    elif spec.gse == "GSE232891":
        metadata["age_numeric"] = pd.to_numeric(metadata["age"], errors="coerce")
        source = metadata["source_name_ch1"].str.upper()
        disease = metadata["disease"].str.casefold()
        control = source.str.fullmatch(r"CONTROL\d{3}") & disease.str.startswith(
            "control sample"
        )
        keep = control & metadata["age_numeric"].ge(18)
        selected = metadata.loc[keep].copy()
        excluded = metadata.loc[~keep].copy()
        excluded["exclusion_reason"] = np.where(
            ~control.loc[excluded.index],
            "crohns_disease_or_ulcerative_colitis_case",
            "control_age_below_18",
        )
        sample_key = "source_name_ch1"
    elif spec.gse == "GSE287331":
        if "description" not in metadata:
            raise ValueError("GSE287331: Series Matrix lacks the IDAT description field")
        metadata = metadata.rename(columns={"description": "IDAT"})
        supplement = pd.read_csv(raw_dir / "GSE287331_meta.csv.gz", compression="gzip")
        supplement["Age"] = pd.to_numeric(supplement["Age"], errors="raise")
        if len(supplement) != 471 or not supplement["IDAT"].is_unique:
            raise ValueError("GSE287331: unexpected supplementary metadata")
        metadata = metadata.merge(supplement, on="IDAT", how="left", validate="one_to_one")
        if metadata[["ID", "Age"]].isna().any().any():
            raise ValueError("GSE287331: post-QC samples lack supplementary metadata")
        keep = (
            metadata["title"].str.startswith("HDB_")
            & metadata["Sample Region"].eq("HDB")
            & metadata["Case/Control"].eq("normal")
            & metadata["Age"].ge(18)
        )
        metadata["age_numeric"] = metadata["Age"].astype(float)
        selected = metadata.loc[keep].copy()
        excluded = metadata.loc[~keep].copy()
        excluded["exclusion_reason"] = (
            "cancer_case_associated_" + excluded["Sample Region"].str.lower()
        )
        sample_key = "IDAT"
    else:  # pragma: no cover - registry and dispatch are tested together
        raise ValueError(f"No cohort adapter for {spec.gse}")

    if len(selected) != spec.expected_selected_samples:
        raise ValueError(
            f"{spec.gse}: expected {spec.expected_selected_samples} selected samples, "
            f"found {len(selected)}"
        )
    if selected[["gsm", sample_key, "age_numeric"]].isna().any().any():
        raise ValueError(f"{spec.gse}: selected samples have missing identifiers or ages")
    if not selected["gsm"].is_unique or not selected[sample_key].is_unique:
        raise ValueError(f"{spec.gse}: selected sample identifiers are not unique")
    return metadata, selected, excluded, sample_key


def read_beta_matrix(path: Path, expected_sample_ids: set[str]) -> pd.DataFrame:
    """Read CpG beta columns only, excluding paired detection-p-value columns."""
    probe_names = {"ID_REF", "CGid", "Probe", "probe", "Unnamed: 0", ""}

    def clean(column: str) -> str:
        value = str(column)
        return value[1:] if value.startswith("X") and value[1:] in expected_sample_ids else value

    def keep(column: str) -> bool:
        return str(column) in probe_names or clean(column) in expected_sample_ids

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        header = next(csv.reader(handle))
    numeric = [column for column in header if clean(column) in expected_sample_ids]
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=keep,
        dtype={column: np.float32 for column in numeric},
        na_values=["NA", "NaN", "nan"],
        low_memory=False,
    )
    probe_column = next((column for column in frame if str(column) in probe_names), None)
    if probe_column is None:
        raise ValueError(f"No probe identifier column found in {path}")
    frame = frame.set_index(probe_column)
    frame.index = frame.index.astype(str)
    frame = frame.loc[frame.index.str.startswith("cg")]
    frame.columns = [clean(column) for column in frame.columns]
    if frame.index.has_duplicates or frame.columns.duplicated().any():
        raise ValueError(f"Duplicate probes or sample columns in {path}")
    return frame


def build_beta_age(spec: GeoDatasetSpec, data_root: Path, *, force: bool = False) -> Path:
    """Normalize one downloaded GEO cohort to samples × CpGs plus chronological age."""
    root = dataset_dir(data_root, spec)
    raw_dir = root / "raw"
    output = root / "intermediate" / "beta_age.parquet"
    if output.is_file() and not force:
        return output
    missing = [
        source.filename for source in spec.files if not (raw_dir / source.filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{spec.gse}: download the missing raw files before preparation: {missing}"
        )

    metadata, selected, excluded, sample_key = _select_samples(spec, raw_dir)
    expected_ids = set(selected[sample_key].astype(str))
    matrices = [
        read_beta_matrix(raw_dir / filename, expected_ids) for filename in spec.beta_filenames
    ]
    probe_index = matrices[0].index
    if any(not matrix.index.equals(probe_index) for matrix in matrices[1:]):
        raise ValueError(f"{spec.gse}: beta batches have different probe order")
    matrix = pd.concat(matrices, axis=1)
    if set(matrix.columns) != expected_ids:
        missing_ids = sorted(expected_ids - set(matrix.columns))[:10]
        extra_ids = sorted(set(matrix.columns) - expected_ids)[:10]
        raise ValueError(
            f"{spec.gse}: beta/metadata mismatch; missing={missing_ids}, extra={extra_ids}"
        )

    key_to_gsm = selected.set_index(sample_key)["gsm"]
    beta = matrix.T
    beta.index = beta.index.map(key_to_gsm)
    beta.index.name = "sample_id"
    age_by_gsm = selected.set_index("gsm")["age_numeric"]
    beta.insert(0, "age", age_by_gsm.loc[beta.index].to_numpy(dtype=float))
    values = beta.drop(columns="age").to_numpy(dtype=np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() < -0.01 or finite.max() > 1.01:
        raise ValueError(f"{spec.gse}: processed values are not beta-like")
    if not beta.index.is_unique or not np.isfinite(beta["age"]).all():
        raise ValueError(f"{spec.gse}: normalized sample IDs or ages are invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(beta, output)
    metadata.to_csv(root / "metadata_all.tsv", sep="\t", index=False)
    selected.to_csv(root / "selected_samples.tsv", sep="\t", index=False)
    excluded.to_csv(root / "excluded_samples.tsv", sep="\t", index=False)
    atomic_json(
        {
            "dataset": spec.gse,
            "domain": spec.domain,
            "title": spec.title,
            "platform": spec.platform,
            "samples": len(beta),
            "cpgs": beta.shape[1] - 1,
            "age_min": float(beta["age"].min()),
            "age_max": float(beta["age"].max()),
            "beta_min": float(finite.min()),
            "beta_max": float(finite.max()),
            "missing_beta_values": int(np.size(values) - np.isfinite(values).sum()),
            "cohort_definition": _cohort_definition(spec.gse),
        },
        root / "intermediate" / "beta_age.provenance.json",
    )
    return output


def _cohort_definition(gse: str) -> str:
    definitions = {
        "GSE184221": "All 384 healthy blood samples from both processed beta batches.",
        "GSE234461": "All 245 healthy whole-blood samples.",
        "GSE175458": "Control, diagnosis=normal, age>=18; 142 healthy lung samples.",
        "GSE232891": "Control samples age>=18; 248 healthy saliva samples.",
        "GSE287331": "Post-QC HDB, Case/Control=normal, age>=18; 182 healthy breast samples.",
    }
    return definitions[gse]
