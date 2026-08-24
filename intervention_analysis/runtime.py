"""Repository-local model, matrix, and PyAging adapters for intervention analysis."""

from __future__ import annotations

import csv
import fcntl
import gc
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from intervention_analysis.paths import DEFAULT_PYAGING_DIR, relative_path


MIN_CPG_COVERAGE = 0.90


@dataclass(frozen=True)
class DatasetSpec:
    accession: str
    matrix_file: str
    matrix_format: str
    samples: tuple[dict[str, Any], ...]


def dataset_spec(analysis_id: str, rows: pd.DataFrame) -> DatasetSpec:
    """Adapt frozen matrix assignments to the established parser contract."""
    samples: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        item: dict[str, Any] = {
            "sample_id": str(row.sample_id),
            "matrix_column": str(row.matrix_column),
        }
        column_index = getattr(row, "matrix_column_index_actual", None)
        if (
            column_index is not None
            and pd.notna(column_index)
            and str(column_index).strip()
        ):
            item["matrix_column_index_actual"] = int(float(column_index))
        samples.append(item)
    first = rows.iloc[0]
    return DatasetSpec(
        accession=str(analysis_id),
        matrix_file=str(first["matrix_file"]),
        matrix_format=str(first["matrix_format"]),
        samples=tuple(samples),
    )


def _numeric_row(values: list[str]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(float("nan"))
    return output


def _frame_from_rows(
    rows: dict[str, list[float]], selected_columns: list[str], spec: DatasetSpec
) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=selected_columns).T
    matrix_to_sample = {
        str(item["matrix_column"]): str(item["sample_id"]) for item in spec.samples
    }
    frame.index = [matrix_to_sample[str(value)] for value in frame.index]
    frame.index.name = "sample_id"
    return frame.astype(np.float32)


def _read_series_matrix(
    path: Path, spec: DatasetSpec, required_cpgs: set[str]
) -> pd.DataFrame:
    requested = [str(item["matrix_column"]) for item in spec.samples]
    rows: dict[str, list[float]] = {}
    selected_indices: list[int] | None = None
    selected_columns: list[str] | None = None
    in_table = False
    with gzip.open(
        path, "rt", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if not in_table:
                continue
            parsed = next(csv.reader([line], delimiter="\t"))
            if selected_indices is None:
                header = [value.strip('"') for value in parsed]
                missing = sorted(set(requested).difference(header))
                if missing:
                    raise KeyError(
                        f"{spec.accession}: sample columns missing from series matrix: {missing}"
                    )
                selected_indices = [header.index(value) for value in requested]
                selected_columns = requested
                continue
            if parsed and parsed[0].startswith("!series_matrix_table_end"):
                break
            cpg = parsed[0].strip('"')
            if cpg in required_cpgs:
                rows[cpg] = _numeric_row([parsed[index] for index in selected_indices])
    if selected_columns is None or not rows:
        raise ValueError(
            f"{spec.accession}: no methylation rows were parsed from "
            f"{relative_path(path)}"
        )
    return _frame_from_rows(rows, selected_columns, spec)


def parse_beta(path: Path, spec: DatasetSpec, required_cpgs: set[str]) -> pd.DataFrame:
    """Parse the series-matrix format not handled by the per-GSE normalized adapters."""
    if spec.matrix_format != "series_matrix":
        raise ValueError(
            f"{spec.accession}: unsupported matrix format {spec.matrix_format}"
        )
    return _read_series_matrix(path, spec, required_cpgs)


@contextmanager
def _cache_lock(pyaging_dir: Path) -> Iterator[None]:
    pyaging_dir.mkdir(parents=True, exist_ok=True)
    with (pyaging_dir / ".pyaging.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_biometa_clock_features(clocks: list[str], pyaging_dir: Path) -> list[str]:
    """Load the union CpG manifest, reading the shared SystemAge manifest once."""
    import pyaging as pya
    from pyaging.logger import LoggerManager

    representative_clocks: list[str] = []
    systemsage_seen = False
    for clock in clocks:
        if clock.startswith("systemsage"):
            if systemsage_seen:
                continue
            systemsage_seen = True
        representative_clocks.append(clock)
    logger = LoggerManager.gen_logger("load_biometaclock_features")
    features: list[str] = []
    seen: set[str] = set()
    with _cache_lock(pyaging_dir):
        for clock in representative_clocks:
            model = pya.pred.load_clock(
                clock, "cpu", relative_path(pyaging_dir), logger, indent_level=0
            )
            for feature in model.features:
                name = str(feature)
                if name not in seen:
                    seen.add(name)
                    features.append(name)
            del model
            gc.collect()
    return features


def run_pyaging_clocks(
    frame: pd.DataFrame,
    sample_id: pd.Series,
    clocks: list[str],
    pyaging_dir: Path = DEFAULT_PYAGING_DIR,
    batch_size: int = 2048,
    required_features: list[str] | None = None,
) -> pd.DataFrame:
    """Run maintained PyAging inference and return ordered component predictions."""
    import anndata as ad
    import pyaging as pya

    if required_features is None:
        required_features = load_biometa_clock_features(clocks, pyaging_dir)
    outputs: list[pd.DataFrame] = []
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        block = frame.iloc[start:end]
        matrix = np.zeros((len(block), len(required_features)), dtype=np.float32)
        positions = {feature: index for index, feature in enumerate(required_features)}
        present = [feature for feature in required_features if feature in block.columns]
        if present:
            matrix[:, [positions[feature] for feature in present]] = (
                block[present]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        adata = ad.AnnData(matrix)
        adata.var_names = required_features
        adata.obs_names = sample_id.iloc[start:end].astype(str).to_numpy()
        for source in ("age", "Age"):
            if source in block.columns:
                ages = pd.to_numeric(block[source], errors="coerce").to_numpy(
                    dtype=float
                )
                adata.obs["age"] = ages
                adata.obs["Age"] = ages
                break
        with _cache_lock(pyaging_dir):
            pya.pred.predict_age(
                adata,
                clocks,
                dir=relative_path(pyaging_dir),
                batch_size=batch_size,
                clean=True,
                verbose=False,
            )
        missing = [clock for clock in clocks if clock not in adata.obs]
        if missing:
            raise RuntimeError(f"PyAging did not return requested clocks: {missing}")
        output = adata.obs.loc[:, clocks].reset_index(drop=True)
        output.insert(0, "sample_id", sample_id.iloc[start:end].astype(str).to_numpy())
        outputs.append(output)
        del adata, matrix
        gc.collect()
    return pd.concat(outputs, ignore_index=True)
