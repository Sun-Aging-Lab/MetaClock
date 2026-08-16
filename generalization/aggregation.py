"""Strict shard aggregation and plot-cutoff selection for Figure 3."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import (
    MODEL_NAMES,
    DatasetSpec,
    GeneralizationConfig,
    atomic_write_json,
    config_record,
    log_spaced_training_sizes,
    project_path,
    validate_point_shard,
)

SUMMARY_COLUMNS = (
    "dataset",
    "model",
    "training_size",
    "mean",
    "q025",
    "q975",
    "plot_start",
)


def atomic_write_csv(frame: pd.DataFrame, output: Path) -> None:
    """Publish a CSV atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, suffix=".csv", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_validated_shards(
    spec: DatasetSpec,
    shard_root: Path,
    config: GeneralizationConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load all 30 shards after grid, configuration, and provenance validation."""
    shard_root = project_path(shard_root)
    frames: list[pd.DataFrame] = []
    provenances: list[dict[str, object]] = []
    training_pool: int | None = None
    for size_index in range(config.n_training_sizes):
        path = shard_root / spec.name / f"size_{size_index:02d}.parquet"
        provenance_path = path.with_suffix(".provenance.json")
        if not path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"Missing shard or provenance: {path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("dataset") != spec.name:
            raise ValueError(f"{provenance_path}: wrong dataset")
        if provenance.get("config") != config_record(config):
            raise ValueError(f"{provenance_path}: configuration differs")
        observed_pool = int(provenance["training_pool_samples"])
        if training_pool is None:
            training_pool = observed_pool
        elif observed_pool != training_pool:
            raise ValueError(f"{provenance_path}: training-pool size differs")
        sizes = log_spaced_training_sizes(training_pool, config)
        expected_size = int(sizes[size_index])
        if int(provenance["training_size"]) != expected_size:
            raise ValueError(f"{provenance_path}: training size differs")
        frame = pd.read_parquet(path)
        validate_point_shard(
            frame,
            size_index=size_index,
            training_size=expected_size,
            config=config,
        )
        frames.append(frame)
        provenances.append(provenance)
    merged = pd.concat(frames, ignore_index=True)
    expected_rows = config.n_training_sizes * config.n_resamples * len(MODEL_NAMES)
    if (
        len(merged) != expected_rows
        or merged.duplicated(["model", "size_index", "repeat_id"]).any()
    ):
        raise ValueError(f"{spec.name}: merged shard grid is incomplete")
    return merged, provenances


def aggregate_dataset(
    spec: DatasetSpec,
    shard_root: Path,
    summary_root: Path,
    config: GeneralizationConfig | None = None,
) -> Path:
    """Summarize 50 repeats and apply the shared EN/Meta-EN plot cutoff."""
    config = config or GeneralizationConfig()
    raw, provenances = load_validated_shards(spec, shard_root, config)
    summary = (
        raw.groupby(["model", "training_size"], observed=True)["mae"]
        .agg(
            mean="mean",
            q025=lambda values: values.quantile(0.025),
            q975=lambda values: values.quantile(0.975),
            repeats="size",
        )
        .reset_index()
    )
    if not summary["repeats"].eq(config.n_resamples).all():
        raise ValueError(f"{spec.name}: a summary cell does not contain 50 repeats")
    summary = summary.drop(columns="repeats")
    if set(summary["model"]) != set(MODEL_NAMES):
        raise ValueError(f"{spec.name}: summary model set differs")
    numeric = summary[["mean", "q025", "q975"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{spec.name}: summary contains non-finite values")
    wide = summary.pivot(index="training_size", columns="model", values="mean")
    eligible = wide.index[(wide["EN"] < wide["Mean"]) & (wide["Meta EN"] < wide["Mean"])]
    if eligible.empty:
        raise ValueError(f"{spec.name}: EN and Meta EN are never both below Mean")
    plot_start = max(10, int(eligible.min()))
    summary = summary.loc[summary["training_size"] >= plot_start].copy()
    summary.insert(0, "dataset", spec.name)
    summary["plot_start"] = plot_start
    summary = summary.loc[:, list(SUMMARY_COLUMNS)].sort_values(["model", "training_size"])
    output_root = project_path(summary_root)
    output = output_root / f"{spec.name}_plot_summary.csv"
    atomic_write_csv(summary, output)
    warnings_by_model = raw.groupby("model")["convergence_warnings"].sum().astype(int)
    audit = {
        "dataset": spec.name,
        "plot_start": plot_start,
        "raw_rows": len(raw),
        "summary_rows": len(summary),
        "convergence_warnings": warnings_by_model.to_dict(),
        "config": config_record(config),
        "point_provenance": [
            str(project_path(shard_root) / spec.name / f"size_{index:02d}.provenance.json")
            for index in range(config.n_training_sizes)
        ],
        "runtime_versions": provenances[0]["runtime"]["packages"],
    }
    atomic_write_json(audit, output.with_suffix(".provenance.json"))
    return output
