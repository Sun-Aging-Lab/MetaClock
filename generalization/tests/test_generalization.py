"""Minimal synthetic regression tests for generalization invariants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generalization.aggregation import aggregate_dataset
from generalization.modeling import (
    MODEL_NAMES,
    DatasetSpec,
    ElasticNetConfig,
    GeneralizationConfig,
    atomic_write_json,
    atomic_write_parquet,
    config_record,
    evaluate_training_size,
    log_spaced_training_sizes,
    split_and_impute,
    validate_point_shard,
)
from generalization.plotting import plot_generalization_panel
from generalization.species.evolution_plot_common import (
    EvolutionConfig,
    load_distances,
    nonnegative_component_encv,
    plot_evolution_relationship,
    spline_encv,
)
from generalization.species.species_config import MLP_LOO_ROOT, SPECIES, mlp_loo_path
from generalization.species.train_species_mlp_elu_components import safe_batch_size
from generalization.validate_generalization_outputs import (
    ExpectedPanel,
    validate_pdf,
    validate_svg,
)


class ModelingTests(unittest.TestCase):
    def test_point_task_safely_reuses_matching_shard(self) -> None:
        rng = np.random.default_rng(7)
        index = [f"sample_{i}" for i in range(40)]
        age = np.linspace(1, 20, len(index))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_path, meta_path = root / "direct.parquet", root / "meta.parquet"
            pd.DataFrame(
                {
                    "cg1": age + rng.normal(0, 0.1, len(age)),
                    "cg2": rng.normal(size=len(age)),
                    "age": age,
                },
                index=index,
            ).to_parquet(direct_path)
            pd.DataFrame(
                {"component_a": age + 0.2, "component_b": age - 0.2, "age": age},
                index=index,
            ).to_parquet(meta_path)
            spec = DatasetSpec(
                "Synthetic",
                "Synthetic",
                direct_path,
                meta_path,
                ("component_a", "component_b"),
            )
            config = GeneralizationConfig(
                n_resamples=1,
                n_training_sizes=1,
                elastic_net=ElasticNetConfig(n_jobs=1),
            )
            first = evaluate_training_size(spec, 0, root / "shards", config=config)
            first_mtime = first.stat().st_mtime_ns
            second = evaluate_training_size(spec, 0, root / "shards", config=config)
            self.assertEqual((first, first_mtime), (second, second.stat().st_mtime_ns))

    def test_training_grid_and_point_cartesian_product(self) -> None:
        config = GeneralizationConfig(n_resamples=2)
        sizes = log_spaced_training_sizes(400, config)
        self.assertEqual((len(sizes), int(sizes[0])), (30, 10))
        rows = []
        for model in MODEL_NAMES:
            for repeat in range(config.n_resamples):
                rows.append(
                    {
                        "model": model,
                        "mae": 1.0,
                        "prediction_sd": 0.2,
                        "convergence_warnings": 0,
                        "selected_alpha": 0.1 if model in {"EN", "Meta EN"} else np.nan,
                        "training_size": 10,
                        "size_index": 0,
                        "repeat_id": repeat,
                    }
                )
        validate_point_shard(pd.DataFrame(rows), size_index=0, training_size=10, config=config)

    def test_meta_component_cannot_be_silently_dropped(self) -> None:
        index = [f"s{i}" for i in range(50)]
        direct = pd.DataFrame({"cg1": np.arange(50), "age": np.arange(50)}, index=index)
        meta = pd.DataFrame({"component": np.nan, "age": np.arange(50)}, index=index)
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            split_and_impute(direct, meta, GeneralizationConfig())

    def test_species_builder_and_trainer_share_default_loo_root(self) -> None:
        self.assertEqual(mlp_loo_path(SPECIES[0]).parent, MLP_LOO_ROOT)

    def test_batchnorm_rejects_one_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            safe_batch_size(1, 32)

    def test_strict_aggregation_accepts_complete_grid(self) -> None:
        config = GeneralizationConfig(n_resamples=2, n_training_sizes=2)
        spec = DatasetSpec("Synthetic", "Synthetic", Path("d"), Path("m"), ("c",))
        sizes = log_spaced_training_sizes(40, config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, summaries = root / "shards", root / "summaries"
            for size_index, size in enumerate(sizes):
                rows = []
                for model in MODEL_NAMES:
                    for repeat in range(config.n_resamples):
                        mae = {"EN": 0.5, "Meta EN": 0.6, "Mean": 1.0}.get(model, 0.8)
                        rows.append(
                            {
                                "model": model,
                                "mae": mae,
                                "prediction_sd": 0.2,
                                "convergence_warnings": 0,
                                "selected_alpha": 0.1 if model in {"EN", "Meta EN"} else np.nan,
                                "training_size": int(size),
                                "size_index": size_index,
                                "repeat_id": repeat,
                            }
                        )
                path = shards / spec.name / f"size_{size_index:02d}.parquet"
                atomic_write_parquet(pd.DataFrame(rows), path)
                atomic_write_json(
                    {
                        "dataset": spec.name,
                        "training_pool_samples": 40,
                        "training_size": int(size),
                        "config": config_record(config),
                        "runtime": {"packages": {"synthetic": "1"}},
                    },
                    path.with_suffix(".provenance.json"),
                )
            output = aggregate_dataset(spec, shards, summaries, config)
            summary = pd.read_csv(output)
            self.assertEqual(set(summary["model"]), set(MODEL_NAMES))
            self.assertEqual(set(summary["plot_start"]), {10})


class EvolutionTests(unittest.TestCase):
    def test_formal_component_and_spline_configuration(self) -> None:
        config = EvolutionConfig()
        self.assertEqual(config.alphas[0], 10.0)
        component = nonnegative_component_encv(config)
        self.assertIn("scaler", component.named_steps)
        self.assertTrue(component.named_steps["model"].positive)
        self.assertNotIn("scaler", spline_encv(config).named_steps)

    def test_distance_table_requires_all_unique_pairs(self) -> None:
        pairs = [(a, b) for index, a in enumerate(SPECIES) for b in SPECIES[index + 1 :]]
        rows = [
            {"species_a": a, "species_b": b, "distance": index + 0.5}
            for index, (a, b) in enumerate(pairs)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distances.tsv"
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
            self.assertEqual(len(load_distances(path)), 45)
            rows[-1] = rows[0]
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "pair set differs"):
                load_distances(path)

    def test_evolution_plot_passes_pdf_svg_validator(self) -> None:
        rows = []
        for index, target in enumerate(SPECIES):
            for component_index, component in enumerate(SPECIES):
                if component == target:
                    continue
                distance = float(abs(index - component_index) + 1)
                rows.append(
                    {
                        "test_species": target,
                        "component_species": component,
                        "distance": distance,
                        "coefficient": 0.01 * (component_index + 1),
                        "maximum_lifespan_difference_years": distance * 2,
                        "gestation_difference_days": distance * 10,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coefficients = root / "coefficients.tsv"
            pd.DataFrame(rows).to_csv(coefficients, sep="\t", index=False)
            stem = root / "component_en_coef_vs_molecular_evolutionary_distance"
            pdf, svg = plot_evolution_relationship(
                "distance",
                "Molecular evolutionary distance",
                stem,
                coefficient_path=coefficients,
                config=EvolutionConfig(bootstrap_repeats=2, trend_points=20, n_jobs=1),
            )
            panel = ExpectedPanel(
                stem.name,
                "evolution",
                None,
                point_color="#839a8e",
                curve_color="#557264",
                shade_color="#c8d4cd",
            )
            validate_pdf(pdf, panel)
            validate_svg(svg, panel)


class PlottingTests(unittest.TestCase):
    def test_formal_plot_passes_pdf_svg_validator(self) -> None:
        rows = []
        for model_index, model in enumerate(MODEL_NAMES):
            for size in (10, 20, 40):
                mean = 2.0 + model_index * 0.1 - np.log10(size) * 0.1
                rows.append(
                    {
                        "dataset": "Synthetic",
                        "model": model,
                        "training_size": size,
                        "mean": mean,
                        "q025": mean - 0.1,
                        "q975": mean + 0.1,
                        "plot_start": 10,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            pd.DataFrame(rows).to_csv(summary, index=False)
            spec = DatasetSpec(
                "Synthetic", "Synthetic", Path("direct"), Path("meta"), ("component",)
            )
            pdf, svg = plot_generalization_panel(
                summary,
                spec,
                root / "Synthetic_fewshot_clock_mi_svm_with_mean_baseline",
            )
            panel = ExpectedPanel(
                "Synthetic_fewshot_clock_mi_svm_with_mean_baseline",
                "generalization",
                "Synthetic",
            )
            validate_pdf(pdf, panel)
            validate_svg(svg, panel)


if __name__ == "__main__":
    unittest.main()
