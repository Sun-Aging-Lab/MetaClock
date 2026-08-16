# Generalization analysis

This directory contains the compact implementation of array, tissue, and
species generalization analyses. It replaces experiment-history paths with a
small set of explicit inputs. Final estimator settings and validation live in
`modeling.py`; strict shard aggregation lives in `aggregation.py`.

## Directory map

- `array/`: array-platform training, aggregation, and plotting for GSE234461
  (MethylChipEPIC v1.0) and GSE184221 (MethylChip40k).
- `tissue/`: healthy saliva (GSE232891), breast (GSE287331), and lung
  (GSE175458) training, aggregation, and plotting.
- `species/`: species MLP-ELU components, leave-one-species-out component
  predictions, species generalization, Universal Clock 2 comparison, and the
  three evolution plots.
- `modeling.py`: estimators, split/resampling rules, strict point validation,
  safe reuse, and point provenance.
- `aggregation.py`: strict 30 x 50 shard aggregation and shared plot cutoff.
- `plotting.py`: exact current formal 90 x 60 mm generalization styling.
- `validate_generalization_outputs.py`: PDF/SVG canvas, font, text, and style
  validation for all 18 final outputs.

The four bundled Liberation Sans font files make vector typography independent
of cluster font installation. Their SHA-256 digests are enforced by `fonts.py`;
copyright and the SIL Open Font License 1.1 are included in
`assets/fonts/OFL.txt`.

Every external dataset path is a project-relative `Path(...)` whose definition
ends in `# fill in your path`. Edit the paths in the appropriate training file
or in `species/species_config.py`; no machine-specific path is embedded here.

## Input schemas

Direct CpG parquet files have sample IDs as the index, numeric `cg...` columns,
and an `age` column. Component-prediction parquet files use the identical sample
index and `age`, with one numeric column per component model. Missing CpGs are
zero-imputed after removing features that are entirely missing in the training
partition.

Species meta inputs contain exactly ten columns: nine leave-one-species-out
predictions named `<Species>_mlp_elu` plus `alltissueconcat_mlp_elu`. These are
fresh predictions from the recovered two-layer MLP with hidden dimensions
100/50, ELU, BatchNorm, dropout 0.5, AdamW, and 300 epochs.

The evolution inputs are intentionally distinct: each target-species parquet
contains the nine held-out component predictions named
`<Species>_elasticnetcv`. The pairwise-distance TSV contains
`species_a`, `species_b`, `distance`; the life-history TSV contains
`model_species`, `maximum_lifespan_years`, and `gestation_days`. Universal Clock
2 baselines are a CSV with `species` and `mae`.

## Final model and resampling settings

- Fixed 80/20 split with seed 105.
- Thirty unique integer training sizes on the log scale, beginning at 10 and
  ending ten samples below the 80% training-pool size.
- Fifty without-replacement resamples per training size. A CPU task is one
  dataset/species and one size index, and therefore owns all fifty resamples.
- EN and Meta EN: `StandardScaler` then scikit-learn `ElasticNetCV`, five-fold
  CV, 30 alphas from 1 to 1e-5, `l1_ratio=0.5`, `max_iter=20000`,
  `tol=1e-4`, and `n_jobs=-1`.
- SVM and Meta SVM: `StandardScaler` then linear `SVR`, `C=1`, `epsilon=0.1`,
  and unlimited iterations (`max_iter=-1`).
- The shared plot begins at the first size at or above 10 where the mean MAE of
  both EN and Meta EN is below the mean-prediction baseline. All five curves use
  that same cutoff. Bands are the 2.5th and 97.5th percentiles across 50
  resamples.
- Evolution reproduces the current formal evolution code: component weights use
  `StandardScaler` followed by non-negative five-fold ENCV; component and spline
  ENCV search 30 alphas from 10 to 1e-5. The cubic spline basis is intentionally
  not scaled a second time. Its interval uses 200 bootstrap resamples.

## Commands

Run one point per scheduler CPU task, changing the dataset/species and array
index from 0 through 29:

```bash
uv run python -m generalization.array.train_array_generalization --dataset GSE234461 --size-index 0
uv run python -m generalization.tissue.train_tissue_generalization --dataset GSE287331 --size-index 0
uv run python -m generalization.species.train_species_generalization --species BosTaurus --size-index 0
```

Before the species benchmark, train all eleven component sources (ten species
and `alltissueconcat`) and build each target's LOO prediction table:

```bash
uv run python -m generalization.species.train_species_mlp_elu_components --component BosTaurus
uv run python -m generalization.species.build_leave_one_species_out_mlp_predictions --species BosTaurus
```

After every point is complete, aggregate and plot:

```bash
uv run python -m generalization.array.summarize_array_generalization
uv run python -m generalization.array.plot_array_generalization
uv run python -m generalization.tissue.summarize_tissue_generalization
uv run python -m generalization.tissue.plot_tissue_generalization
uv run python -m generalization.species.summarize_species_generalization
uv run python -m generalization.species.plot_species_generalization
```

Generate the three evolution plots independently:

```bash
uv run python -m generalization.species.prepare_evolution_coefficients
uv run python -m generalization.species.plot_molecular_evolutionary_distance
uv run python -m generalization.species.plot_maximum_lifespan_difference
uv run python -m generalization.species.plot_gestation_time_difference
```

The preparation command fits the 90 directed component weights once. The three
plot commands reuse that validated table and only fit their relationship-specific
spline and bootstrap interval.

Run the synthetic tests and validate completed vector outputs from the repository
root:

```bash
uv run python -m unittest generalization.tests.test_generalization
uv run python -m generalization.validate_generalization_outputs
```

Point shards are only skipped when their complete model × repeat Cartesian grid
and JSON provenance match the requested configuration. Aggregation revalidates
all 30 shards and emits its own provenance audit.

## Outputs

The plotting scripts generate these output groups:

- array/tissue: `<GSE>_fewshot_clock_mi_svm_with_mean_baseline.{pdf,svg}`;
- species:
  `<Species>_fewshot_clock_with_mean_and_universal_baselines.{pdf,svg}`;
- evolution: `component_en_coef_vs_molecular_evolutionary_distance`,
  `component_en_coef_vs_max_lifespan_difference`, and
  `component_en_coef_vs_gestation_difference`, each as PDF/SVG plus a TSV audit.

The maximum-lifespan and gestation labels use mathematical absolute-value bars:
`|Maximum lifespan difference| (years)` and
`|Gestation time difference| (days)`.

Only the final baseline-comparison panels are emitted: every generalization
panel includes Mean, and every species panel additionally includes Universal
Clock. Baseline-free historical variants are intentionally not generated.
