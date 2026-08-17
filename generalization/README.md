# Generalization workflows

Array, tissue, and species generalization each use exactly the same three-step
workflow: one Python script fits/tests a single sample-size point, one Slurm
array submits all 30 points in parallel, and one Python script aggregates every
shard and writes all plots.  There is no separate summarize or validate command.

## Input table

For the relevant domain, edit the single line below in its training script:

```python
DATASET_PATH = Path('/path/to/your/file')  # fill in your path
```

The path can be CSV, TSV, or parquet.  It must contain:

- `age`, with one row per unique sample;
- direct methylation features named `cg...`;
- array/tissue: the 16 chronological component predictions declared in the
  training script;
- species: exactly ten `*_mlp_elu` predictions (the other nine formal species
  models and `alltissueconcat_mlp_elu`); the target species' own component is
  rejected to prevent leakage.

`sample_id`, `dataset`, `species`, and `title` are optional metadata columns.
If `sample_id` is absent, the existing table index is used.  A species table may
also contain nine `*_elasticnetcv` columns.  Those columns preserve the formal
evolution analysis, must likewise exclude the target species, and are not mixed
with the MLP-ELU few-shot Meta models.  The supplied MLP-ELU predictions use the
formal two-hidden-layer 100/50 architecture with ELU, BatchNorm, dropout 0.5,
AdamW (`learning_rate=1e-3`, `weight_decay=1e-3`), batch size 32, 300 epochs,
and seed 105.

The species evolution plots need all ten target-species runs.  Place shared
`pairwise_distance.tsv` and `life_history.tsv` beside the ten species input
files.  The first file has 45 `species_a`, `species_b`, `distance` rows.  The
second has one row per species with `model_species`,
`maximum_lifespan_years`, and `gestation_days`.

## Array

Edit `DATASET_PATH` in
`generalization/array/train_array_generalization.py`, then run:

```bash
sbatch generalization/array/submit_array_generalization.sbatch
uv run python -m generalization.array.plot_array_generalization
```

For a local one-point check, size index 0 is the default; another point can be
selected through the same environment variable used by Slurm:

```bash
SLURM_ARRAY_TASK_ID=7 uv run python -m generalization.array.train_array_generalization
```

## Tissue

Edit `DATASET_PATH` in
`generalization/tissue/train_tissue_generalization.py`, then run:

```bash
sbatch generalization/tissue/submit_tissue_generalization.sbatch
uv run python -m generalization.tissue.plot_tissue_generalization
```

## Species

Edit `DATASET_PATH` in
`generalization/species/train_species_generalization.py`, then run:

```bash
sbatch generalization/species/submit_species_generalization.sbatch
uv run python -m generalization.species.plot_species_generalization
```

The species plotting command also creates the molecular-evolutionary-distance,
maximum-lifespan-difference, and gestation-difference panels after all ten
species coefficient tables exist.  Before that, it still writes the completed
species generalization panels and reports that the cross-species panels are
pending.

## Fixed scientific configuration and outputs

All three training scripts use a fixed 80/20 split (seed 105), 30 unique
integer log-space training sizes from 10 through `training_pool - 10`, and 50
without-replacement resamples per size.  Each resample fits five rows:
EN, Meta EN, SVM, Meta SVM, and the training-mean baseline.

EN/Meta EN use `StandardScaler` followed by scikit-learn `ElasticNetCV` with
five folds, 30 alphas from 1 to 1e-5, `l1_ratio=0.5`, `max_iter=20000`,
`tol=1e-4`, and `n_jobs=-1`.  SVM/Meta SVM use `StandardScaler` followed by a
linear `SVR` with `C=1`, `epsilon=0.1`, `tol=1e-3`, and unlimited iterations.
Every estimator is refit within every resample.  Resample IDs use the formal
seed `42 + repeat_id + training_size * 1000`.

Each plotting script writes merged resampling source data, a
mean/2.5th/97.5th percentile summary, and four PDF/SVG layouts per dataset:

- starting at training size 10, with and without the Mean baseline;
- starting at the shared above-Mean cutoff, with and without the Mean baseline.

For EN and SVM separately, the family threshold is the first size at or above
10 where both the direct and corresponding Meta model have mean MAE below the
mean-prediction baseline.  The plotted above-Mean start shared by both subplots
is the maximum of those two family thresholds.  Training shards are written
under `results/generalization/`; figures are written under
`figures/generalization/`.

The three evolution panels retain the formal fixed-alpha definitions: directed
weights use `StandardScaler` plus non-negative
`ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000)`, and each cubic-spline
trend uses the same fixed-alpha Elastic Net with a 200-row bootstrap interval
with replacement (seed 42).
