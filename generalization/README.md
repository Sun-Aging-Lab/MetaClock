# Generalization workflows

Array and tissue reproduction now start from the five public GEO accessions;
species retains its separate leave-one-species-out workflow. Human datasets
use three cached preparation stages, followed by parallel train/test shards and
plotting.

## Human input contract

The beta matrix alone is not sufficient: each sample must also have a unique
identifier and chronological age. The GEO download adapters obtain and audit
both. They intentionally use dataset-specific metadata keys and cohort rules
rather than guessing an age field from arbitrary files.

| Domain | GEO datasets | Selected healthy adults |
|---|---|---:|
| array | GSE184221, GSE234461 | 384, 245 |
| tissue | GSE175458, GSE232891, GSE287331 | 142, 248, 182 |

Each dataset is normalized below `data/generalization/<domain>/<GSE>/` and the
preparation script writes two aligned sample-indexed files:

- `prepared/direct_cpg_age.parquet`: 1,000 selected `cg...` beta features plus `age`;
- `prepared/component_predictions_age.parquet`: the exact 16 chronological
  PyAging predictions plus `age`.

The direct features reproduce the mutual-information 10,000 then mRMR 1,000
selection. PyAging 0.1.30 runs once per dataset and caches model weights; the 30
training tasks consume the cached prediction file instead of racing or doing
the same 16 predictions 30 times. Provenance records source URLs, cohort
definitions, package versions, parameters, and missingness.

## Array: download, PyAging, prepare, train/test

With no `--gse`, each of the first three commands processes both array datasets:

```bash
python -m generalization.array.download_geo
python -m generalization.array.predict_components
python -m generalization.array.prepare_training_inputs
sbatch generalization/array/submit_array_generalization.sbatch
python -m generalization.array.plot_array_generalization
```

To run one dataset only, pass its accession to every preparation stage and use
its named Slurm file:

```bash
python -m generalization.array.download_geo --gse GSE184221
python -m generalization.array.predict_components --gse GSE184221
python -m generalization.array.prepare_training_inputs --gse GSE184221
sbatch generalization/array/GSE184221_methylchip40.sbatch
```

`GSE234461_methylchipepic_v1.sbatch` is the corresponding GSE234461 submission.
A local point can be checked with:

```bash
python -m generalization.array.GSE184221_methylchip40 --size-index 0
```

## Tissue: download, PyAging, prepare, train/test

With no `--gse`, each preparation command processes all three tissue datasets:

```bash
python -m generalization.tissue.download_geo
python -m generalization.tissue.predict_components
python -m generalization.tissue.prepare_training_inputs
sbatch generalization/tissue/submit_tissue_generalization.sbatch
python -m generalization.tissue.plot_tissue_generalization
```

The one-dataset Slurm files are `GSE175458_lung.sbatch`,
`GSE232891_saliva.sbatch`, and `GSE287331_breast.sbatch`. The three preparation
commands accept the matching repeatable `--gse` option. `--data-root` and
`--cache-dir` provide explicit location overrides; run `--help` for force and
batch-size controls.

## Species input and workflow

For species, saved submodel artifacts generate nine non-self MLP-ELU
predictions plus `alltissueconcat_mlp_elu`; nine non-self ENCV predictions are
used only for evolution coefficients. The target species' own submodels are
never applied to that target. The MLP-ELU submodels use the specified
two-hidden-layer 100/50 architecture with ELU, BatchNorm, dropout 0.5, AdamW
(`learning_rate=1e-3`, `weight_decay=1e-3`), batch size 32, 300 epochs, and seed
105.

The species evolution plots read three pairwise tables from
`generalization/species/`: `pairwise_evolutionary_distances.csv`,
`pairwise_maximum_lifespan_differences.csv`, and
`pairwise_gestation_time_differences.csv`. Each table contains the same 45
unordered `species_a`/`species_b` pairs and its corresponding distance or
difference column.

## Species

Fetch the normalized beta matrix and series metadata from NCBI GEO, then
prepare the target and pooled inputs, train the saved submodels, build
target-specific LOO features, and run all ten species across the 30 sample
sizes. The beta download is approximately 4.2 GB; the fetcher supports
resuming a partial download and does not download the raw IDAT archive:

The target transformation rounds every gestation value to three decimal
places before calculating relative age, matching the retained recovery
inputs.

```bash
python -m generalization.species.fetch_gse223748
python -m generalization.species.prepare_species_gse223748_inputs
sbatch generalization/species/submit_species_mlp_elu_components.sbatch
sbatch generalization/species/submit_species_elasticnetcv_components.sbatch
sbatch generalization/species/submit_species_loo_features.sbatch
sbatch generalization/species/submit_species_generalization.sbatch
python -m generalization.species.plot_species_generalization
python -m generalization.species.plot_species_evolution
```

Submit these jobs in order, using Slurm `afterok` dependencies when they are
queued together. `PapioHamadryas` uses exact GSE223748 `Papio hamadryas`
samples; `papio_anubis` is only the Ensembl proxy recorded in the evolutionary
distance table.

The ENCV submodels generate the nine cross-species features for each target.
The size-zero species task then fits the fixed positive Elastic Net directed
coefficients used by the molecular-evolutionary-distance,
maximum-lifespan-difference, and gestation-difference panels. The plot command
reads the resulting ten coefficient tables and the three pairwise CSV tables.

## Fixed scientific configuration and outputs

All three training scripts use a fixed 80/20 split (seed 105), 30 unique
integer log-space training sizes from 10 through `training_pool - 10`, and 50
without-replacement resamples per size.  Each resample fits five rows:
EN, Meta EN, SVM, Meta SVM, and the training-mean baseline.

EN/Meta EN use `StandardScaler` followed by scikit-learn `ElasticNetCV` with
five folds, 30 alphas from 1 to 1e-5, `l1_ratio=0.5`, `max_iter=20000`,
`tol=1e-4`, and `n_jobs=-1`.  SVM/Meta SVM use `StandardScaler` followed by a
linear `SVR` with `C=1`, `epsilon=0.1`, `tol=1e-3`, and unlimited iterations.
Every estimator is refit within every resample.  Resample IDs use the
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

The three evolution panels use fixed-alpha definitions: directed
weights use `StandardScaler` plus non-negative
`ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000)`, and each cubic-spline
trend uses the same fixed-alpha Elastic Net with a 200-row bootstrap interval
with replacement (seed 42).
