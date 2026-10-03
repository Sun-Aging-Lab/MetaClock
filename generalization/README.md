

## Array: download, PyAging, prepare, train/test

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

## Tissue: download, PyAging, prepare, train/test

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

For species, saved ENCV component pipelines generate nine non-self species
predictions plus `alltissueconcat_elasticnetcv` for both Meta models. The same
nine non-self species predictions support evolution coefficients; the pooled
clock is excluded from those directed coefficients. The target species' own
clock is never applied to that target, and the pooled training data exclude all
ten target species.

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
sizes. The fetcher supports resuming a partial download and does not download the raw IDAT archive.

The target transformation rounds every gestation value to three decimal
places before calculating relative age.

```bash
python -m generalization.species.fetch_gse223748
python -m generalization.species.prepare_species_gse223748_inputs
sbatch generalization/species/submit_species_elasticnetcv_components.sbatch
sbatch generalization/species/submit_species_loo_features.sbatch
sbatch generalization/species/submit_species_generalization.sbatch
python -m generalization.species.plot_species_generalization
python -m generalization.species.plot_species_evolution
```

After changing the component configuration, rerun input preparation (including
the pooled CpG input), all eleven component jobs, LOO feature generation, and
the species generalization jobs before plotting. Old component pipelines and
LOO tables are rejected by their configuration identifiers. Training and plotting
share the current species configuration identifier so old result shards cannot
be mixed with this run.

The ENCV submodels generate nine cross-species features and one pooled feature
for each target.
The size-zero species task then fits the fixed positive Elastic Net directed
coefficients used by the molecular-evolutionary-distance,
maximum-lifespan-difference, and gestation-difference panels. The plot command
reads the resulting ten coefficient tables and the three pairwise CSV tables.
