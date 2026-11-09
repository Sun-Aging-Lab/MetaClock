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
commands accept the matching repeatable `--gse` option.

## Species

The species evolution plots read three pairwise tables from
`generalization/species/`: `pairwise_evolutionary_distances.csv`,
`pairwise_maximum_lifespan_differences.csv`, and
`pairwise_gestation_time_differences.csv`.

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
the species generalization jobs before plotting.
