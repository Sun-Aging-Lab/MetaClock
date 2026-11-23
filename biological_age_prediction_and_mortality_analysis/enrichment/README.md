# CpG-weight GO enrichment

Both models exclude clocks without fixed-affine CpG coefficients before fitting. The chronological Elastic Net uses 8 of 16 candidates and 3,353 training profiles. The biological Elastic Net uses 17 of 19 candidates and one primary profile per InCHIANTI participant (672 total), with clinical PhenoAge as the target. It excludes the overall SystemsAge clock and DunedinPACE. Both fits select alpha by five-fold CV with training-fold standardization. The biological fit uses constant-zero imputation inside each training fold.

## Reproduce the GO BP results

Run from the repository root. Supply the UCSC hg18-to-hg19 chain for the 27K probes. The default foreground table contains the 1,000 highest and 1,000 lowest signed CpG weights for each model.

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union path/to/your/file
python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds
for model in chrono bio; do
  Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" go_bp
done
python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```

rGREAT 2.8.0 uses the 896,980-locus hg19 union of 27K, 450K and EPIC v1 probes. BH adjustment is within each model and direction after requiring at least five genes per term and five foreground region hits. The plotting command checks the region-level binomial P values and BH adjustments, and writes the figures, summary and significant terms. The reference run has one significant term, lipid droplet fusion (GO:0160077), among the lowest chronological weights (adjusted P = 0.0009421).

## Rebuild the CpG weights

Supply the audited component-level CpG coefficient tables. Chronological input columns are `clock_name,cpg_site,clock_cpg_weight`. Biological input columns are `clock,cpg,clock_feature_weight`. The following commands reuse the bundled fitted meta weights. The two model rebuilds can run independently.

```bash
go=biological_age_prediction_and_mortality_analysis.enrichment
union=results/enrichment/array_union/platform_membership.csv.gz
python -m "$go.rebuild_cpg_weights" chrono --component-weights path/to/chrono_coefficients.csv --membership "$union" --output-dir results/enrichment/weights/chrono
python -m "$go.rebuild_cpg_weights" bio --component-weights path/to/bio_coefficients.csv.gz --membership "$union" --output-dir results/enrichment/weights/bio
python -m "$go.select_top_bottom_cpg_foregrounds" --selected-cpgs results/enrichment/weights/chrono/selected_cpg_weights.csv results/enrichment/weights/bio/selected_cpg_weights.csv
```

To refit from component predictions, use the following commands. For biological age, the prepared metadata provide `sampleID` and `clinical_phenoage`. The pinned primary sample IDs determine the training cohort and its order. Component predictions must include `sampleID` and the selected clocks.

```bash
python -m "$go.rebuild_cpg_weights" chrono --component-weights path/to/chrono_coefficients.csv --membership "$union" --output-dir results/enrichment/refit/chrono --predictions chronological_age_prediction_performance/training_component_clock_predictions.csv --n-jobs 4
python -m "$go.rebuild_cpg_weights" bio --component-weights path/to/bio_coefficients.csv.gz --membership "$union" --output-dir results/enrichment/refit/bio --predictions biological_age_prediction_and_mortality_analysis/outputs/biological_age_prediction/component_clock_predictions.csv --matched-samples biological_age_prediction_and_mortality_analysis/outputs/preprocessing/inchianti_analysis_ready.parquet --n-jobs 4
```

## Other gene-set libraries

For KEGG, Reactome or Hallmark, supply the corresponding GMT file in the fifth argument's directory. After running all four libraries, use `--library all` to render every panel.

```bash
for library in kegg_2021_human reactome_pathways_2024 msigdb_hallmark_2020; do
  for model in chrono bio; do
    Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" "$library" results/enrichment/inputs results/enrichment/array_union/background_27k_450k_epic_union_hg19.bed.gz path/to/your/directory
  done
done
```
