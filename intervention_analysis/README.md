# GEO intervention analysis: frozen 77-comparison workflow

This directory contains the two entry points for the formal analysis universe of
77 intervention comparisons from 64 GEO series. The later smoking-filtered,
POST-UV sensitivity, and minimum-P analyses are not inputs to this workflow.

## Main scripts

1. `/orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/fetch_geo_datasets_and_metadata.py`

   Fetches GEO metadata and the exact 64 DNAm matrix artifacts listed in the
   frozen acquisition manifest. Existing complete files are reused and file
   sizes are validated. The expected matrix download is 30,793,620,077 bytes
   (28.679 GiB).

   ```bash
   conda run --no-capture-output -n metaclock python \
     /orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/fetch_geo_datasets_and_metadata.py \
     --output-dir /orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam/geo_intervention_fetch_77
   ```

   Add `--dry-run` to validate and print the acquisition plan without network
   access or output writes.

2. `/orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/run_biometaclock_intervention_analysis.py`

   Loads the frozen BioMetaClock weights, predicts every retained GSE with its
   complete frozen branch context, performs the design-aware two-sided paired or
   Welch t-test, applies Benjamini-Hochberg correction across the 77 retained
   comparisons, cross-checks BH with SciPy, and creates PDF/SVG FDR volcano
   plots.

   ```bash
   conda run --no-capture-output -n metaclock python \
     /orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/run_biometaclock_intervention_analysis.py \
     --matrix-root /orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam/geo_intervention_fetch_77/matrices \
     --output-dir /orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam/geo_intervention_analysis_77_reproduction
   ```

The established GEO readers, PyAging inference, test implementation, and
volcano implementation remain in
`/orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam` and are
selected with `--pipeline-root`. The default frozen model is
`/orcd/pool/007/edsun_shared/yncheng/meta_clock_bio/rejuvenation_dnam/models/participant_level/external_all672_without_dnamfitage/bio_metaclock.joblib`.
Strict original beta-distribution QC remains enabled.

## Frozen configuration

`/orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/config`
contains the 77-result anchor, 64-GSE acquisition manifest, age-corrected and
paired-no-age sample/matrix manifests, selection audit, and source lineage.
These are immutable analysis inputs. Each analysis execution should use a new
output directory.

The run summaries record Python, NumPy, pandas, SciPy, statsmodels, and
Matplotlib versions plus the key runtime parameters. Benjamini-Hochberg uses
`statsmodels.stats.multitest.multipletests(method="fdr_bh")` and is validated
against `scipy.stats.false_discovery_control(method="bh")`.
