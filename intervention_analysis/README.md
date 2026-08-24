# GEO intervention analysis: frozen 77-comparison workflow

This directory contains the two entry points for the formal analysis universe of
77 intervention comparisons from 64 GEO series.

## Main scripts

1. `/orcd/pool/007/edsun_shared/yncheng/metaclock/intervention_analysis/fetch_geo_datasets_and_metadata.py`

   Fetches GEO metadata and the exact 64 DNAm matrix artifacts listed in the
   frozen acquisition manifest. The expected matrix download is 28.7 G.

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



