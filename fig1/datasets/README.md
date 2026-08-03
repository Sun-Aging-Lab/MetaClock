# Dataset placeholder

Add your sample-level component-clock prediction table here as
`all_predicted_ages_with_id.csv`, or supply another location with:

```bash
python fig1/scripts/run_ablation.py \
  --predictions fig1/datasets/all_predicted_ages_with_id.csv
```

Do not commit large methylation matrices or fitted model files to this folder.
The required columns and cohort assignments are documented in `../README.md`.
