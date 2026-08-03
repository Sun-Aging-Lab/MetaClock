# Figure 4 dataset placeholders

The repository intentionally does not include raw methylation matrices,
participant-level clinical records, or fitted models. Put your own inputs at
the following relative paths, or override them on the command line.

| Panel | Default relative path | Required columns | Note |
|---|---|---|---|
| 1 | `fig4/datasets/inchianti/oof_participant_predictions.csv` | `participant_id`, `model`, `observed_age`, `predicted_age` | One participant-level OOF prediction per model. Replace with your dataset path. |
| 2 | `fig4/datasets/inchianti/oof_all_record_predictions.csv` | `participant_id`, `sample_id`, `fold`, `model`, `predicted_age` | Includes the two repeated records for duplicate participants. Replace with your dataset path. |
| 3 | `fig4/datasets/disease/external_disease_predictions.csv` | `dataset`, `disease`, `sample_id`, `age`, `group`, `model`, `predicted_age` | `group` must be `control` or `disease`; every dataset/disease comparison must contain the complete 19-model correction family, with one row per sample/model. Replace with your disease datasets, e.g. `fig4/datasets/disease/GSE149318/...`. |
| 4 | `fig4/datasets/inchianti/cox_analysis_participants.csv` | `participant_id`, `survival_time_years`, `event`, `age_years`, `sex_male`, `bio_metaclock`, `baseline`, `clinical_phenoage` | One row per participant; `event` is 0/1. Replace with your dataset path. |
| 5 | `fig4/datasets/inchianti/km_analysis_participants.csv` | `participant_id`, `followup_wave`, `event`, `age_years`, `bio_metaclock`, `baseline`, `clinical_phenoage` | One row per participant. `followup_wave` is the study visit/wave endpoint used on the KM x-axis (primary event waves: 4, 5, 6), not survival time in years. Replace with your dataset path. |

The checked-in result tables are derived summaries, not substitutes for these
participant-level inputs.

The panel-3 correction family is exactly: `baseline`, `bio_metaclock`,
`dnamic`, `dnamphenoage`, `dunedinpace`, `hrsinchphenoage`, `systemsage`,
`systemsageblood`, `systemsagebrain`, `systemsageheart`, `systemsagehormone`,
`systemsageimmune`, `systemsageinflammation`, `systemsagekidney`,
`systemsageliver`, `systemsagelung`, `systemsagemetabolic`,
`systemsagemusculoskeletal`, and `zhangmortality`.
