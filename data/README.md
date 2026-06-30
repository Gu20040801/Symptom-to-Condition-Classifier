# Data Folder

Place the Kaggle Disease Symptom Prediction CSV files here.

Current archive format:

- `dataset.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`
- `Symptom-severity.csv`

Milestone 1 uses `dataset.csv`, which has a `Disease` column and `Symptom_1` through `Symptom_17` columns. The script converts those symptom columns into binary model features.

The script also supports an alternate one-hot format with `Training.csv`, optional `Testing.csv`, a target column named `prognosis`, and symptom columns encoded as numeric/binary features.

This dataset is useful for a class demo, but it is clean and simplified. It should not be described as real clinical records or used to claim clinical diagnostic performance.
