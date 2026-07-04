# Disease Symptom Classifier

Milestone 1 for an educational symptom-to-condition classifier.

## Safety framing

This project is only an educational machine-learning demo. It is not medical advice, not a diagnostic tool, and should not be used for real diagnosis or treatment decisions.

## Dataset

Use the Kaggle Disease Symptom Prediction dataset. For the downloaded archive used in this project, put these files in `data/`:

- `dataset.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`
- `Symptom-severity.csv`

Milestone 1 trains on `dataset.csv`. The other CSV files are kept for later milestones.

Important limitation: this dataset is clean, simplified, and may not represent real clinical records. Results on it should be presented as demo performance, not clinical validity.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run milestone 1

```bash
python src/milestone1.py --train data/dataset.csv
```

The script also supports the alternate one-hot Kaggle format with `Training.csv` and optional `Testing.csv`:

```bash
python src/milestone1.py --train data/Training.csv --test data/Testing.csv
```

Outputs are written to `outputs/`:

- `metrics.json`
- `classification_report.csv`
- `knn_baseline.joblib`

## Models

Milestone 1 trains a KNN baseline model. Decision Tree and Random Forest are left for the later model improvement phase.

## Evaluation

Milestone 1 reports accuracy, macro precision, macro recall, and a per-class classification report. Top-3 output and explainability are left for later milestones.
