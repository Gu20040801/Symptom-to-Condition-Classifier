# Design Doc: Milestone 1 Symptom-to-Condition Classifier

## 1. Goal

The goal of milestone 1 is to build a working Python machine learning pipeline for an educational symptom-to-condition classifier.

This project is only for a course demo. It is not a medical diagnostic tool, does not provide medical advice, and must not be used for real diagnosis.

## 2. What Has Been Built

The current milestone 1 implementation includes:

- A basic project structure.
- A Python dependency file: `requirements.txt`.
- A main training script: `src/milestone1.py`.
- Support for loading the Kaggle Disease Symptom Prediction `dataset.csv` file.
- Support for converting raw symptom columns into binary model features.
- Support for training a KNN baseline model.
- Support for evaluating models with multiple metrics:
  - Accuracy
  - Macro precision
  - Macro recall
  - Per-class precision / recall / F1
- Support for saving evaluation results and the trained KNN baseline model.
- A medical safety disclaimer in the README and output files.
- Documentation noting that the Kaggle dataset is clean and simplified, so the project should not overclaim that the model represents performance on real clinical records.

## 3. Current File Structure

```text
group_310/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
├── docs/
│   ├── design_doc.md
│   └── milestone1_scope.md
├── outputs/
│   ├── metrics.json
│   ├── classification_report.csv
│   └── knn_baseline.joblib
└── src/
    └── milestone1.py
```

## 4. Main Workflow

The entry point for milestone 1 is:

```text
src/milestone1.py
```

The script follows this workflow:

1. Load `data/dataset.csv`.
2. Check that the file has a `Disease` column and `Symptom_1...Symptom_17` columns.
3. Convert the raw symptom columns into binary model features.
4. Create a stratified train/test split from the dataset.
5. Use `LabelEncoder` to convert disease labels into numeric labels for model training.
6. Train a KNN baseline model.
7. Evaluate the baseline model.
8. Save the metrics, classification report, and model artifact.

The script also supports the alternate one-hot Kaggle format with `Training.csv`, optional `Testing.csv`, and a target column named `prognosis`.

## 5. Dataset Assumptions

The script assumes the Kaggle Disease Symptom Prediction dataset format.

The required milestone 1 training file should be placed at:

```text
data/dataset.csv
```

Additional files from the archive should also be kept in `data/` for later milestones:

```text
data/symptom_Description.csv
data/symptom_precaution.csv
data/Symptom-severity.csv
```

Expected data format:

- Each row contains a set of symptoms and one disease label.
- The disease label column is named `Disease`.
- Symptom columns are named `Symptom_1`, `Symptom_2`, and so on.
- The script converts these symptom strings into numeric/binary features.

Important limitation:

The Kaggle dataset is useful for a demo, but it is clean, simplified, and may be close to deterministic. Milestone 1 results only show performance on this course dataset. They do not prove that the model works on real clinical records.

## 6. Model Design

Milestone 1 uses a KNN baseline model. This matches the project timeline: the first milestone focuses on a complete and reproducible data pipeline plus a simple baseline model.

Decision Tree, Random Forest, feature importance, top-3 output, and explainability are intentionally left for later phases.

## 7. Evaluation Design

The TA feedback noted that accuracy alone may be misleading, so milestone 1 reports more than accuracy.

Current evaluation metrics:

- `accuracy`: Overall prediction correctness.
- `macro_precision`: Gives equal weight to each class when measuring precision.
- `macro_recall`: Gives equal weight to each class when measuring recall.
- `classification_report.csv`: Stores precision, recall, and F1 for each disease class.

## 8. Outputs

After running the script, these files are generated:

```text
outputs/
├── metrics.json
├── classification_report.csv
└── knn_baseline.joblib
```

File purposes:

- `metrics.json`: Stores the main model scores, the k value used, disclaimer, and dataset limitation. 
- `classification_report.csv`: Stores per-class precision, recall, and F1.
- `knn_baseline.joblib`: Stores the trained KNN baseline, label encoder, and feature column order for later milestones.

## 9. How To Run

Install the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the current dataset archive:

```bash
python src/milestone1.py --train data/dataset.csv
```

For the alternate one-hot Kaggle format:

```bash
python src/milestone1.py --train data/Training.csv --test data/Testing.csv
```

## 10. What Is Not Included In Milestone 1

The following items are not part of milestone 1:

- No web app or GUI.
- No symptom input interface for users.
- No real medical diagnosis functionality.
- No user-facing explanation text generator.
- No Decision Tree or Random Forest model.
- No top-3 output.
- No visualization charts.
- No deployment.
- No real clinical records.

## 11. Next Steps After Milestone 1

Possible next steps for later milestones:

- Add a simple prediction script that accepts symptoms and outputs top-3 conditions.
- Train Decision Tree and Random Forest models.
- Load a saved model artifact for inference.
- Convert raw feature names into readable explanations.
- Add a clearer explanation display for Decision Tree predictions.
- Add a confusion matrix and model comparison plots.
- Build a simple demo interface.
- Improve the final report with a clearer safety disclaimer and dataset limitation discussion.
