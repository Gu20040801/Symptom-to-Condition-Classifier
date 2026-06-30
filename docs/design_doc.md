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
- Support for training two supervised classification models:
  - Decision Tree
  - Random Forest
- Support for evaluating models with multiple metrics:
  - Accuracy
  - Macro-F1
  - Top-3 accuracy
  - Per-class precision / recall / F1
- Support for saving evaluation results and the best trained model.
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
│   └── best_model.joblib
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
6. Use cross-validation to search for better model parameters.
7. Train a Decision Tree model and a Random Forest model.
8. Evaluate both models.
9. Select the best model based on test macro-F1.
10. Save the metrics, classification report, and model artifact.

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

### Decision Tree

The Decision Tree model is included because it is interpretable. This is useful for later explanation features.

However, Decision Trees can overfit easily, so the code tunes these parameters:

- `max_depth`
- `min_samples_leaf`
- `ccp_alpha`

### Random Forest

The Random Forest model is included as a stronger comparison baseline. It is usually more stable than a single Decision Tree, although it is less directly interpretable.

Milestone 1 does not assume one model is automatically better. The models are compared using evaluation metrics.

## 7. Evaluation Design

The TA feedback noted that accuracy alone may be misleading, so milestone 1 uses more than accuracy.

Current evaluation metrics:

- `accuracy`: Overall prediction correctness.
- `macro_f1`: Gives equal weight to each class, which is useful for multi-disease classification.
- `top_3_accuracy`: Relevant because the planned system outputs the top 3 possible conditions.
- `classification_report.csv`: Stores precision, recall, and F1 for each disease class.

Model selection rule:

```text
best model = highest test macro-F1
```

## 8. Outputs

After running the script, these files are generated:

```text
outputs/
├── metrics.json
├── classification_report.csv
└── best_model.joblib
```

File purposes:

- `metrics.json`: Stores the main model scores, best parameters, disclaimer, and dataset limitation.
- `classification_report.csv`: Stores per-class precision, recall, and F1.
- `best_model.joblib`: Stores the best model, label encoder, and feature column order for later milestones.

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

The script uses single-core training by default because this is more reliable in restricted environments. On a normal laptop, training can be sped up with:

```bash
python src/milestone1.py --train data/dataset.csv --n-jobs -1
```

## 10. What Is Not Included In Milestone 1

The following items are not part of milestone 1:

- No web app or GUI.
- No symptom input interface for users.
- No real medical diagnosis functionality.
- No user-facing explanation text generator.
- No visualization charts.
- No deployment.
- No real clinical records.

## 11. Next Steps After Milestone 1

Possible next steps for later milestones:

- Add a simple prediction script that accepts symptoms and outputs top-3 conditions.
- Load `best_model.joblib` for inference.
- Convert raw feature names into readable explanations.
- Add a clearer explanation display for Decision Tree predictions.
- Add a confusion matrix and model comparison plots.
- Build a simple demo interface.
- Improve the final report with a clearer safety disclaimer and dataset limitation discussion.
