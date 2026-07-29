# Project Milestone 2 Report

## 1. Brief Project Recap

Our project is an educational symptom-to-condition classification system. It uses the Kaggle Disease Symptom Prediction dataset, where each record contains a disease label and a set of symptoms. The current method converts raw symptom entries into binary machine-learning features and trains a KNN baseline classifier to predict the most likely condition from the provided symptoms.

This project is only a course demonstration. It is not medical advice, not a diagnostic tool, and should not be used for real medical decisions.

## 2. Significant Accomplishments

### Accomplishment 1: Built a Reproducible Data Processing Pipeline

We completed a working data loading and preprocessing pipeline for the Kaggle symptom dataset. The script loads `data/dataset.csv`, validates that the expected disease and symptom columns are present, cleans symptom names into a consistent format, and converts each row of symptom strings into binary features. This gives the model a numeric input table where each column represents whether a symptom is present or absent.

This is significant because the original dataset stores symptoms as text across multiple symptom columns, which cannot be used directly by a standard scikit-learn classifier. By converting the symptom lists into binary features, we created the foundation for model training, evaluation, and later prediction features.

### Accomplishment 2: Trained and Saved a Baseline Machine Learning Model

We implemented and trained a KNN baseline model using scikit-learn. The training script performs a stratified train/test split so that each disease class is represented consistently in both training and evaluation. The labels are encoded with `LabelEncoder`, the model is trained on the processed symptom features, and the trained model artifact is saved to `outputs/knn_baseline.joblib`.

This is an important milestone because the project now has an end-to-end machine-learning workflow instead of only a project idea or dataset. The saved model artifact also gives us a starting point for future milestones, where we can add user input, top-k predictions, or compare KNN against other models such as Decision Tree or Random Forest.

### Accomplishment 3: Generated Structured Evaluation Results and Visual Proof

We evaluated the KNN baseline using multiple metrics instead of relying only on accuracy. The script now outputs overall accuracy, macro precision, macro recall, a per-class classification report, and a confusion matrix visualization. The current run produced the following summary:

```text
Model: KNN baseline
k: 5
Accuracy: 1.000
Macro precision: 1.000
Macro recall: 1.000
```

The per-class report in `outputs/classification_report.csv` also shows precision, recall, and F1-score for each disease class. In addition, the new `outputs/confusion_matrix.svg` file provides a visual summary of actual versus predicted classes.

This is significant because it gives concrete evidence that the baseline pipeline runs successfully and produces measurable outputs. It also helps us identify an important limitation: the very high scores likely reflect the simplified and clean nature of the Kaggle dataset, not real-world clinical reliability.

## 3. Proof of Accomplishment

### Proof 1: Data Processing and Model Training Code

The file `src/milestone1.py` contains the main project pipeline. It includes functions for loading the dataset, cleaning symptom names, converting raw symptom columns into binary features, splitting the data, encoding labels, training the KNN baseline, evaluating the model, and saving outputs.

This proves progress because the project has executable code that turns the raw dataset into a working trained classifier.

Suggested screenshot for PDF: show the `convert_raw_symptom_dataset`, `build_knn_model`, and `main` workflow sections from `src/milestone1.py`.

### Proof 2: Metrics Output

The file `outputs/metrics.json` stores the main evaluation metrics from a completed model run:

```json
{
  "model": "knn_baseline",
  "accuracy": 1.0,
  "macro_precision": 1.0,
  "macro_recall": 1.0,
  "k": 5
}
```

This proves progress because it shows that the model was trained, evaluated, and saved measurable performance results.

Suggested screenshot for PDF: show `outputs/metrics.json`.

### Proof 3: Classification Report and Confusion Matrix

The file `outputs/classification_report.csv` contains per-class precision, recall, F1-score, and support values for the disease classes. The generated `outputs/confusion_matrix.svg` visualizes the same evaluation from an actual-versus-predicted perspective.

This proves progress because it shows that the evaluation is not limited to a single summary number. The model's performance can be inspected across all disease classes, which is especially important for a multi-class classification problem.

Suggested screenshot for PDF: include the confusion matrix image from `outputs/confusion_matrix.svg` and a partial screenshot of `outputs/classification_report.csv`.

## 4. Challenges or Roadblocks

The main challenge is that the Kaggle symptom dataset is very clean and simplified. The model currently receives structured symptom lists, and many disease classes appear to have highly distinctive symptom combinations. Because of this, the baseline KNN model achieves perfect scores on the current train/test split. This is useful for confirming that the pipeline works, but it does not prove that the model would work on real patient data.

Another challenge is the safety framing of the project. Since the project involves disease labels, we need to be careful not to present it as a medical diagnostic system. The current README and output files include a disclaimer that this is only an educational demo and not medical advice.

We also have not yet built a user-facing interface, top-3 prediction output, model comparison, or explainability features. These are good candidates for later milestones after the baseline pipeline is stable.

## 5. Changes from Original Plan

The overall project direction has not changed. We are still building an educational symptom-to-condition classifier using the Kaggle Disease Symptom Prediction dataset.

The main adjustment is that we are treating the current KNN model as a baseline rather than a final model. Because the dataset appears highly simplified, we are avoiding overclaiming the evaluation results. Instead of saying the model is clinically accurate, we describe the current result as evidence that the pipeline works on the course dataset.

For future work, we plan to improve the project by adding richer model comparisons, user-friendly prediction output, clearer explanation features, and possibly a simple interface. We may also use the additional dataset files, such as symptom descriptions, precautions, and symptom severity scores, to make the system output more informative while still keeping the educational safety disclaimer clear.
