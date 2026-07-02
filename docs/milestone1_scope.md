# Milestone 1 Scope

## Project framing

This project is an educational symptom-to-condition classification demo. It is not a diagnostic tool, not medical advice, and must not be used for real medical decisions. A real symptom checker would require clinical validation, careful safety review, privacy controls, and oversight from qualified medical professionals.

## Dataset limitation

The Kaggle Disease Symptom Prediction dataset is commonly used for demos, but it is simplified and may be close to deterministic. It should be treated as a controlled machine-learning exercise rather than evidence that the model can generalize to real clinical settings.

## Milestone 1 deliverables

- Load and validate the symptom dataset.
- Binary-encode symptom columns.
- Perform an 80/20 stratified train/test split.
- Train a KNN baseline classifier.
- Evaluate with:
  - Accuracy.
  - Macro precision.
  - Macro recall.
  - Per-class precision, recall, and F1.
- Save the KNN baseline model and evaluation report under `outputs/`.

## Explainability note

Decision Tree paths, top-3 outputs, severity weights, and user-facing explanations are not part of milestone 1. They should be handled in later milestones.
