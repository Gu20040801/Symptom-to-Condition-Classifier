# Milestone 1 Scope

## Project framing

This project is an educational symptom-to-condition classification demo. It is not a diagnostic tool, not medical advice, and must not be used for real medical decisions. A real symptom checker would require clinical validation, careful safety review, privacy controls, and oversight from qualified medical professionals.

## Dataset limitation

The Kaggle Disease Symptom Prediction dataset is commonly used for demos, but it is simplified and may be close to deterministic. It should be treated as a controlled machine-learning exercise rather than evidence that the model can generalize to real clinical settings.

## Milestone 1 deliverables

- Load and validate the symptom dataset.
- Train baseline supervised classifiers:
  - Decision Tree with overfitting controls.
  - Random Forest for comparison.
- Evaluate with:
  - Accuracy.
  - Macro-F1.
  - Top-3 accuracy.
  - Per-class precision, recall, and F1.
- Save the best model and evaluation report under `outputs/`.

## Explainability note

Decision Tree paths can support interpretability, but raw feature names are not enough for a user-facing explanation. Later milestones should convert model paths into cleaned text such as: "The model weighted fever, cough, and fatigue as important symptoms for this prediction."
