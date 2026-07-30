# Project Milestone 2 Report

## 1. Brief Project Recap

Our project is an educational symptom-to-condition classification system. It uses the Kaggle Disease Symptom Prediction dataset, where each record contains a disease label and a set of symptoms. We convert raw symptom entries into binary machine-learning features (plus engineered severity-summary features) and train and compare KNN, Decision Tree, and Random Forest classifiers to predict the most likely condition(s) from a given symptom list.

This project is only a course demonstration. It is not medical advice, not a diagnostic tool, and should not be used for real medical decisions.

## 2. Significant Accomplishments

### Accomplishment 1: Built a Leakage-Aware Data Pipeline and Discovered a Duplication Problem

We extended the Milestone 1 pipeline (`src/milestone1.py`) with a duplicate-analysis step in `src/milestone2.py`. The raw dataset has 4,920 rows across 41 disease classes, but after converting symptoms to binary vectors we found only **304 unique symptom patterns** — 4,616 rows (94%) are exact duplicates of another row. We verified there are zero "ambiguous" patterns (the same symptom set never maps to two different diseases), so a naive random row-level split would let **259 of those duplicate patterns appear in both the training and test sets**, silently inflating accuracy. We fixed this by deduplicating to unique symptom patterns *before* splitting, so training and testing never share an identical symptom vector.

This is significant because it moves the project from "a model can memorize repeated rows" to "the pipeline actively detects and removes an evaluation-leakage bug." This is a much stronger foundation for measuring real generalization.

### Accomplishment 2: Trained and Compared Three Models, Revealing Real Overfitting Behavior

We trained and tuned three classifiers (KNN, Decision Tree, Random Forest) with `GridSearchCV` and stratified cross-validation, then compared them on held-out test accuracy, macro precision/recall/F1, and top-3 accuracy. Unlike the Milestone 1 baseline (which scored a suspicious 100% across the board), this comparison shows meaningfully different behavior per model:

```text
Model            Test Accuracy   Macro F1   Top-3 Accuracy
Random Forest    1.000           1.000      1.000
KNN              0.984           0.987      1.000
Decision Tree    0.689           0.554      0.705
```

We also ran a dedicated Decision Tree depth-vs-overfitting analysis (`decision_tree_depth_analysis.png`): as `max_depth` grows from 3 to unrestricted, training accuracy climbs to 100% while test accuracy tops out around 69%, a textbook overfitting curve. This is significant because it gives us actual evidence for model selection, rather than a single trivially-perfect number, and demonstrates the bias/variance tradeoff directly on our own data.

### Accomplishment 3: Produced Evaluation Visualizations and a Working Top-3 Prediction Demo

The pipeline now generates a grouped bar chart comparing all three models, a confusion matrix for the best model, a feature-importance chart from the Random Forest, and a saved end-to-end top-3 prediction demo. On a held-out example labeled "Hepatitis B," the model correctly ranked Hepatitis B first (43.5% probability) ahead of clinically related conditions (Hepatitis D, Chronic cholestasis) as runners-up — showing the top-3 output behaves sensibly, not just correctly.

This is significant because it moves the project from "a model file exists" to "we can inspect *why* the model does well, where it fails, and what a real prediction looks like end to end."

## 3. Proof of Accomplishment

### Proof 1: Duplicate/Leakage Analysis and Model Comparison Code

The file [`src/milestone2.py`](../src/milestone2.py) contains `analyze_and_deduplicate()` (pattern-key deduplication and leakage detection), `make_model_searches()` (KNN/Decision Tree/Random Forest tuning with `GridSearchCV`), and `analyze_tree_depths()` (the overfitting sweep). Running it produced this console summary:

```text
Rows: 4920 | Unique symptom patterns: 304 | Duplicates removed: 4616 | Classes: 41
A normal row-level split would place 259 symptom patterns in both training and testing.
Training and evaluating KNN...
KNN: test_accuracy=0.984, macro_f1=0.987, top_3_accuracy=1.000
Training and evaluating Decision Tree...
Decision Tree: test_accuracy=0.689, macro_f1=0.554, top_3_accuracy=0.705
Training and evaluating Random Forest...
Random Forest: test_accuracy=1.000, macro_f1=1.000, top_3_accuracy=1.000
Best model by macro F1: Random Forest
```

This proves progress because it shows executable code that detects a real data-quality problem and quantifies its impact, not just a script that trains one model.

### Proof 2: Model Comparison Chart and Decision Tree Overfitting Chart

`outputs/milestone2/model_comparison.png` (below, left) shows test accuracy, macro F1, and top-3 accuracy side by side for all three models. `outputs/milestone2/decision_tree_depth_analysis.png` (below, right) shows training accuracy diverging from testing accuracy as tree depth increases — direct visual evidence of overfitting.

![Model comparison](../outputs/milestone2/model_comparison.png)
![Decision tree depth analysis](../outputs/milestone2/decision_tree_depth_analysis.png)

This proves progress because the evaluation goes beyond a single accuracy number: it shows a controlled comparison across models and an explicit study of *why* one model (unconstrained Decision Tree) fails to generalize while another (Random Forest) does not.

### Proof 3: Confusion Matrix, Feature Importance, and Top-3 Prediction Demo

`outputs/milestone2/best_model_confusion_matrix.png` visualizes actual-vs-predicted classes for the best model (Random Forest) across all 41 diseases. `outputs/milestone2/feature_importance.png` shows the engineered severity features (`total_severity`, `symptom_count`) as the top predictors, ahead of any single symptom. `outputs/milestone2/sample_top3_prediction.txt` contains a full worked example: given the symptoms of a held-out "Hepatitis B" case, the model's top-3 output is Hepatitis B (43.5%), Hepatitis D (4.4%), and Chronic cholestasis (4.2%).

![Feature importance](../outputs/milestone2/feature_importance.png)

This proves progress because it demonstrates per-class evaluation, interpretability (which features drive predictions), and a realistic, reproducible end-to-end prediction — not just aggregate metrics.

## 4. Challenges or Roadblocks

The biggest technical challenge was discovering that 94% of the dataset rows are exact duplicates of other rows. This wasn't obvious from a first look at `dataset.csv` and only surfaced once we hashed the binary symptom vectors. It forced us to rethink the train/test split strategy: a standard `train_test_split` on raw rows would have let identical symptom patterns appear on both sides of the split, making the Milestone 1 "100% accuracy" result largely a measurement artifact rather than a real result. Deduplicating first reduced our usable data to 304 unique patterns across 41 classes (some classes have as few as 5 unique patterns), which also constrained how many cross-validation folds we could safely use.

A second challenge was the unconstrained Decision Tree overfitting badly (68.9% test accuracy vs. 100% training accuracy), which required us to add the explicit depth-sweep analysis to explain the gap rather than just reporting a worse number.

We also have not yet built a user-facing interface and still rely on the safety disclaimer embedded in script output and README to frame this as a non-diagnostic, educational tool.

## 5. Changes from Original Plan

The overall project direction has not changed. We are still building an educational symptom-to-condition classifier using the Kaggle Disease Symptom Prediction dataset.

The main adjustment is methodological: after finding the near-total row duplication and its leakage effect on a naive split, we changed our evaluation approach to split on unique symptom patterns instead of raw rows, and we now report macro F1 and top-3 accuracy alongside accuracy so a single dominant class can't hide poor per-class performance. We also expanded from a single KNN baseline to a three-model comparison (KNN, Decision Tree, Random Forest) with hyperparameter tuning, since a single model's near-perfect score gave us no way to judge whether the result was meaningful. Finally, we incorporated the `Symptom-severity.csv` file to engineer summary features (symptom count, total/average/maximum severity), which the feature-importance analysis shows are now the most predictive features in the model — this was an enhancement over the original plan of using only binary symptom indicators.
