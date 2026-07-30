"""Milestone 2 training, comparison, and prediction script.

This script extends the Milestone 1 KNN baseline by:

1. analyzing duplicate symptom patterns,
2. creating a duplicate-aware train/test dataset,
3. tuning and comparing KNN, Decision Tree, and Random Forest models,
4. reporting accuracy, macro precision, macro recall, macro F1, and top-3 accuracy,
5. analyzing Decision Tree depth and overfitting,
6. creating evaluation charts and a feature-importance chart,
7. saving the best model and a top-3 prediction demonstration.

Educational demonstration only. This is not medical advice and must not be
used to diagnose or treat any condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

# Use a non-interactive backend so plots can be created from PowerShell.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Reuse the tested Milestone 1 data-loading and symptom-cleaning functions.
from milestone1 import clean_symptom_name, load_dataset, split_features_target


RANDOM_STATE = 42
DISCLAIMER = (
    "Educational demonstration only. Not medical advice and not for real diagnosis."
)
DATASET_LIMITATION = (
    "The Kaggle symptom dataset is highly simplified and repetitive. High scores "
    "on this dataset do not establish clinical validity or real-world reliability."
)
SEVERITY_FEATURES = [
    "symptom_count",
    "total_severity",
    "average_severity",
    "maximum_severity",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Run Milestone 2 model comparison and top-3 prediction."
    )
    parser.add_argument(
        "--train",
        default="data/dataset.csv",
        help="Path to dataset.csv or another supported symptom CSV.",
    )
    parser.add_argument(
        "--severity",
        default="data/Symptom-severity.csv",
        help="Path to Symptom-severity.csv. Severity summaries are used when found.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/milestone2",
        help="Directory where Milestone 2 artifacts will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of unique symptom patterns reserved for testing.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified cross-validation folds used for model tuning.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Use one reasonable configuration per model instead of GridSearchCV.",
    )
    parser.add_argument(
        "--demo-symptoms",
        nargs="*",
        default=None,
        help=(
            "Optional symptoms for the saved top-3 demo, for example: "
            "--demo-symptoms itching skin_rash fatigue. "
            "Without this option, a held-out test example is used."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid command-line values with clear error messages."""
    if not 0.05 <= args.test_size <= 0.50:
        raise ValueError("--test-size must be between 0.05 and 0.50.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")


def pattern_keys(x_binary: pd.DataFrame) -> pd.Series:
    """Create an exact, order-independent key for each binary symptom vector.

    The original CSV stores symptoms across Symptom_1 to Symptom_17. The same
    symptom set can appear in a different column order, so duplicates must be
    detected after binary encoding rather than only on the raw CSV rows.
    """
    values = x_binary.to_numpy(dtype=np.uint8, copy=True)
    return pd.Series(
        [row.tobytes() for row in values],
        index=x_binary.index,
        name="pattern_key",
    )


def analyze_and_deduplicate(
    x_binary: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int | float]]:
    """Analyze repeated patterns and return one row per symptom pattern."""
    keys = pattern_keys(x_binary)
    pattern_labels = pd.DataFrame({"pattern_key": keys, "target": y})
    labels_per_pattern = pattern_labels.groupby("pattern_key")["target"].nunique()
    ambiguous_patterns = int((labels_per_pattern > 1).sum())

    # Measure how much leakage a normal row-level split would create.
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_patterns = set(keys.iloc[train_indices])
    test_patterns = set(keys.iloc[test_indices])
    overlapping_patterns = len(train_patterns & test_patterns)

    # A symptom vector linked to multiple labels is impossible to classify
    # consistently. This dataset currently has none, but we check explicitly.
    if ambiguous_patterns:
        raise ValueError(
            f"Found {ambiguous_patterns} symptom patterns connected to multiple "
            "disease labels. Resolve those conflicts before training."
        )

    keep_mask = ~keys.duplicated(keep="first")
    x_unique = x_binary.loc[keep_mask].reset_index(drop=True)
    y_unique = y.loc[keep_mask].reset_index(drop=True)

    class_counts = y_unique.value_counts()
    stats: dict[str, int | float] = {
        "total_rows": int(len(x_binary)),
        "symptom_features": int(x_binary.shape[1]),
        "disease_classes": int(y.nunique()),
        "unique_symptom_patterns": int(len(x_unique)),
        "duplicate_rows_removed": int(len(x_binary) - len(x_unique)),
        "ambiguous_patterns": ambiguous_patterns,
        "random_row_split_overlapping_patterns": int(overlapping_patterns),
        "minimum_unique_patterns_per_class": int(class_counts.min()),
        "rows_used_for_modeling": int(len(x_unique)),
    }
    return x_unique, y_unique, stats


def load_severity_weights(path: str | Path) -> dict[str, float]:
    """Load the optional symptom-severity mapping."""
    severity_path = Path(path)
    if not severity_path.exists():
        print(
            f"Severity file not found at {severity_path}. "
            "Continuing with binary symptom features only."
        )
        return {}

    severity_df = pd.read_csv(severity_path)
    required = {"Symptom", "weight"}
    missing = required - set(severity_df.columns)
    if missing:
        raise ValueError(
            f"Severity file is missing required columns: {sorted(missing)}"
        )

    weights: dict[str, float] = {}
    for _, row in severity_df.iterrows():
        symptom = clean_symptom_name(row["Symptom"])
        if symptom is None or pd.isna(row["weight"]):
            continue
        weights[symptom] = float(row["weight"])
    return weights


def add_severity_summary_features(
    x_binary: pd.DataFrame,
    severity_weights: dict[str, float],
) -> pd.DataFrame:
    """Add four summary features derived from the selected symptoms.

    Severity weights are fixed metadata, not patient-specific measurements.
    They are used only as a small feature-engineering experiment.
    """
    if not severity_weights:
        return x_binary.copy()

    weight_vector = np.array(
        [severity_weights.get(column, 0.0) for column in x_binary.columns],
        dtype=float,
    )
    values = x_binary.to_numpy(dtype=float)
    symptom_count = values.sum(axis=1)
    weighted_values = values * weight_vector
    total_severity = weighted_values.sum(axis=1)
    maximum_severity = weighted_values.max(axis=1)
    average_severity = np.divide(
        total_severity,
        symptom_count,
        out=np.zeros_like(total_severity),
        where=symptom_count > 0,
    )

    x_engineered = x_binary.copy()
    x_engineered["symptom_count"] = symptom_count
    x_engineered["total_severity"] = total_severity
    x_engineered["average_severity"] = average_severity
    x_engineered["maximum_severity"] = maximum_severity
    return x_engineered


def encode_split_labels(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode text disease labels as integers."""
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    unseen = sorted(set(y_test) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"Test set contains labels not seen in training: {unseen}")

    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder


def make_model_searches(
    cv: StratifiedKFold,
    skip_tuning: bool,
) -> dict[str, ClassifierMixin]:
    """Create the KNN, Decision Tree, and Random Forest training objects."""
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    if skip_tuning:
        knn.set_params(model__n_neighbors=5, model__weights="distance")
        tree.set_params(max_depth=10, min_samples_leaf=1, ccp_alpha=0.0)
        forest.set_params(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
        )
        return {
            "KNN": knn,
            "Decision Tree": tree,
            "Random Forest": forest,
        }

    # Macro F1 gives equal importance to all 41 classes.
    return {
        "KNN": GridSearchCV(
            estimator=knn,
            param_grid={
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "Decision Tree": GridSearchCV(
            estimator=tree,
            param_grid={
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_leaf": [1, 2],
                "ccp_alpha": [0.0, 0.001],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "Random Forest": GridSearchCV(
            estimator=forest,
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }


def unwrap_estimator(model: ClassifierMixin) -> ClassifierMixin:
    """Return the fitted estimator selected by GridSearchCV, when applicable."""
    if isinstance(model, GridSearchCV):
        return model.best_estimator_
    return model


def best_parameters(model: ClassifierMixin) -> dict[str, Any]:
    """Return simple, JSON-serializable parameters for a fitted model."""
    parameters = model.best_params_ if isinstance(model, GridSearchCV) else model.get_params(deep=True)
    simple_types = (str, int, float, bool, type(None))
    return {
        key: value
        for key, value in parameters.items()
        if isinstance(value, simple_types)
    }


def evaluate_classifier(
    name: str,
    fitted_model: ClassifierMixin,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    training_seconds: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Calculate the required holdout metrics and a per-class report."""
    estimator = unwrap_estimator(fitted_model)
    train_predictions = estimator.predict(x_train)
    test_predictions = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test)

    class_labels = np.asarray(estimator.classes_)
    row: dict[str, Any] = {
        "model": name,
        "training_accuracy": float(accuracy_score(y_train, train_predictions)),
        "testing_accuracy": float(accuracy_score(y_test, test_predictions)),
        "macro_precision": float(
            precision_score(y_test, test_predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, test_predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, test_predictions, average="macro", zero_division=0)
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(y_test, probabilities, k=3, labels=class_labels)
        ),
        "training_seconds": float(training_seconds),
        "best_parameters": json.dumps(best_parameters(fitted_model), sort_keys=True),
    }

    report = classification_report(
        y_test,
        test_predictions,
        labels=class_labels,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="class")
    report_df.insert(0, "model", name)
    return row, report_df, test_predictions, probabilities


def analyze_tree_depths(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Show how tree complexity changes training and testing performance."""
    depth_values: list[int | None] = [3, 5, 10, 15, 20, None]
    rows: list[dict[str, Any]] = []

    for depth in depth_values:
        model = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=depth,
            min_samples_leaf=1,
        )
        model.fit(x_train, y_train)
        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        rows.append(
            {
                "max_depth": "None" if depth is None else depth,
                "actual_tree_depth": int(model.get_depth()),
                "number_of_leaves": int(model.get_n_leaves()),
                "training_accuracy": float(
                    accuracy_score(y_train, train_predictions)
                ),
                "testing_accuracy": float(accuracy_score(y_test, test_predictions)),
                "macro_f1": float(
                    f1_score(y_test, test_predictions, average="macro", zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_model_comparison_chart(results: pd.DataFrame, output_path: Path) -> None:
    """Save a grouped chart for the main model metrics."""
    chart = results.set_index("model")[[
        "testing_accuracy",
        "macro_f1",
        "top_3_accuracy",
    ]]
    axis = chart.plot(kind="bar", figsize=(10, 6), rot=0)
    axis.set_title("Milestone 2 Model Comparison")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(["Test accuracy", "Macro F1", "Top-3 accuracy"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_tree_depth_chart(depth_results: pd.DataFrame, output_path: Path) -> None:
    """Save the Decision Tree overfitting analysis chart."""
    labels = depth_results["max_depth"].astype(str)
    positions = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(
        positions,
        depth_results["training_accuracy"],
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        positions,
        depth_results["testing_accuracy"],
        marker="o",
        label="Testing accuracy",
    )
    plt.plot(
        positions,
        depth_results["macro_f1"],
        marker="o",
        label="Macro F1",
    )
    plt.xticks(positions, labels)
    plt.xlabel("Configured max_depth")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Decision Tree Depth and Overfitting Analysis")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def find_tree_or_forest_model(
    fitted_models: dict[str, ClassifierMixin],
) -> tuple[str, ClassifierMixin]:
    """Prefer Random Forest for feature importance, then Decision Tree."""
    for name in ("Random Forest", "Decision Tree"):
        if name in fitted_models:
            return name, unwrap_estimator(fitted_models[name])
    raise ValueError("No tree-based model is available for feature importance.")


def save_feature_importance_chart(
    fitted_models: dict[str, ClassifierMixin],
    feature_columns: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save the most important features from a tree-based model."""
    model_name, estimator = find_tree_or_forest_model(fitted_models)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        raise ValueError(f"{model_name} does not expose feature_importances_.")

    importance_df = (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"Top {top_n} Features from {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrix_chart(
    predictions: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix for the selected best model."""
    figure, axis = plt.subplots(figsize=(18, 18))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=encoder.classes_,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name} Confusion Matrix")
    axis.tick_params(axis="both", labelsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_input_row(
    symptoms: list[str],
    symptom_columns: list[str],
    severity_weights: dict[str, float],
    final_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Convert a symptom list into the same features used for training."""
    cleaned = [clean_symptom_name(value) for value in symptoms]
    cleaned = sorted({value for value in cleaned if value is not None})
    known = [value for value in cleaned if value in symptom_columns]
    unknown = [value for value in cleaned if value not in symptom_columns]

    row = pd.DataFrame(0, index=[0], columns=symptom_columns, dtype=int)
    if known:
        row.loc[0, known] = 1

    row = add_severity_summary_features(row, severity_weights)
    row = row.reindex(columns=final_feature_columns, fill_value=0)
    return row, known, unknown


def top_three_predictions(
    estimator: ClassifierMixin,
    x_row: pd.DataFrame,
    encoder: LabelEncoder,
) -> list[tuple[str, float]]:
    """Return the three highest model probabilities."""
    probabilities = estimator.predict_proba(x_row)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    model_classes = np.asarray(estimator.classes_)
    return [
        (
            str(encoder.inverse_transform([int(model_classes[index])])[0]),
            float(probabilities[index]),
        )
        for index in top_indices
    ]


def save_prediction_demo(
    output_path: Path,
    best_model_name: str,
    best_estimator: ClassifierMixin,
    encoder: LabelEncoder,
    symptom_columns: list[str],
    final_feature_columns: list[str],
    severity_weights: dict[str, float],
    x_test_binary: pd.DataFrame,
    y_test_text: pd.Series,
    requested_symptoms: list[str] | None,
) -> None:
    """Save a readable top-3 prediction example for the report screenshot."""
    actual_label: str | None = None

    if requested_symptoms:
        demo_symptoms = requested_symptoms
        demo_source = "Command-line symptom list"
    else:
        # Use a real held-out example so the demonstration is reproducible.
        demo_symptoms = [
            column
            for column in symptom_columns
            if int(x_test_binary.iloc[0][column]) == 1
        ]
        actual_label = str(y_test_text.iloc[0])
        demo_source = "First held-out test example"

    x_row, known, unknown = build_input_row(
        demo_symptoms,
        symptom_columns,
        severity_weights,
        final_feature_columns,
    )
    predictions = top_three_predictions(best_estimator, x_row, encoder)

    lines = [
        DISCLAIMER,
        DATASET_LIMITATION,
        "",
        f"Model: {best_model_name}",
        f"Demo source: {demo_source}",
    ]
    if actual_label is not None:
        lines.append(f"Held-out dataset label: {actual_label}")

    lines.extend(["", "Selected known symptoms:"])
    if known:
        lines.extend(f"- {symptom}" for symptom in known)
    else:
        lines.append("- None")

    if unknown:
        lines.extend(["", "Ignored unknown symptoms:"])
        lines.extend(f"- {symptom}" for symptom in unknown)

    lines.extend(["", "Top three model predictions:"])
    for rank, (condition, probability) in enumerate(predictions, start=1):
        lines.append(f"{rank}. {condition}: {probability:.2%}")

    lines.extend(
        [
            "",
            "These are model probabilities from a simplified course dataset,",
            "not medical confidence scores.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    """Convert NumPy and pandas scalar types into JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    """Run the complete Milestone 2 workflow."""
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and encoding the symptom dataset...")
    raw_df = load_dataset(args.train, drop_duplicates=False)
    x_binary_full, y_full = split_features_target(raw_df)
    x_binary_unique, y_unique, dataset_stats = analyze_and_deduplicate(
        x_binary_full,
        y_full,
        args.test_size,
    )

    severity_weights = load_severity_weights(args.severity)
    x_features = add_severity_summary_features(x_binary_unique, severity_weights)
    dataset_stats["severity_weights_loaded"] = int(len(severity_weights))
    dataset_stats["engineered_severity_features"] = int(
        len(SEVERITY_FEATURES) if severity_weights else 0
    )
    dataset_stats["final_feature_count"] = int(x_features.shape[1])

    print(
        f"Rows: {dataset_stats['total_rows']} | "
        f"Unique symptom patterns: {dataset_stats['unique_symptom_patterns']} | "
        f"Duplicates removed: {dataset_stats['duplicate_rows_removed']} | "
        f"Classes: {dataset_stats['disease_classes']}"
    )
    print(
        "A normal row-level split would place "
        f"{dataset_stats['random_row_split_overlapping_patterns']} symptom patterns "
        "in both training and testing."
    )

    # Split only after canonical deduplication, preventing a symptom pattern from
    # appearing in both training and testing.
    indices = np.arange(len(y_unique))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y_unique,
    )
    x_train = x_features.iloc[train_indices].reset_index(drop=True)
    x_test = x_features.iloc[test_indices].reset_index(drop=True)
    x_test_binary = x_binary_unique.iloc[test_indices].reset_index(drop=True)
    y_train_text = y_unique.iloc[train_indices].reset_index(drop=True)
    y_test_text = y_unique.iloc[test_indices].reset_index(drop=True)

    y_train, y_test, encoder = encode_split_labels(y_train_text, y_test_text)

    minimum_training_class_count = int(pd.Series(y_train).value_counts().min())
    if args.cv_folds > minimum_training_class_count:
        raise ValueError(
            f"--cv-folds={args.cv_folds} is too large. The smallest training "
            f"class has {minimum_training_class_count} examples."
        )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    training_objects = make_model_searches(cv, args.skip_tuning)

    fitted_models: dict[str, ClassifierMixin] = {}
    metric_rows: list[dict[str, Any]] = []
    reports: list[pd.DataFrame] = []
    test_predictions_by_model: dict[str, np.ndarray] = {}

    for name, training_object in training_objects.items():
        print(f"Training and evaluating {name}...")
        start = perf_counter()
        training_object.fit(x_train, y_train)
        elapsed = perf_counter() - start

        row, report_df, predictions, _ = evaluate_classifier(
            name,
            training_object,
            x_train,
            x_test,
            y_train,
            y_test,
            encoder,
            elapsed,
        )
        fitted_models[name] = training_object
        metric_rows.append(row)
        reports.append(report_df)
        test_predictions_by_model[name] = predictions

        print(
            f"{name}: test_accuracy={row['testing_accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}, "
            f"top_3_accuracy={row['top_3_accuracy']:.3f}"
        )

    comparison = pd.DataFrame(metric_rows).sort_values(
        by=["macro_f1", "top_3_accuracy", "testing_accuracy"],
        ascending=False,
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(
        output_dir / "classification_reports.csv",
        index=False,
    )

    print("Running Decision Tree depth analysis...")
    depth_results = analyze_tree_depths(x_train, x_test, y_train, y_test)
    depth_results.to_csv(output_dir / "decision_tree_depth_results.csv", index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    best_training_object = fitted_models[best_model_name]
    best_estimator = unwrap_estimator(best_training_object)

    print(f"Best model by macro F1: {best_model_name}")
    save_model_comparison_chart(
        comparison,
        output_dir / "model_comparison.png",
    )
    save_tree_depth_chart(
        depth_results,
        output_dir / "decision_tree_depth_analysis.png",
    )
    save_feature_importance_chart(
        fitted_models,
        list(x_features.columns),
        output_dir / "feature_importance.png",
    )
    save_confusion_matrix_chart(
        test_predictions_by_model[best_model_name],
        y_test,
        encoder,
        best_model_name,
        output_dir / "best_model_confusion_matrix.png",
    )

    save_prediction_demo(
        output_dir / "sample_top3_prediction.txt",
        best_model_name,
        best_estimator,
        encoder,
        list(x_binary_unique.columns),
        list(x_features.columns),
        severity_weights,
        x_test_binary,
        y_test_text,
        args.demo_symptoms,
    )

    bundle = {
        "model_name": best_model_name,
        "model": best_estimator,
        "label_encoder": encoder,
        "symptom_columns": list(x_binary_unique.columns),
        "feature_columns": list(x_features.columns),
        "severity_weights": severity_weights,
        "dataset_stats": dataset_stats,
        "disclaimer": DISCLAIMER,
        "dataset_limitation": DATASET_LIMITATION,
    }
    joblib.dump(bundle, output_dir / "best_model_bundle.joblib")

    summary = {
        "disclaimer": DISCLAIMER,
        "dataset_limitation": DATASET_LIMITATION,
        "dataset_analysis": dataset_stats,
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": args.test_size,
            "training_rows": int(len(x_train)),
            "testing_rows": int(len(x_test)),
            "pattern_overlap_between_train_and_test": 0,
            "cross_validation_folds": args.cv_folds,
        },
        "best_model": best_model_name,
        "model_results": comparison.to_dict(orient="records"),
        "decision_tree_depth_results": depth_results.to_dict(orient="records"),
    }
    with (output_dir / "milestone2_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2)

    print(f"\nMilestone 2 artifacts written to: {output_dir}")
    print("Created:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
