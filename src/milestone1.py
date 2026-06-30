"""Milestone 1 training and evaluation script.

This is an educational symptom-to-condition classifier. It is not medical
advice and must not be used for real diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


ONE_HOT_TARGET_COLUMN = "prognosis"
RAW_TARGET_COLUMN = "Disease"
SYMPTOM_PREFIX = "Symptom_"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run milestone 1 model training and evaluation.")
    parser.add_argument(
        "--train",
        default="data/dataset.csv",
        help="Path to dataset.csv or one-hot Training.csv.",
    )
    parser.add_argument("--test", default=None, help="Optional path to one-hot Testing.csv.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for reports and model artifacts.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for model search. Use -1 for all cores if your environment allows it.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop exact duplicate rows before training. Off by default for this simplified dataset.",
    )
    return parser.parse_args()


def load_dataset(path: str | Path, drop_duplicates: bool = False) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Put the Kaggle CSV in data/ or pass --train/--test."
        )

    df = pd.read_csv(csv_path)
    unnamed_columns = [column for column in df.columns if column.startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if drop_duplicates:
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def clean_symptom_name(value: object) -> str | None:
    if pd.isna(value):
        return None

    symptom = str(value).strip().lower()
    if not symptom:
        return None

    return "_".join(symptom.replace("-", "_").split())


def convert_raw_symptom_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    symptom_columns = [column for column in df.columns if column.startswith(SYMPTOM_PREFIX)]
    if RAW_TARGET_COLUMN not in df.columns or not symptom_columns:
        raise ValueError(
            "Expected either a raw Kaggle dataset with columns "
            "'Disease' and 'Symptom_1...Symptom_17', or a one-hot dataset "
            "with target column 'prognosis'."
        )

    y = df[RAW_TARGET_COLUMN].astype(str).str.strip()
    symptom_sets = []
    all_symptoms: set[str] = set()

    for _, row in df[symptom_columns].iterrows():
        row_symptoms = {
            symptom
            for symptom in (clean_symptom_name(value) for value in row)
            if symptom is not None
        }
        symptom_sets.append(row_symptoms)
        all_symptoms.update(row_symptoms)

    feature_columns = sorted(all_symptoms)
    x = pd.DataFrame(0, index=df.index, columns=feature_columns, dtype=int)
    for row_index, row_symptoms in enumerate(symptom_sets):
        if row_symptoms:
            x.loc[row_index, list(row_symptoms)] = 1

    return x, y


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if RAW_TARGET_COLUMN in df.columns and any(
        column.startswith(SYMPTOM_PREFIX) for column in df.columns
    ):
        return convert_raw_symptom_dataset(df)

    if ONE_HOT_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{ONE_HOT_TARGET_COLUMN}' for one-hot encoded data."
        )

    x = df.drop(columns=[ONE_HOT_TARGET_COLUMN])
    y = df[ONE_HOT_TARGET_COLUMN].astype(str)

    non_numeric = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Expected symptom feature columns to be numeric/binary. "
            f"Non-numeric columns found: {non_numeric}"
        )

    return x, y


def choose_cv_splits(y_train: pd.Series) -> int:
    smallest_class_count = int(y_train.value_counts().min())
    if smallest_class_count < 2:
        raise ValueError(
            "Each disease class needs at least 2 training rows for stratified evaluation. "
            f"Smallest class has {smallest_class_count}."
        )
    return min(5, smallest_class_count)


def build_models(n_jobs: int, cv_splits: int) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    decision_tree = Pipeline(
        [
            (
                "model",
                DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            )
        ]
    )
    decision_tree_grid = {
        "model__max_depth": [6, 8, 12, 16, 20, 24],
        "model__min_samples_leaf": [1, 2, 5],
        "model__ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }

    random_forest = Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                ),
            )
        ]
    )
    random_forest_grid = {
        "model__n_estimators": [200],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [1, 2, 5],
    }

    return {
        "decision_tree": GridSearchCV(
            decision_tree,
            decision_tree_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        ),
        "random_forest": GridSearchCV(
            random_forest,
            random_forest_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        ),
    }


def encode_labels(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    unseen_labels = sorted(set(y_test) - set(encoder.classes_))
    if unseen_labels:
        raise ValueError(f"Test set contains labels not seen during training: {unseen_labels}")

    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder


def top3_accuracy(model: Pipeline, x_test: pd.DataFrame, y_test_encoded: np.ndarray) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(x_test)
    labels = np.arange(probabilities.shape[1])
    k = min(3, probabilities.shape[1])
    return float(top_k_accuracy_score(y_test_encoded, probabilities, k=k, labels=labels))


def evaluate_model(
    name: str,
    model: GridSearchCV,
    x_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = model.predict(x_test)
    best_estimator = model.best_estimator_

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test_encoded, predictions)),
        "macro_f1": float(f1_score(y_test_encoded, predictions, average="macro")),
        "top_3_accuracy": top3_accuracy(best_estimator, x_test, y_test_encoded),
        "best_cv_macro_f1": float(model.best_score_),
        "best_params": model.best_params_,
    }

    report = classification_report(
        y_test_encoded,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="class")
    report_df.insert(0, "model", name)

    return metrics, report_df


def make_train_test_data(
    train_path: str | Path, test_path: str | Path | None, drop_duplicates: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = load_dataset(train_path, drop_duplicates=drop_duplicates)
    x_train, y_train = split_features_target(train_df)

    if test_path:
        if not Path(test_path).exists():
            raise FileNotFoundError(f"Could not find test CSV: {test_path}")
        test_df = load_dataset(test_path, drop_duplicates=drop_duplicates)
        x_test, y_test = split_features_target(test_df)
        missing_columns = sorted(set(x_train.columns) - set(x_test.columns))
        extra_columns = sorted(set(x_test.columns) - set(x_train.columns))
        if missing_columns or extra_columns:
            raise ValueError(
                "Train/test feature columns do not match. "
                f"Missing in test: {missing_columns}; extra in test: {extra_columns}"
            )
        x_test = x_test[x_train.columns]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    return x_train, x_test, y_train, y_test


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = make_train_test_data(
        args.train, args.test, args.drop_duplicates
    )
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    all_metrics: list[dict[str, Any]] = []
    all_reports: list[pd.DataFrame] = []
    cv_splits = choose_cv_splits(y_train)
    trained_models = build_models(args.n_jobs, cv_splits)

    for name, search in trained_models.items():
        print(f"Training {name}...")
        search.fit(x_train, y_train_encoded)
        metrics, report_df = evaluate_model(name, search, x_test, y_test_encoded, label_encoder)
        all_metrics.append(metrics)
        all_reports.append(report_df)
        top_3 = metrics["top_3_accuracy"]
        top_3_text = "n/a" if top_3 is None else f"{top_3:.3f}"
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"top_3_accuracy={top_3_text}"
        )

    best = max(all_metrics, key=lambda item: item["macro_f1"])
    best_model = trained_models[best["model"]].best_estimator_

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "disclaimer": (
                    "Educational demo only. Not medical advice and not for real diagnosis."
                ),
                "dataset_limitation": (
                    "Kaggle symptom data is clean and simplified; results do not prove "
                    "clinical validity on real medical records."
                ),
                "metrics": all_metrics,
                "best_model": best["model"],
            },
            file,
            indent=2,
        )

    pd.concat(all_reports, ignore_index=True).to_csv(
        output_dir / "classification_report.csv", index=False
    )
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": label_encoder,
            "feature_columns": list(x_train.columns),
            "disclaimer": "Educational demo only. Not medical advice.",
        },
        output_dir / "best_model.joblib",
    )

    print(f"Best model by test macro-F1: {best['model']}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
