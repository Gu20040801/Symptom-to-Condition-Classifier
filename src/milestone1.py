"""Milestone 1 training and evaluation script.

This is an educational symptom-to-condition classifier. It is not medical
advice and must not be used for real diagnosis.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


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
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for the KNN baseline.")
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


def build_knn_model(k: int) -> Pipeline:
    return Pipeline([("model", KNeighborsClassifier(n_neighbors=k))])


def encode_labels(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    unseen_labels = sorted(set(y_test) - set(encoder.classes_))
    if unseen_labels:
        raise ValueError(f"Test set contains labels not seen during training: {unseen_labels}")

    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test_encoded: pd.Series,
    label_encoder: LabelEncoder,
) -> tuple[dict[str, object], pd.DataFrame, pd.Series]:
    predictions = model.predict(x_test)

    metrics = {
        "model": "knn_baseline",
        "accuracy": float(accuracy_score(y_test_encoded, predictions)),
        "macro_precision": float(
            precision_score(y_test_encoded, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test_encoded, predictions, average="macro", zero_division=0)
        ),
        "k": model.named_steps["model"].n_neighbors,
    }

    report = classification_report(
        y_test_encoded,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="class")
    report_df.insert(0, "model", "knn_baseline")

    return metrics, report_df, predictions


def save_confusion_matrix(
    y_test_encoded: pd.Series,
    predictions: pd.Series,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_test_encoded, predictions)
    labels = list(label_encoder.classes_)
    cell_size = 18
    label_width = 250
    top_margin = 260
    width = label_width + len(labels) * cell_size + 30
    height = top_margin + len(labels) * cell_size + 40
    max_value = int(matrix.max()) or 1

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" font-family="Arial, sans-serif" font-size="22" '
        'font-weight="700">KNN Baseline Confusion Matrix</text>',
        '<text x="20" y="58" font-family="Arial, sans-serif" font-size="13" '
        'fill="#444">Rows are actual classes; columns are predicted classes.</text>',
    ]

    for index, label in enumerate(labels):
        x = label_width + index * cell_size + 12
        y = top_margin - 8
        svg_lines.append(
            f'<text x="{x}" y="{y}" transform="rotate(-60 {x} {y})" '
            'font-family="Arial, sans-serif" font-size="10" text-anchor="start">'
            f"{escape(label)}</text>"
        )

    for row_index, label in enumerate(labels):
        y = top_margin + row_index * cell_size + 13
        svg_lines.append(
            f'<text x="{label_width - 8}" y="{y}" font-family="Arial, sans-serif" '
            f'font-size="10" text-anchor="end">{escape(label)}</text>'
        )

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = int(matrix[row_index, column_index])
            intensity = value / max_value
            red = int(250 - 205 * intensity)
            green = int(248 - 95 * intensity)
            blue = int(245 - 155 * intensity)
            x = label_width + column_index * cell_size
            y = top_margin + row_index * cell_size
            svg_lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="rgb({red},{green},{blue})" stroke="#d6dde8" stroke-width="0.5"/>'
            )
            if value:
                svg_lines.append(
                    f'<text x="{x + cell_size / 2}" y="{y + 12}" '
                    'font-family="Arial, sans-serif" font-size="8" text-anchor="middle" '
                    f'fill="#102a43">{value}</text>'
                )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


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

    model = build_knn_model(args.k)
    print("Training knn_baseline...")
    model.fit(x_train, y_train_encoded)
    metrics, report_df, predictions = evaluate_model(
        model, x_test, y_test_encoded, label_encoder
    )
    print(
        f"knn_baseline: accuracy={metrics['accuracy']:.3f}, "
        f"macro_precision={metrics['macro_precision']:.3f}, "
        f"macro_recall={metrics['macro_recall']:.3f}"
    )

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
                "metrics": metrics,
            },
            file,
            indent=2,
        )

    report_df.to_csv(output_dir / "classification_report.csv", index=False)
    save_confusion_matrix(
        y_test_encoded,
        predictions,
        label_encoder,
        output_dir / "confusion_matrix.svg",
    )
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": list(x_train.columns),
            "disclaimer": "Educational demo only. Not medical advice.",
        },
        output_dir / "knn_baseline.joblib",
    )

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
