from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_LABELS = [0, 1, 2]

def load_split(
    experiment_root: Path,
    split: str,
) -> pd.DataFrame:
    path = experiment_root / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Experiment table does not exist: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Experiment table is empty: {path}")

    return dataframe

def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model feature columns were found")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric:
        raise TypeError(
            "Model features must be numeric: "
            + ", ".join(non_numeric)
        )

    return feature_columns

def validate_experiment_tables(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for dataframe, expected_split in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
        values = set(dataframe["split"].astype(str))

        if values != {expected_split}:
            raise ValueError(
                f"Expected split {expected_split}, found {values}"
            )

        if dataframe["subject_id"].isna().any():
            raise ValueError(
                f"{expected_split} contains missing subject IDs"
            )

        if dataframe["target_label"].isna().any():
            raise ValueError(
                f"{expected_split} contains missing target labels"
            )

    train_subjects = set(train["subject_id"].astype(str))
    validation_subjects = set(
        validation["subject_id"].astype(str)
    )
    test_subjects = set(test["subject_id"].astype(str))

    if train_subjects & validation_subjects:
        raise ValueError("Train/validation subject leakage detected")

    if train_subjects & test_subjects:
        raise ValueError("Train/test subject leakage detected")

    if validation_subjects & test_subjects:
        raise ValueError("Validation/test subject leakage detected")

def create_model(random_seed: int) -> Pipeline:
    """
    Create a multiclass model using the installed scikit-learn API.

    Features are already imputed and scaled in Stage 14.
    """
    return Pipeline(
        steps=[
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )

def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(probabilities, dtype=float)

    for index, label in enumerate(y_true):
        if 0 <= int(label) < probabilities.shape[1]:
            one_hot[index, int(label)] = 1.0

    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    metrics["accuracy"] = float(
        accuracy_score(y_true, predictions)
    )

    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true, predictions)
    )

    metrics["precision_macro"] = float(
        precision_score(
            y_true,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    metrics["recall_macro"] = float(
        recall_score(
            y_true,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    metrics["macro_f1"] = float(
        f1_score(
            y_true,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    metrics["weighted_f1"] = float(
        f1_score(
            y_true,
            predictions,
            labels=CLASS_LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    per_class_f1 = f1_score(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )

    metrics["per_class_f1"] = {
        class_name: float(value)
        for class_name, value in zip(
            CLASS_NAMES,
            per_class_f1,
        )
    }

    metrics["mcc"] = float(
        matthews_corrcoef(y_true, predictions)
    )

    metrics["cohen_kappa"] = float(
        cohen_kappa_score(y_true, predictions)
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        predictions,
        labels=CLASS_LABELS,
    ).tolist()

    try:
        metrics["log_loss"] = float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
            )
        )
    except ValueError:
        metrics["log_loss"] = None

    metrics["brier_score_multiclass"] = multiclass_brier_score(
        y_true,
        probabilities,
    )

    unique_true = np.unique(y_true)

    if len(unique_true) == len(CLASS_LABELS):
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=CLASS_LABELS,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None

        try:
            metrics["pr_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=CLASS_LABELS,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["pr_auc_ovr_macro"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_ovr_macro"] = None

    metrics["classification_report"] = classification_report(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return metrics

def run_experiment(
    experiment_root: Path,
    output_root: Path,
    experiment_name: str,
    random_seed: int = 42,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    train = load_split(experiment_root, "train")
    validation = load_split(experiment_root, "validation")
    test = load_split(experiment_root, "test")

    validate_experiment_tables(
        train=train,
        validation=validation,
        test=test,
    )

    feature_columns = get_feature_columns(train)

    if get_feature_columns(validation) != feature_columns:
        raise ValueError(
            "Train and validation feature columns differ"
        )

    if get_feature_columns(test) != feature_columns:
        raise ValueError(
            "Train and test feature columns differ"
        )

    x_train = train[feature_columns].to_numpy(float)
    x_validation = validation[feature_columns].to_numpy(float)
    x_test = test[feature_columns].to_numpy(float)

    y_train = train["target_label"].astype(int).to_numpy()
    y_validation = (
        validation["target_label"].astype(int).to_numpy()
    )
    y_test = test["target_label"].astype(int).to_numpy()

    if not np.isfinite(x_train).all():
        raise ValueError("Training features contain NaN or Inf")

    if not np.isfinite(x_validation).all():
        raise ValueError("Validation features contain NaN or Inf")

    if not np.isfinite(x_test).all():
        raise ValueError("Test features contain NaN or Inf")

    model = create_model(random_seed)

    training_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - training_start

    inference_start = time.perf_counter()
    validation_probabilities = model.predict_proba(
        x_validation
    )
    validation_predictions = model.predict(x_validation)

    test_probabilities = model.predict_proba(x_test)
    test_predictions = model.predict(x_test)
    inference_time = time.perf_counter() - inference_start

    validation_metrics = calculate_metrics(
        y_true=y_validation,
        predictions=validation_predictions,
        probabilities=validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_true=y_test,
        predictions=test_predictions,
        probabilities=test_probabilities,
    )

    experiment_dir = output_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(
        dataframe: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        split: str,
    ) -> None:
        output = dataframe[
            [
                "subject_id",
                "acquisition_group_id",
                "split",
                "diagnosis_xml",
                "target_label",
            ]
        ].copy()

        output["prediction"] = predictions
        output["confidence"] = probabilities.max(axis=1)

        for index, class_name in enumerate(CLASS_NAMES):
            output[f"probability_{class_name}"] = (
                probabilities[:, index]
            )

        output.to_csv(
            experiment_dir / f"{split}_predictions.csv",
            index=False,
        )

    save_predictions(
        validation,
        validation_predictions,
        validation_probabilities,
        "validation",
    )

    save_predictions(
        test,
        test_predictions,
        test_probabilities,
        "test",
    )

    summary = {
        "experiment_name": experiment_name,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "train_record_count": len(train),
        "validation_record_count": len(validation),
        "test_record_count": len(test),
        "train_subject_count": train["subject_id"].nunique(),
        "validation_subject_count": (
            validation["subject_id"].nunique()
        ),
        "test_subject_count": test["subject_id"].nunique(),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "total_runtime_seconds": float(
            time.perf_counter() - start_time
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model": "class-weighted LogisticRegression",
        "random_seed": random_seed,
        "subject_leakage_checked": True,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
    }

    with (
        experiment_dir / "experiment_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    joblib.dump(
        model,
        experiment_dir / "model.joblib",
    )

    return summary
