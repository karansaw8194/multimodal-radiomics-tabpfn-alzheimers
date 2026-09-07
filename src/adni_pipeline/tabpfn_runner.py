from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier

from .experiment_runner import (
    calculate_metrics,
)

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["CN", "MCI", "AD"]

def load_split(
    experiment_root: Path,
    split: str,
) -> pd.DataFrame:
    path = experiment_root / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"TabPFN input table does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"TabPFN input table is empty: {path}"
        )

    return dataframe

def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No TabPFN model features were found"
        )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "TabPFN features must be numeric: "
            + ", ".join(non_numeric)
        )

    return feature_columns

def validate_tables(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for dataframe, expected_split in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
        actual_split = set(
            dataframe["split"].astype(str)
        )

        if actual_split != {expected_split}:
            raise ValueError(
                f"Expected split={expected_split}, "
                f"found {actual_split}"
            )

        if dataframe["subject_id"].isna().any():
            raise ValueError(
                f"{expected_split} contains missing subject IDs"
            )

        if dataframe["target_label"].isna().any():
            raise ValueError(
                f"{expected_split} contains missing targets"
            )

    train_subjects = set(
        train["subject_id"].astype(str)
    )
    validation_subjects = set(
        validation["subject_id"].astype(str)
    )
    test_subjects = set(
        test["subject_id"].astype(str)
    )

    if train_subjects & validation_subjects:
        raise ValueError(
            "Train/validation subject leakage detected"
        )

    if train_subjects & test_subjects:
        raise ValueError(
            "Train/test subject leakage detected"
        )

    if validation_subjects & test_subjects:
        raise ValueError(
            "Validation/test subject leakage detected"
        )

def align_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """
    Align TabPFN probabilities to class order [0, 1, 2].

    This prevents incorrect class-probability column interpretation if an
    estimator returns classes in a different order.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    classes = np.asarray(classes)

    aligned = np.zeros(
        shape=(probabilities.shape[0], len(CLASS_LABELS)),
        dtype=float,
    )

    for source_index, class_value in enumerate(classes):
        class_value = int(class_value)

        if class_value in CLASS_LABELS:
            target_index = CLASS_LABELS.index(class_value)
            aligned[:, target_index] = (
                probabilities[:, source_index]
            )

    row_sums = aligned.sum(axis=1)
    nonzero_rows = row_sums > 0

    aligned[nonzero_rows] = (
        aligned[nonzero_rows]
        / row_sums[nonzero_rows, None]
    )

    return aligned

def create_tabpfn_model(
    device: str,
    n_estimators: int,
) -> TabPFNClassifier:
    """
    Create a TabPFN 8.x classifier.

    The verified installed API supports device and n_estimators. We do not
    pass undocumented or version-specific constructor arguments.
    """
    return TabPFNClassifier(
        device=device,
        n_estimators=n_estimators,
        show_progress_bar=False,
    )

def save_predictions(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
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
        output_path,
        index=False,
    )

def run_tabpfn_experiment(
    experiment_root: Path,
    output_root: Path,
    experiment_name: str,
    device: str = "cpu",
    n_estimators: int = 4,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    train = load_split(
        experiment_root,
        "train",
    )

    validation = load_split(
        experiment_root,
        "validation",
    )

    test = load_split(
        experiment_root,
        "test",
    )

    validate_tables(
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

    x_train = train[feature_columns].to_numpy(
        dtype=np.float32
    )

    x_validation = validation[feature_columns].to_numpy(
        dtype=np.float32
    )

    x_test = test[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_train = train["target_label"].astype(int).to_numpy()
    y_validation = (
        validation["target_label"].astype(int).to_numpy()
    )
    y_test = test["target_label"].astype(int).to_numpy()

    for split_name, values in (
        ("train", x_train),
        ("validation", x_validation),
        ("test", x_test),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"{split_name} features contain NaN or Inf"
            )

    model = create_tabpfn_model(
        device=device,
        n_estimators=n_estimators,
    )

    training_start = time.perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    training_time = time.perf_counter() - training_start

    inference_start = time.perf_counter()

    validation_probabilities_raw = model.predict_proba(
        x_validation
    )

    validation_predictions = model.predict(
        x_validation
    )

    test_probabilities_raw = model.predict_proba(
        x_test
    )

    test_predictions = model.predict(
        x_test
    )

    inference_time = time.perf_counter() - inference_start

    classes = getattr(
        model,
        "classes_",
        np.array(CLASS_LABELS),
    )

    validation_probabilities = align_probabilities(
        probabilities=validation_probabilities_raw,
        classes=classes,
    )

    test_probabilities = align_probabilities(
        probabilities=test_probabilities_raw,
        classes=classes,
    )

    validation_predictions = np.asarray(
        validation_predictions,
        dtype=int,
    )

    test_predictions = np.asarray(
        test_predictions,
        dtype=int,
    )

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

    experiment_directory = (
        output_root / experiment_name
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_predictions(
        dataframe=validation,
        predictions=validation_predictions,
        probabilities=validation_probabilities,
        output_path=(
            experiment_directory
            / "validation_predictions.csv"
        ),
    )

    save_predictions(
        dataframe=test,
        predictions=test_predictions,
        probabilities=test_probabilities,
        output_path=(
            experiment_directory
            / "test_predictions.csv"
        ),
    )

    model_path = (
        experiment_directory / "model.joblib"
    )

    model_serialization_error = None

    try:
        joblib.dump(
            model,
            model_path,
        )
    except Exception as exc:
        model_serialization_error = (
            f"{type(exc).__name__}: {exc}"
        )

    summary = {
        "experiment_name": experiment_name,
        "model": "TabPFNClassifier",
        "tabpfn_version": "8.5.0",
        "device": device,
        "n_estimators": n_estimators,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "train_record_count": len(train),
        "validation_record_count": len(validation),
        "test_record_count": len(test),
        "train_subject_count": int(
            train["subject_id"].nunique()
        ),
        "validation_subject_count": int(
            validation["subject_id"].nunique()
        ),
        "test_subject_count": int(
            test["subject_id"].nunique()
        ),
        "training_time_seconds": float(
            training_time
        ),
        "inference_time_seconds": float(
            inference_time
        ),
        "total_runtime_seconds": float(
            time.perf_counter() - start_time
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "subject_leakage_checked": True,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "model_serialization_path": str(
            model_path.resolve()
        ),
        "model_serialization_error": (
            model_serialization_error
        ),
    }

    with (
        experiment_directory
        / "experiment_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    return summary
