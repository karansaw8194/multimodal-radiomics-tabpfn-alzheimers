from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_LABELS = [0, 1, 2]
PROBABILITY_COLUMNS = [
    "probability_CN",
    "probability_MCI",
    "probability_AD",
]

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Prediction file is empty: {path}"
        )

    required_columns = {
        "target_label",
        "prediction",
        *PROBABILITY_COLUMNS,
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Prediction file is missing columns: "
            + ", ".join(sorted(missing))
        )

    return dataframe

def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    for index, label in enumerate(y_true):
        if 0 <= int(label) < probabilities.shape[1]:
            one_hot[index, int(label)] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )

def calibration_errors(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[float, float]:
    confidence = probabilities.max(axis=1)
    correctness = (
        predictions == y_true
    ).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0
    mce = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        bin_confidence = float(
            confidence[mask].mean()
        )

        bin_accuracy = float(
            correctness[mask].mean()
        )

        gap = abs(
            bin_confidence - bin_accuracy
        )

        ece += (
            count
            / len(y_true)
            * gap
        )

        mce = max(mce, gap)

    return float(ece), float(mce)

def specificity_by_class(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=CLASS_LABELS,
    )

    output = {}

    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[index, index]
        false_positive = (
            matrix[:, index].sum()
            - true_positive
        )
        false_negative = (
            matrix[index, :].sum()
            - true_positive
        )
        true_negative = (
            matrix.sum()
            - true_positive
            - false_positive
            - false_negative
        )

        denominator = true_negative + false_positive

        output[class_name] = float(
            true_negative / denominator
        ) if denominator > 0 else 0.0

    return output

def sensitivity_by_class(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    recalls = recall_score(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )

    return {
        class_name: float(value)
        for class_name, value in zip(
            CLASS_NAMES,
            recalls,
        )
    }

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    y_true = dataframe["target_label"].astype(int).to_numpy()
    predictions = dataframe["prediction"].astype(int).to_numpy()

    probabilities = dataframe[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Prediction probabilities contain NaN or Inf"
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Prediction probability rows do not sum to 1"
        )

    ece, mce = calibration_errors(
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
    )

    metrics: dict[str, Any] = {
        "sample_count": len(dataframe),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class_f1": {
            class_name: float(value)
            for class_name, value in zip(
                CLASS_NAMES,
                f1_score(
                    y_true,
                    predictions,
                    labels=CLASS_LABELS,
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "sensitivity_by_class": sensitivity_by_class(
            y_true,
            predictions,
        ),
        "specificity_by_class": specificity_by_class(
            y_true,
            predictions,
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                predictions,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=CLASS_LABELS,
        ).tolist(),
        "multiclass_brier_score": multiclass_brier_score(
            y_true,
            probabilities,
        ),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
    }

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

    observed_classes = np.unique(y_true)

    if len(observed_classes) == len(CLASS_LABELS):
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
                average_precision_score(
                    pd.get_dummies(
                        y_true
                    ).reindex(
                        columns=CLASS_LABELS,
                        fill_value=0,
                    ),
                    probabilities,
                    average="macro",
                )
            )
        except ValueError:
            metrics["pr_auc_ovr_macro"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_ovr_macro"] = None

    return metrics

def bootstrap_metric_intervals(
    dataframe: pd.DataFrame,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Bootstrap prediction rows to estimate percentile confidence intervals.

    These are exploratory intervals. Subject-cluster bootstrap is preferable
    for repeated longitudinal records and can be added when needed.
    """
    if iterations < 1:
        raise ValueError(
            "iterations must be at least 1"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    rng = np.random.default_rng(
        random_seed
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "cohen_kappa",
        "multiclass_brier_score",
        "expected_calibration_error",
        "maximum_calibration_error",
    ]

    values = {
        name: []
        for name in metric_names
    }

    sample_count = len(dataframe)

    for _ in range(iterations):
        indices = rng.integers(
            low=0,
            high=sample_count,
            size=sample_count,
        )

        sample = dataframe.iloc[
            indices
        ].reset_index(drop=True)

        try:
            metrics = calculate_metrics(sample)
        except (ValueError, ZeroDivisionError):
            continue

        for name in metric_names:
            metric_value = metrics.get(name)

            if metric_value is not None and np.isfinite(
                metric_value
            ):
                values[name].append(
                    float(metric_value)
                )

    alpha = 1.0 - confidence_level

    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (
        1.0 - alpha / 2.0
    )

    intervals = {}

    for name, metric_values in values.items():
        if not metric_values:
            intervals[name] = {
                "estimate": None,
                "lower": None,
                "upper": None,
                "valid_bootstrap_samples": 0,
            }
            continue

        intervals[name] = {
            "estimate": float(
                np.mean(metric_values)
            ),
            "lower": float(
                np.percentile(
                    metric_values,
                    lower_percentile,
                )
            ),
            "upper": float(
                np.percentile(
                    metric_values,
                    upper_percentile,
                )
            ),
            "valid_bootstrap_samples": len(
                metric_values
            ),
        }

    return {
        "confidence_level": confidence_level,
        "bootstrap_iterations": iterations,
        "random_seed": random_seed,
        "intervals": intervals,
        "note": (
            "Intervals are row-bootstrap estimates. With small test sets "
            "and repeated records, uncertainty estimates may be unstable."
        ),
    }

def run_evaluation(
    predictions_path: Path,
    output_dir: Path,
    experiment_name: str,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    dataframe = load_predictions(
        predictions_path
    )

    metrics = calculate_metrics(dataframe)

    intervals = bootstrap_metric_intervals(
        dataframe=dataframe,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_dir
        / f"{experiment_name}_predictions_copy.csv",
        index=False,
    )

    summary = {
        "experiment_name": experiment_name,
        "predictions_path": str(
            predictions_path.resolve()
        ),
        "metrics": metrics,
        "bootstrap_confidence_intervals": intervals,
        "bootstrap_iterations": bootstrap_iterations,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
    }

    with (
        output_dir
        / f"{experiment_name}_evaluation.json"
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
