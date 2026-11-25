from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Any

import joblib
import matplotlib
from sympy import python

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_LABELS = [0, 1, 2]

def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Feature table is empty: {path}"
        )

    required_metadata = {
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "target_label",
    }

    missing = required_metadata - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Feature table is missing columns: "
            + ", ".join(sorted(missing))
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
            "No model feature columns found"
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
            "SHAP feature columns must be numeric: "
            + ", ".join(non_numeric)
        )

    return feature_columns

def validate_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    train_features = get_feature_columns(train)
    test_features = get_feature_columns(test)

    if train_features != test_features:
        raise ValueError(
            "Training and test feature columns differ"
        )

    return train_features

def validate_subject_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_subjects = set(
        train["subject_id"].astype(str)
    )

    test_subjects = set(
        test["subject_id"].astype(str)
    )

    overlap = train_subjects.intersection(
        test_subjects
    )

    if overlap:
        raise ValueError(
            "Subject leakage between background and test data: "
            + "|".join(sorted(overlap))
        )

    if set(train["split"].astype(str)) != {"train"}:
        raise ValueError(
            "Background table must contain only split=train"
        )

    if set(test["split"].astype(str)) != {"test"}:
        raise ValueError(
            "Explanation table must contain only split=test"
        )

def normalize_shap_values(
    shap_values: Any,
    sample_count: int,
    feature_count: int,
    class_count: int,
) -> np.ndarray:
    """
    Normalize SHAP output into:

        samples x features x classes

    SHAP versions and explainers may return:
    - a 3D NumPy array
    - a list of arrays, one per class
    - a shap.Explanation object
    """
    values = getattr(
        shap_values,
        "values",
        shap_values,
    )

    if isinstance(values, list):
        arrays = [
            np.asarray(item)
            for item in values
        ]

        normalized = np.stack(
            arrays,
            axis=-1,
        )

        if normalized.shape[:2] == (
            sample_count,
            feature_count,
        ):
            return normalized

    array = np.asarray(values)

    if array.ndim == 2:
        if array.shape == (
            sample_count,
            feature_count,
        ):
            return array[:, :, None]

        if array.shape == (
            feature_count,
            sample_count,
        ):
            return array.T[:, :, None]

    if array.ndim == 3:
        if array.shape == (
            sample_count,
            feature_count,
            class_count,
        ):
            return array

        if array.shape == (
            sample_count,
            class_count,
            feature_count,
        ):
            return np.transpose(
                array,
                (0, 2, 1),
            )

        if array.shape == (
            class_count,
            sample_count,
            feature_count,
        ):
            return np.transpose(
                array,
                (1, 2, 0),
            )

    raise ValueError(
        "Unsupported SHAP output shape: "
        f"{array.shape}. Expected samples x features x classes."
    )

def make_prediction_function(
    model: Any,
    feature_count: int,
    class_count: int,
):
    """
    Return a stable probability function for SHAP.

    The output is always samples x classes and classes are aligned to
    CN, MCI, AD where possible.
    """
    def predict_proba(values: np.ndarray) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        probabilities = np.asarray(
            model.predict_proba(values),
            dtype=float,
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "Model predict_proba must return a 2D array"
            )

        model_classes = getattr(
            model,
            "classes_",
            np.arange(probabilities.shape[1]),
        )

        model_classes = np.asarray(
            model_classes,
            dtype=int,
        )

        aligned = np.zeros(
            (
                probabilities.shape[0],
                class_count,
            ),
            dtype=float,
        )

        for source_index, class_value in enumerate(
            model_classes
        ):
            if 0 <= class_value < class_count:
                aligned[:, class_value] = (
                    probabilities[:, source_index]
                )

        row_sums = aligned.sum(axis=1)
        valid = row_sums > 0

        aligned[valid] = (
            aligned[valid]
            / row_sums[valid, None]
        )

        return aligned

    return predict_proba

def save_global_importance(
    shap_array: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Save mean absolute SHAP importance overall and by class.
    """
    if shap_array.shape[2] == 1:
        class_names = ["model_output"]
    else:
        class_names = CLASS_NAMES[
            :shap_array.shape[2]
        ]

    overall_values = np.mean(
        np.abs(shap_array),
        axis=(0, 2),
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": overall_values,
        }
    )

    for class_index, class_name in enumerate(
        class_names
    ):
        importance[
            f"mean_abs_shap_{class_name}"
        ] = np.mean(
            np.abs(
                shap_array[:, :, class_index]
            ),
            axis=0,
        )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance.to_csv(
        output_dir / "global_shap_importance.csv",
        index=False,
    )

    top_count = min(20, len(importance))
    plot_data = importance.head(top_count)

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["feature"][::-1],
        plot_data["mean_abs_shap"][::-1],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(
        output_dir / "global_shap_importance.png",
        dpi=200,
    )
    plt.close()

    return importance

def save_sample_explanations(
    shap_array: np.ndarray,
    test_dataframe: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    rows = []

    for sample_index, row in test_dataframe.reset_index(
        drop=True
    ).iterrows():
        true_label = int(
            row["target_label"]
        )

        class_index = min(
            true_label,
            shap_array.shape[2] - 1,
        )

        for feature_index, feature_name in enumerate(
            feature_names
        ):
            rows.append(
                {
                    "sample_index": sample_index,
                    "subject_id": row["subject_id"],
                    "acquisition_group_id": row[
                        "acquisition_group_id"
                    ],
                    "true_diagnosis": row[
                        "diagnosis_xml"
                    ],
                    "true_target": true_label,
                    "explained_class": (
                        CLASS_NAMES[class_index]
                        if class_index < len(CLASS_NAMES)
                        else str(class_index)
                    ),
                    "feature": feature_name,
                    "shap_value": float(
                        shap_array[
                            sample_index,
                            feature_index,
                            class_index,
                        ]
                    ),
                    "absolute_shap_value": float(
                        abs(
                            shap_array[
                                sample_index,
                                feature_index,
                                class_index,
                            ]
                        )
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "sample_shap_values.csv",
        index=False,
    )

def run_shap_explanation(
    model_path: Path,
    background_path: Path,
    explanation_path: Path,
    output_dir: Path,
    max_background_samples: int = 20,
    max_explanation_samples: int = 20,
    random_seed: int = 42,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: {model_path}"
        )

    background = load_feature_table(
        background_path
    )

    explanation = load_feature_table(
        explanation_path
    )

    validate_subject_split(
        train=background,
        test=explanation,
    )

    feature_names = validate_feature_columns(
        train=background,
        test=explanation,
    )

    rng = np.random.default_rng(
        random_seed
    )

    background_count = min(
        max_background_samples,
        len(background),
    )

    explanation_count = min(
        max_explanation_samples,
        len(explanation),
    )

    background = background.sample(
        n=background_count,
        random_state=random_seed,
    ).reset_index(drop=True)

    explanation = explanation.sample(
        n=explanation_count,
        random_state=random_seed,
    ).reset_index(drop=True)

    background_values = background[
        feature_names
    ].to_numpy(dtype=np.float32)

    explanation_values = explanation[
        feature_names
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(background_values).all():
        raise ValueError(
            "Background features contain NaN or Inf"
        )

    if not np.isfinite(explanation_values).all():
        raise ValueError(
            "Explanation features contain NaN or Inf"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not expose predict_proba"
        )

    prediction_function = make_prediction_function(
        model=model,
        feature_count=len(feature_names),
        class_count=len(CLASS_LABELS),
    )

    background_probabilities = prediction_function(
        background_values
    )

    class_count = background_probabilities.shape[1]

    explainer = shap.Explainer(
        prediction_function,
        background_values,
        feature_names=feature_names,
    )

    shap_values = explainer(
        explanation_values
    )

    shap_array = normalize_shap_values(
        shap_values=shap_values,
        sample_count=len(explanation_values),
        feature_count=len(feature_names),
        class_count=class_count,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "shap_values.npy",
        shap_array,
    )

    with (
        output_dir / "feature_names.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            feature_names,
            handle,
            indent=2,
        )

    save_global_importance(
        shap_array=shap_array,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    save_sample_explanations(
        shap_array=shap_array,
        test_dataframe=explanation,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    predicted_probabilities = prediction_function(
        explanation_values
    )

    predicted_classes = np.argmax(
        predicted_probabilities,
        axis=1,
    )

    prediction_output = explanation[
        [
            "subject_id",
            "acquisition_group_id",
            "split",
            "diagnosis_xml",
            "target_label",
        ]
    ].copy()

    prediction_output["prediction"] = (
        predicted_classes
    )

    prediction_output["confidence"] = (
        predicted_probabilities.max(axis=1)
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        prediction_output[
            f"probability_{class_name}"
        ] = predicted_probabilities[
            :, class_index
        ]

    prediction_output.to_csv(
        output_dir / "explanation_predictions.csv",
        index=False,
    )

    summary = {
        "model_path": str(
            model_path.resolve()
        ),
        "background_path": str(
            background_path.resolve()
        ),
        "explanation_path": str(
            explanation_path.resolve()
        ),
        "background_sample_count": len(background),
        "explanation_sample_count": len(explanation),
        "feature_count": len(feature_names),
        "class_count": class_count,
        "class_names": CLASS_NAMES[
            :class_count
        ],
        "shap_values_shape": list(
            shap_array.shape
        ),
        "explainer_type": type(explainer).__name__,
        "random_seed": random_seed,
        "runtime_seconds": float(
            time.perf_counter() - start_time
        ),
        "subject_id_used_as_feature": False,
        "target_used_as_feature": False,
        "raw_data_modified": False,
    }

    with (
        output_dir / "shap_summary.json"
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


# ============================================================
# SHAP EXPLAINABILITY - PROJECT EXPLANATION
# ============================================================
#
# NAME:
# SHAP Explainability
#
# MODEL:
# SHAP is used to explain the contribution of input features
# to the model predictions.
#
# TECHNIQUE:
# Explainable AI (XAI) is used to understand which features
# influence the model's predictions.
#
# ALGORITHM:
# SHAP calculates a contribution value for each feature for
# the model output.
#
# CLASSES:
# CN = 0, MCI = 1, AD = 2.
#
# INPUT:
# Trained classification model, training background features,
# and test/explanation features.
#
# PREDICTION:
# The loaded model uses predict_proba() to generate class
# probabilities for CN, MCI, and AD.
#
# GLOBAL EXPLANATION:
# Mean absolute SHAP values are calculated to identify the
# overall importance of each feature.
#
# SAMPLE EXPLANATION:
# SHAP values are calculated for individual test samples to
# show the contribution of each feature.
#
# OUTPUT:
# global_shap_importance.csv stores overall and class-wise
# feature importance.
#
# global_shap_importance.png visualizes the top global features.
#
# sample_shap_values.csv stores feature-level SHAP values for
# individual samples.
#
# shap_values.npy stores the normalized SHAP values.
#
# explanation_predictions.csv stores predictions, confidence,
# and CN/MCI/AD probabilities.
#
# SUMMARY:
# shap_summary.json stores SHAP configuration, sample counts,
# feature count, runtime, and leakage information.
#
# IMPORTANT:
# subject_id and target_label are not used as model features.
# ============================================================


