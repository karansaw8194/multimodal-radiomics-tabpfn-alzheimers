from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["CN", "MCI", "AD"]


def load_table(path: Path) -> pd.DataFrame:
    """Load and validate a non-empty feature table."""
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Feature table does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Feature table is empty: {path}"
        )

    required_columns = {
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "target_label",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Feature table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    return dataframe


def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return only model feature columns.

    Subject ID, acquisition ID, split, diagnosis, and target are excluded.
    """
    feature_columns = sorted(
        column
        for column in dataframe.columns
        if column not in META_COLUMNS
    )

    if not feature_columns:
        raise ValueError(
            "No model feature columns were found"
        )

    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if non_numeric_columns:
        raise TypeError(
            "Counterfactual features must be numeric: "
            + ", ".join(non_numeric_columns)
        )

    return feature_columns


def validate_training_table(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the training reference table."""
    split_values = set(
        dataframe["split"].astype(str)
    )

    if split_values != {"train"}:
        raise ValueError(
            "Training reference table must contain only split=train. "
            f"Found: {split_values}"
        )

    if dataframe["subject_id"].isna().any():
        raise ValueError(
            "Training table contains missing subject IDs"
        )

    feature_columns = get_feature_columns(dataframe)

    values = dataframe[feature_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Training table contains NaN or infinite feature values"
        )


def validate_test_table(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the test explanation table."""
    split_values = set(
        dataframe["split"].astype(str)
    )

    if split_values != {"test"}:
        raise ValueError(
            "Test explanation table must contain only split=test. "
            f"Found: {split_values}"
        )

    if dataframe["subject_id"].isna().any():
        raise ValueError(
            "Test table contains missing subject IDs"
        )

    feature_columns = get_feature_columns(dataframe)

    values = dataframe[feature_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Test table contains NaN or infinite feature values"
        )


def validate_subject_separation(
    training: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure no subject occurs in both training and test data."""
    training_subjects = set(
        training["subject_id"].astype(str)
    )

    test_subjects = set(
        test["subject_id"].astype(str)
    )

    overlap = training_subjects.intersection(
        test_subjects
    )

    if overlap:
        raise ValueError(
            "Subject leakage between training and test data: "
            + "|".join(sorted(overlap))
        )


def load_model(
    model_path: Path,
) -> Any:
    """Load a saved model with probability prediction support."""
    model_path = model_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: {model_path}"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded model does not provide predict"
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not provide predict_proba"
        )

    return model


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Align probability columns to the fixed order:

        CN=0, MCI=1, AD=2
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Probabilities must be a two-dimensional array"
        )

    if model_classes is None:
        model_classes = np.arange(
            probabilities.shape[1]
        )

    model_classes = np.asarray(
        model_classes,
        dtype=int,
    )

    aligned = np.zeros(
        (
            probabilities.shape[0],
            len(CLASS_LABELS),
        ),
        dtype=float,
    )

    for source_index, class_value in enumerate(
        model_classes
    ):
        class_value = int(class_value)

        if class_value in CLASS_LABELS:
            aligned[
                :,
                class_value,
            ] = probabilities[
                :,
                source_index,
            ]

    row_sums = aligned.sum(axis=1)
    valid_rows = row_sums > 0

    aligned[valid_rows] = (
        aligned[valid_rows]
        / row_sums[valid_rows, None]
    )

    return aligned


def predict_probabilities(
    model: Any,
    values: np.ndarray,
) -> np.ndarray:
    """Predict class probabilities in CN, MCI, AD order."""
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    raw_probabilities = model.predict_proba(values)

    model_classes = getattr(
        model,
        "classes_",
        np.arange(
            raw_probabilities.shape[1]
        ),
    )

    return align_probabilities(
        probabilities=raw_probabilities,
        model_classes=model_classes,
    )


def predict_classes(
    model: Any,
    values: np.ndarray,
) -> np.ndarray:
    """Predict integer class labels."""
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    predictions = model.predict(values)

    return np.asarray(
        predictions,
        dtype=int,
    )


def compute_training_bounds(
    training_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute minimum, maximum, and scale values from training data only.
    """
    training_values = np.asarray(
        training_values,
        dtype=float,
    )

    lower_bounds = np.min(
        training_values,
        axis=0,
    )

    upper_bounds = np.max(
        training_values,
        axis=0,
    )

    scales = np.std(
        training_values,
        axis=0,
    )

    scales = np.where(
        np.isfinite(scales)
        & (scales > 1e-12),
        scales,
        1.0,
    )

    lower_bounds = np.where(
        np.isfinite(lower_bounds),
        lower_bounds,
        0.0,
    )

    upper_bounds = np.where(
        np.isfinite(upper_bounds),
        upper_bounds,
        lower_bounds,
    )

    upper_bounds = np.maximum(
        upper_bounds,
        lower_bounds,
    )

    return (
        lower_bounds,
        upper_bounds,
        scales,
    )


def infer_discrete_features(
    training: pd.DataFrame,
    feature_columns: list[str],
) -> set[int]:
    """
    Identify low-cardinality integer-like features.

    These are sampled from observed training values rather than continuously
    perturbed. This is a heuristic and does not establish clinical validity.
    """
    discrete_indices: set[int] = set()

    for index, column in enumerate(feature_columns):
        values = pd.to_numeric(
            training[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        unique_values = np.unique(
            values.to_numpy(dtype=float)
        )

        integer_like = np.allclose(
            unique_values,
            np.round(unique_values),
        )

        known_discrete_features = {
            "clin_sex",
            "clin_apoe_a1",
            "clin_apoe_a2",
            "clin_cdr",
            "clin_gds",
            "clin_mmse",
        }

        if (
            (integer_like and len(unique_values) <= 10)
            or column in known_discrete_features
        ):
            discrete_indices.add(index)

    return discrete_indices


def build_observed_discrete_values(
    training: pd.DataFrame,
    feature_columns: list[str],
    discrete_indices: set[int],
) -> dict[int, np.ndarray]:
    observed_values = {}

    for index in discrete_indices:
        column = feature_columns[index]

        values = pd.to_numeric(
            training[column],
            errors="coerce",
        ).dropna()

        observed_values[index] = np.unique(
            values.to_numpy(dtype=float)
        )

    return observed_values


def normalized_distance(
    original: np.ndarray,
    candidate: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Calculate Euclidean distance after training-derived scaling."""
    difference = (
        candidate - original
    ) / scales

    return float(
        np.sqrt(np.sum(difference ** 2))
    )


def generate_candidate(
    original: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    scales: np.ndarray,
    discrete_indices: set[int],
    observed_discrete_values: dict[int, np.ndarray],
    rng: np.random.Generator,
    mutation_probability: float,
    mutation_scale: float,
) -> np.ndarray:
    """Generate one bounded candidate feature vector."""
    candidate = original.copy()

    for feature_index in range(len(original)):
        if rng.random() > mutation_probability:
            continue

        if feature_index in discrete_indices:
            values = observed_discrete_values.get(
                feature_index,
                np.array([]),
            )

            if len(values) > 0:
                candidate[feature_index] = rng.choice(
                    values
                )

        else:
            perturbation = rng.normal(
                loc=0.0,
                scale=(
                    mutation_scale
                    * scales[feature_index]
                ),
            )

            candidate[feature_index] += perturbation

    candidate = np.clip(
        candidate,
        lower_bounds,
        upper_bounds,
    )

    return candidate


def search_counterfactual(
    model: Any,
    original: np.ndarray,
    target_class: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    scales: np.ndarray,
    discrete_indices: set[int],
    observed_discrete_values: dict[int, np.ndarray],
    candidate_count: int,
    mutation_probability: float,
    mutation_scale: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Search for the nearest successful candidate."""
    original_probabilities = predict_probabilities(
        model,
        original.reshape(1, -1),
    )[0]

    original_prediction = int(
        np.argmax(original_probabilities)
    )

    if target_class == original_prediction:
        return {
            "found": False,
            "reason": (
                "Requested target equals the original prediction"
            ),
            "original_prediction": original_prediction,
            "original_prediction_name": CLASS_NAMES[
                original_prediction
            ],
            "target_class": target_class,
            "target_class_name": CLASS_NAMES[
                target_class
            ],
            "original_target_probability": float(
                original_probabilities[target_class]
            ),
            "counterfactual_prediction": None,
            "counterfactual_target_probability": None,
            "distance": None,
            "candidate": None,
        }

    best_candidate = None
    best_probabilities = None
    best_distance = np.inf

    for _ in range(candidate_count):
        candidate = generate_candidate(
            original=original,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scales=scales,
            discrete_indices=discrete_indices,
            observed_discrete_values=(
                observed_discrete_values
            ),
            rng=rng,
            mutation_probability=mutation_probability,
            mutation_scale=mutation_scale,
        )

        probabilities = predict_probabilities(
            model,
            candidate.reshape(1, -1),
        )[0]

        prediction = int(
            np.argmax(probabilities)
        )

        if prediction != target_class:
            continue

        distance = normalized_distance(
            original=original,
            candidate=candidate,
            scales=scales,
        )

        if distance < best_distance:
            best_candidate = candidate.copy()
            best_probabilities = probabilities.copy()
            best_distance = distance

    if best_candidate is None:
        return {
            "found": False,
            "reason": (
                "No candidate reached the requested target class"
            ),
            "original_prediction": original_prediction,
            "original_prediction_name": CLASS_NAMES[
                original_prediction
            ],
            "target_class": target_class,
            "target_class_name": CLASS_NAMES[
                target_class
            ],
            "original_target_probability": float(
                original_probabilities[target_class]
            ),
            "counterfactual_prediction": None,
            "counterfactual_target_probability": None,
            "distance": None,
            "candidate": None,
        }

    counterfactual_prediction = int(
        np.argmax(best_probabilities)
    )

    return {
        "found": True,
        "reason": "Bounded search succeeded",
        "original_prediction": original_prediction,
        "original_prediction_name": CLASS_NAMES[
            original_prediction
        ],
        "target_class": target_class,
        "target_class_name": CLASS_NAMES[
            target_class
        ],
        "original_target_probability": float(
            original_probabilities[target_class]
        ),
        "counterfactual_prediction": counterfactual_prediction,
        "counterfactual_prediction_name": CLASS_NAMES[
            counterfactual_prediction
        ],
        "counterfactual_target_probability": float(
            best_probabilities[target_class]
        ),
        "distance": float(best_distance),
        "candidate": best_candidate.tolist(),
    }


def add_changed_feature_columns(
    row: dict[str, Any],
    original: np.ndarray,
    candidate: list[float] | None,
    feature_names: list[str],
) -> dict[str, Any]:
    """Add original, counterfactual, and change columns."""
    if candidate is None:
        return row

    candidate_array = np.asarray(
        candidate,
        dtype=float,
    )

    for index, feature_name in enumerate(
        feature_names
    ):
        row[
            f"original_{feature_name}"
        ] = float(original[index])

        row[
            f"counterfactual_{feature_name}"
        ] = float(candidate_array[index])

        row[
            f"change_{feature_name}"
        ] = float(
            candidate_array[index]
            - original[index]
        )

    return row


def run_counterfactuals(
    model_path: Path,
    training_path: Path,
    test_path: Path,
    output_dir: Path,
    candidate_count: int = 100,
    max_samples: int | None = None,
    target_classes: list[int] | None = None,
    mutation_probability: float = 0.5,
    mutation_scale: float = 0.25,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Generate bounded counterfactual explanations.

    Training data is used only for:
    - feature bounds
    - feature scales
    - observed values for discrete features

    Test data is used only for explanation.
    """
    start_time = time.perf_counter()

    if candidate_count < 1:
        raise ValueError(
            "candidate_count must be at least 1"
        )

    if max_samples is not None and max_samples < 1:
        raise ValueError(
            "max_samples must be at least 1"
        )

    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError(
            "mutation_probability must be between 0 and 1"
        )

    if mutation_scale <= 0:
        raise ValueError(
            "mutation_scale must be greater than zero"
        )

    if target_classes is None:
        target_classes = CLASS_LABELS.copy()

    invalid_classes = set(target_classes) - set(
        CLASS_LABELS
    )

    if invalid_classes:
        raise ValueError(
            "Invalid target classes: "
            + ", ".join(
                str(value)
                for value in sorted(invalid_classes)
            )
        )

    training = load_table(training_path)
    test = load_table(test_path)
    model = load_model(model_path)

    validate_training_table(training)
    validate_test_table(test)

    validate_subject_separation(
        training=training,
        test=test,
    )

    feature_names = get_feature_columns(training)
    test_feature_names = get_feature_columns(test)

    if feature_names != test_feature_names:
        raise ValueError(
            "Training and test feature columns differ"
        )

    if max_samples is not None:
        test = test.head(
            max_samples
        ).copy()

    training_values = training[
        feature_names
    ].to_numpy(dtype=float)

    test_values = test[
        feature_names
    ].to_numpy(dtype=float)

    lower_bounds, upper_bounds, scales = (
        compute_training_bounds(training_values)
    )

    discrete_indices = infer_discrete_features(
        training=training,
        feature_columns=feature_names,
    )

    observed_discrete_values = (
        build_observed_discrete_values(
            training=training,
            feature_columns=feature_names,
            discrete_indices=discrete_indices,
        )
    )

    rng = np.random.default_rng(
        random_seed
    )

    original_probabilities = predict_probabilities(
        model,
        test_values,
    )

    original_predictions = np.argmax(
        original_probabilities,
        axis=1,
    )

    result_rows = []

    for sample_index, original in enumerate(
        test_values
    ):
        source_row = test.iloc[sample_index]

        original_prediction = int(
            original_predictions[sample_index]
        )

        for target_class in target_classes:
            if target_class == original_prediction:
                continue

            result = search_counterfactual(
                model=model,
                original=original,
                target_class=target_class,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                scales=scales,
                discrete_indices=discrete_indices,
                observed_discrete_values=(
                    observed_discrete_values
                ),
                candidate_count=candidate_count,
                mutation_probability=mutation_probability,
                mutation_scale=mutation_scale,
                rng=rng,
            )

            row = {
                "sample_index": sample_index,
                "subject_id": source_row[
                    "subject_id"
                ],
                "acquisition_group_id": source_row[
                    "acquisition_group_id"
                ],
                "split": source_row["split"],
                "true_diagnosis": source_row[
                    "diagnosis_xml"
                ],
                "true_target": int(
                    source_row["target_label"]
                ),
                "original_prediction": original_prediction,
                "original_prediction_name": CLASS_NAMES[
                    original_prediction
                ],
                "target_class": target_class,
                "target_class_name": CLASS_NAMES[
                    target_class
                ],
                "requested_target_class": target_class,
                "requested_target_class_name": CLASS_NAMES[
                    target_class
                ],
                "found": bool(result["found"]),
                "reason": result["reason"],
                "original_target_probability": result[
                    "original_target_probability"
                ],
                "counterfactual_target_probability": result[
                    "counterfactual_target_probability"
                ],
                "distance": result["distance"],
            }

            row = add_changed_feature_columns(
                row=row,
                original=original,
                candidate=result.get("candidate"),
                feature_names=feature_names,
            )

            result_rows.append(row)

    output_dir = output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe = pd.DataFrame(
        result_rows
    )

    results_dataframe.to_csv(
        output_dir / "counterfactuals.csv",
        index=False,
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

    successful_count = int(
        results_dataframe["found"].sum()
    ) if not results_dataframe.empty else 0

    summary = {
        "model_path": str(
            Path(model_path).expanduser().resolve()
        ),
        "training_path": str(
            Path(training_path).expanduser().resolve()
        ),
        "test_path": str(
            Path(test_path).expanduser().resolve()
        ),
        "output_directory": str(output_dir),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_sample_count": len(test),
        "target_classes": target_classes,
        "candidate_count_per_target": candidate_count,
        "mutation_probability": mutation_probability,
        "mutation_scale": mutation_scale,
        "counterfactual_requests": len(
            result_rows
        ),
        "successful_counterfactual_count": (
            successful_count
        ),
        "discrete_feature_names": [
            feature_names[index]
            for index in sorted(discrete_indices)
        ],
        "bounds_fitted_on": "train",
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "raw_data_modified": False,
        "causal_interpretation": False,
        "interpretation_warning": (
            "Counterfactuals are model-based examples, not causal "
            "explanations, treatment recommendations, or clinical advice."
        ),
        "runtime_seconds": float(
            time.perf_counter()
            - start_time
        ),
    }

    with (
        output_dir / "counterfactual_summary.json"
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


# Compatibility alias for any earlier code that used the old function name.
def run_counterfactual_explanations(
    model_path: Path,
    training_path: Path,
    test_path: Path,
    output_dir: Path,
    candidate_count: int = 100,
    max_samples: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    return run_counterfactuals(
        model_path=model_path,
        training_path=training_path,
        test_path=test_path,
        output_dir=output_dir,
        candidate_count=candidate_count,
        max_samples=max_samples,
        random_seed=random_seed,
    )
