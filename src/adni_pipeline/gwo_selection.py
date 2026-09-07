

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

@dataclass
class GWOSummary:
    modality: str
    input_feature_count: int
    selected_feature_count: int
    selected_features: list[str]
    best_fitness: float
    population_size: int
    iterations: int
    cv_folds: int
    fitted_on: str
    output_directory: str

def load_split_table(
    input_root: Path,
    modality: str,
    split: str,
) -> pd.DataFrame:
    path = input_root / modality / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Input table is empty: {path}")

    required_columns = META_COLUMNS

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    return dataframe

def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No candidate feature columns found")

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise TypeError(
                f"Feature column is not numeric: {column}"
            )

    return feature_columns

def validate_subject_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    subjects_by_split: dict[str, set[str]] = {}

    for dataframe, split_name in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
        actual_split_values = set(
            dataframe["split"].astype(str)
        )

        if actual_split_values != {split_name}:
            raise ValueError(
                f"Expected only split={split_name}, found "
                f"{actual_split_values}"
            )

        for subject_id in dataframe["subject_id"].astype(str):
            subjects_by_split.setdefault(subject_id, set()).add(
                split_name
            )

    leakage = {
        subject_id: sorted(splits)
        for subject_id, splits in subjects_by_split.items()
        if len(splits) > 1
    }

    if leakage:
        raise ValueError(
            "Subject leakage detected: "
            + json.dumps(leakage, sort_keys=True)
        )

def choose_cv_folds(
    target: pd.Series,
    groups: pd.Series,
    requested_folds: int,
) -> int:
    """
    Choose a safe number of grouped CV folds.

    At least two folds are required. The number of folds cannot exceed the
    number of unique subjects or the count of the least represented class.
    """
    unique_subject_count = groups.nunique()
    class_counts = target.value_counts()

    if class_counts.empty:
        raise ValueError("Training target is empty")

    minimum_class_count = int(class_counts.min())

    folds = min(
        requested_folds,
        unique_subject_count,
        minimum_class_count,
    )

    if folds < 2:
        raise ValueError(
            "Grouped cross-validation requires at least two folds. "
            f"Subjects={unique_subject_count}, "
            f"minimum_class_count={minimum_class_count}"
        )

    return folds

def make_model() -> Pipeline:
    """
    Create the GWO fitness model.
    StandardScaler is fitted separately inside every grouped CV training
    fold. LogisticRegression automatically handles the multiclass target
    using the installed scikit-learn API.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

def evaluate_mask(
    feature_values: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    cv_folds: int,
    random_seed: int,
) -> float:
    """Evaluate one feature mask using subject-grouped CV."""
    if mask.sum() == 0:
        return 0.0

    selected_values = feature_values[:, mask]

    splitter = GroupKFold(n_splits=cv_folds)
    predictions = np.full(
        shape=len(target),
        fill_value=-1,
        dtype=int,
    )

    for train_indices, validation_indices in splitter.split(
        selected_values,
        target,
        groups,
    ):
        model = make_model()
        model.set_params(
            classifier__random_state=random_seed
        )

        model.fit(
            selected_values[train_indices],
            target[train_indices],
        )

        predictions[validation_indices] = model.predict(
            selected_values[validation_indices]
        )

    valid_predictions = predictions >= 0

    if not valid_predictions.all():
        return 0.0

    return float(
        balanced_accuracy_score(
            target,
            predictions,
        )
    )

def initialize_population(
    population_size: int,
    feature_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    population = rng.integers(
        low=0,
        high=2,
        size=(population_size, feature_count),
        dtype=np.int8,
    )

    # Ensure every wolf has at least one selected feature.
    for row_index in range(population_size):
        if population[row_index].sum() == 0:
            feature_index = rng.integers(
                0,
                feature_count,
            )
            population[row_index, feature_index] = 1

    return population

def repair_population(
    population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    repaired = population.copy()

    for row_index in range(repaired.shape[0]):
        repaired[row_index] = (
            repaired[row_index] > 0
        ).astype(np.int8)

        if repaired[row_index].sum() == 0:
            feature_index = rng.integers(
                0,
                repaired.shape[1],
            )
            repaired[row_index, feature_index] = 1

    return repaired

def update_population(
    population: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: np.ndarray,
    iteration: int,
    total_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Binary GWO population update.

    Continuous GWO positions are converted to binary masks using a sigmoid
    transfer function.
    """
    feature_count = population.shape[1]

    if total_iterations <= 1:
        coefficient_a = 0.0
    else:
        coefficient_a = 2.0 - (
            2.0 * iteration / (total_iterations - 1)
        )

    updated = np.zeros_like(population)

    for wolf_index in range(population.shape[0]):
        current = population[wolf_index].astype(float)
        candidate_positions = []

        for leader in (alpha, beta, delta):
            leader_values = leader.astype(float)

            r1 = rng.random(feature_count)
            r2 = rng.random(feature_count)

            coefficient_a_vector = (
                2.0 * coefficient_a * r1
                - coefficient_a
            )

            coefficient_c_vector = 2.0 * r2

            distance = np.abs(
                coefficient_c_vector * leader_values
                - current
            )

            candidate = (
                leader_values
                - coefficient_a_vector * distance
            )

            candidate_positions.append(candidate)

        continuous_position = np.mean(
            candidate_positions,
            axis=0,
        )

        probability = 1.0 / (
            1.0 + np.exp(
                -np.clip(
                    continuous_position,
                    -30.0,
                    30.0,
                )
            )
        )

        updated[wolf_index] = (
            rng.random(feature_count) < probability
        ).astype(np.int8)

    return repair_population(updated, rng)

def run_gwo(
    feature_values: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    population_size: int,
    iterations: int,
    cv_folds: int,
    random_seed: int,
) -> tuple[list[str], float, dict[str, Any]]:
    """Run binary Grey Wolf Optimizer feature selection."""
    if population_size < 3:
        raise ValueError(
            "population_size must be at least 3"
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be at least 1"
        )

    rng = np.random.default_rng(random_seed)

    population = initialize_population(
        population_size=population_size,
        feature_count=len(feature_names),
        rng=rng,
    )

    alpha_mask = population[0].copy()
    beta_mask = population[1].copy()
    delta_mask = population[2].copy()

    alpha_score = -np.inf
    beta_score = -np.inf
    delta_score = -np.inf

    history = []

    for iteration in range(iterations):
        scores = []

        for wolf in population:
            score = evaluate_mask(
                feature_values=feature_values,
                target=target,
                groups=groups,
                mask=wolf.astype(bool),
                cv_folds=cv_folds,
                random_seed=random_seed,
            )

            scores.append(score)

        scores_array = np.asarray(scores)

        ranking = np.argsort(
            scores_array
        )[::-1]

        current_alpha_index = ranking[0]
        current_beta_index = ranking[1]
        current_delta_index = ranking[2]

        current_alpha_score = scores_array[
            current_alpha_index
        ]
        current_beta_score = scores_array[
            current_beta_index
        ]
        current_delta_score = scores_array[
            current_delta_index
        ]

        if current_alpha_score > alpha_score:
            alpha_score = float(current_alpha_score)
            alpha_mask = population[
                current_alpha_index
            ].copy()

        if current_beta_score > beta_score:
            beta_score = float(current_beta_score)
            beta_mask = population[
                current_beta_index
            ].copy()

        if current_delta_score > delta_score:
            delta_score = float(current_delta_score)
            delta_mask = population[
                current_delta_index
            ].copy()

        history.append(
            {
                "iteration": iteration + 1,
                "best_iteration_fitness": float(
                    current_alpha_score
                ),
                "best_global_fitness": float(
                    alpha_score
                ),
                "best_feature_count": int(
                    alpha_mask.sum()
                ),
            }
        )

        population = update_population(
            population=population,
            alpha=alpha_mask,
            beta=beta_mask,
            delta=delta_mask,
            iteration=iteration,
            total_iterations=iterations,
            rng=rng,
        )

    selected_features = [
        feature_name
        for feature_name, selected in zip(
            feature_names,
            alpha_mask,
        )
        if selected
    ]

    if not selected_features:
        raise RuntimeError(
            "GWO returned an empty feature subset"
        )

    details = {
        "history": history,
        "best_mask": alpha_mask.tolist(),
        "best_fitness": float(alpha_score),
    }

    return selected_features, float(alpha_score), details

def save_selected_tables(
    output_directory: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_columns = [
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "target_label",
    ]

    columns = metadata_columns + selected_features

    train[columns].to_csv(
        output_directory / "train.csv",
        index=False,
    )

    validation[columns].to_csv(
        output_directory / "validation.csv",
        index=False,
    )

    test[columns].to_csv(
        output_directory / "test.csv",
        index=False,
    )

def run_gwo_selection(
    input_root: Path,
    output_root: Path,
    modality: str,
    population_size: int,
    iterations: int,
    requested_cv_folds: int,
    random_seed: int,
) -> dict[str, Any]:
    train = load_split_table(
        input_root=input_root,
        modality=modality,
        split="train",
    )

    validation = load_split_table(
        input_root=input_root,
        modality=modality,
        split="validation",
    )

    test = load_split_table(
        input_root=input_root,
        modality=modality,
        split="test",
    )

    validate_subject_splits(
        train=train,
        validation=validation,
        test=test,
    )

    feature_names = get_feature_columns(train)

    if get_feature_columns(validation) != feature_names:
        raise ValueError(
            "Train and validation feature columns differ"
        )

    if get_feature_columns(test) != feature_names:
        raise ValueError(
            "Train and test feature columns differ"
        )

    train_values = train[feature_names].to_numpy(
        dtype=np.float64
    )

    target = train["target_label"].to_numpy(
        dtype=int
    )

    groups = train["subject_id"].astype(str).to_numpy()

    if not np.isfinite(train_values).all():
        raise ValueError(
            "Training features contain NaN or infinite values"
        )

    cv_folds = choose_cv_folds(
        target=train["target_label"],
        groups=train["subject_id"],
        requested_folds=requested_cv_folds,
    )

    selected_features, best_fitness, details = run_gwo(
        feature_values=train_values,
        target=target,
        groups=groups,
        feature_names=feature_names,
        population_size=population_size,
        iterations=iterations,
        cv_folds=cv_folds,
        random_seed=random_seed,
    )

    output_directory = output_root / modality

    save_selected_tables(
        output_directory=output_directory,
        train=train,
        validation=validation,
        test=test,
        selected_features=selected_features,
    )

    with (
        output_directory / "selected_features.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            selected_features,
            handle,
            indent=2,
        )

    joblib.dump(
        {
            "selected_features": selected_features,
            "input_features": feature_names,
            "fitted_on": "train",
            "population_size": population_size,
            "iterations": iterations,
            "cv_folds": cv_folds,
            "random_seed": random_seed,
        },
        output_directory / "gwo_selection.joblib",
    )

    with (
        output_directory / "gwo_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            details["history"],
            handle,
            indent=2,
        )

    summary = {
        "modality": modality,
        "input_feature_count": len(feature_names),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "best_fitness_balanced_accuracy": best_fitness,
        "population_size": population_size,
        "iterations": iterations,
        "cv_folds": cv_folds,
        "fitness_model": (
            "StandardScaler plus class-weighted LogisticRegression"
        ),
        "fitness_split": "grouped cross-validation within train",
        "group_column": "subject_id",
        "target_column": "target_label",
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "fitted_on": "train",
        "output_directory": str(
            output_directory.resolve()
        ),
    }

    with (
        output_directory / "gwo_selection_summary.json"
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
