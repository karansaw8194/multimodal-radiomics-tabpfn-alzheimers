from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .gwo_selection import (
    META_COLUMNS,
    choose_cv_folds,
    evaluate_mask,
    get_feature_columns,
    load_split_table,
    save_selected_tables,
    validate_subject_splits,
)

def initialize_hawks(
    population_size: int,
    feature_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a binary hawk population."""
    population = rng.integers(
        low=0,
        high=2,
        size=(population_size, feature_count),
        dtype=np.int8,
    )

    for index in range(population_size):
        if population[index].sum() == 0:
            selected_index = rng.integers(
                low=0,
                high=feature_count,
            )
            population[index, selected_index] = 1

    return population

def repair_hawks(
    population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ensure every hawk contains at least one selected feature."""
    repaired = (population > 0).astype(np.int8)

    for index in range(repaired.shape[0]):
        if repaired[index].sum() == 0:
            selected_index = rng.integers(
                low=0,
                high=repaired.shape[1],
            )
            repaired[index, selected_index] = 1

    return repaired

def binary_position(
    continuous_position: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert continuous HHO positions to a binary feature mask using a
    sigmoid transfer function.
    """
    clipped = np.clip(
        continuous_position,
        -30.0,
        30.0,
    )

    probability = 1.0 / (1.0 + np.exp(-clipped))

    return (
        rng.random(len(probability)) < probability
    ).astype(np.int8)

def update_hawks(
    population: np.ndarray,
    rabbit: np.ndarray,
    iteration: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Binary Harris Hawks update.

    The continuous HHO exploration and exploitation equations are followed
    by sigmoid binarization for feature-subset selection.
    """
    hawk_count, feature_count = population.shape
    updated = np.zeros_like(population)

    if iterations <= 1:
        progress = 1.0
    else:
        progress = iteration / float(iterations - 1)

    for hawk_index in range(hawk_count):
        current = population[hawk_index].astype(float)
        rabbit_position = rabbit.astype(float)

        escaping_energy = (
            2.0
            * (1.0 - progress)
            * (2.0 * rng.random() - 1.0)
        )

        escaping_abs = abs(escaping_energy)
        q = rng.random()
        r = rng.random()

        if escaping_abs >= 1.0:
            if q < 0.5:
                random_hawk = population[
                    rng.integers(0, hawk_count)
                ].astype(float)

                updated_position = (
                    random_hawk
                    - rng.random(feature_count)
                    * np.abs(
                        random_hawk
                        - 2.0
                        * rng.random()
                        * current
                    )
                )
            else:
                mean_position = np.mean(
                    population.astype(float),
                    axis=0,
                )

                updated_position = (
                    rabbit_position
                    - mean_position
                    - rng.random(feature_count)
                    * (
                        0.5
                        + rng.random()
                        * (
                            np.max(population)
                            - np.min(population)
                        )
                    )
                )

        else:
            jump_strength = (
                2.0 * (1.0 - rng.random())
            )

            if r >= 0.5 and escaping_abs >= 0.5:
                updated_position = (
                    rabbit_position
                    - escaping_energy
                    * np.abs(
                        rabbit_position - current
                    )
                )

            elif r >= 0.5 and escaping_abs < 0.5:
                updated_position = (
                    rabbit_position
                    - escaping_energy
                    * np.abs(
                        rabbit_position - rabbit_position
                    )
                )

            elif r < 0.5 and escaping_abs >= 0.5:
                random_jump = (
                    jump_strength * rabbit_position
                    - current
                )

                updated_position = (
                    random_jump
                    - rng.random(feature_count)
                    * np.abs(random_jump)
                )

            else:
                random_jump = (
                    jump_strength * rabbit_position
                    - current
                )

                updated_position = (
                    rabbit_position
                    - escaping_energy
                    * np.abs(random_jump)
                )

        updated[hawk_index] = binary_position(
            updated_position,
            rng,
        )

    return repair_hawks(updated, rng)

def run_hho(
    feature_values: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    population_size: int,
    iterations: int,
    cv_folds: int,
    random_seed: int,
) -> tuple[list[str], float, dict[str, Any]]:
    """Run binary HHO feature selection."""
    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2"
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be at least 1"
        )

    rng = np.random.default_rng(random_seed)

    population = initialize_hawks(
        population_size=population_size,
        feature_count=len(feature_names),
        rng=rng,
    )

    rabbit_mask = population[0].copy()
    rabbit_score = -np.inf
    history = []

    for iteration in range(iterations):
        scores = []

        for hawk in population:
            score = evaluate_mask(
                feature_values=feature_values,
                target=target,
                groups=groups,
                mask=hawk.astype(bool),
                cv_folds=cv_folds,
                random_seed=random_seed,
            )

            scores.append(score)

        scores_array = np.asarray(scores)

        best_index = int(np.argmax(scores_array))
        best_score = float(scores_array[best_index])

        if best_score > rabbit_score:
            rabbit_score = best_score
            rabbit_mask = population[best_index].copy()

        history.append(
            {
                "iteration": iteration + 1,
                "best_iteration_fitness": best_score,
                "best_global_fitness": float(rabbit_score),
                "best_feature_count": int(
                    rabbit_mask.sum()
                ),
            }
        )

        population = update_hawks(
            population=population,
            rabbit=rabbit_mask,
            iteration=iteration,
            iterations=iterations,
            rng=rng,
        )

    selected_features = [
        feature_name
        for feature_name, selected in zip(
            feature_names,
            rabbit_mask,
        )
        if selected
    ]

    if not selected_features:
        raise RuntimeError(
            "HHO returned an empty feature subset"
        )

    details = {
        "history": history,
        "best_mask": rabbit_mask.tolist(),
        "best_fitness": float(rabbit_score),
    }

    return selected_features, float(rabbit_score), details

def run_hho_selection(
    input_root: Path,
    output_root: Path,
    modality: str,
    population_size: int,
    iterations: int,
    requested_cv_folds: int,
    random_seed: int,
) -> dict[str, Any]:
    """Run HHO using only the training partition."""
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

    if not np.isfinite(train_values).all():
        raise ValueError(
            "Training features contain NaN or infinite values"
        )

    target = train["target_label"].to_numpy(
        dtype=int
    )

    groups = train["subject_id"].astype(str).to_numpy()

    cv_folds = choose_cv_folds(
        target=train["target_label"],
        groups=train["subject_id"],
        requested_folds=requested_cv_folds,
    )

    selected_features, best_fitness, details = run_hho(
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

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
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
        output_directory / "hho_selection.joblib",
    )

    with (
        output_directory / "hho_history.json"
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
        "fitness_split": (
            "grouped cross-validation within train"
        ),
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
        output_directory / "hho_selection_summary.json"
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
