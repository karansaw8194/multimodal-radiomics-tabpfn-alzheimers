from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

@dataclass
class BaselineSelectionResult:
    modality: str
    input_feature_count: int
    variance_kept_feature_count: int
    selected_feature_count: int
    selected_features_path: str
    selector_path: str
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

    required = META_COLUMNS

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            + ", ".join(sorted(missing))
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

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric:
        raise TypeError(
            "All candidate features must be numeric. "
            "Non-numeric columns: "
            + ", ".join(non_numeric)
        )

    return feature_columns

def validate_metadata_alignment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for dataframe, split_name in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
        if dataframe["split"].astype(str).nunique() != 1:
            raise ValueError(
                f"{split_name} table contains multiple split values"
            )

        if dataframe["split"].iloc[0] != split_name:
            raise ValueError(
                f"Expected split={split_name}, found "
                f"{dataframe['split'].iloc[0]}"
            )

    subjects_by_split = {}

    for dataframe, split_name in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
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

def fit_baseline_selector(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    top_k: int,
    variance_threshold: float,
) -> tuple[VarianceThreshold, SelectKBest, list[str]]:
    """
    Fit both feature-selection stages using training data only.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if variance_threshold < 0:
        raise ValueError(
            "variance_threshold cannot be negative"
        )

    variance_selector = VarianceThreshold(
        threshold=variance_threshold
    )

    variance_selector.fit(train_features)

    variance_mask = variance_selector.get_support()
    variance_columns = list(
        train_features.columns[variance_mask]
    )

    if not variance_columns:
        raise ValueError(
            "Variance filtering removed every feature"
        )

    variance_train = variance_selector.transform(
        train_features
    )

    effective_k = min(top_k, variance_train.shape[1])

    kbest_selector = SelectKBest(
        score_func=f_classif,
        k=effective_k,
    )

    kbest_selector.fit(
        variance_train,
        train_target,
    )

    selected_mask = kbest_selector.get_support()
    selected_features = [
        column
        for column, selected in zip(
            variance_columns,
            selected_mask,
        )
        if selected
    ]

    if not selected_features:
        raise ValueError(
            "ANOVA selection returned no features"
        )

    return (
        variance_selector,
        kbest_selector,
        selected_features,
    )

def transform_with_selectors(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    variance_selector: VarianceThreshold,
    kbest_selector: SelectKBest,
) -> pd.DataFrame:
    """
    Apply training-fitted selectors to any split.
    """
    feature_values = dataframe[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Input features contain NaN or infinite values"
        )

    variance_values = variance_selector.transform(
        feature_values
    )

    selected_values = kbest_selector.transform(
        variance_values
    )

    selected_indices = kbest_selector.get_support(indices=True)

    variance_columns = list(
        np.asarray(feature_columns)[
            variance_selector.get_support()
        ]
    )

    selected_columns = [
        variance_columns[index]
        for index in selected_indices
    ]

    result = pd.DataFrame(
        selected_values,
        columns=selected_columns,
        index=dataframe.index,
    )

    metadata = dataframe[
        [
            "subject_id",
            "acquisition_group_id",
            "split",
            "diagnosis_xml",
            "target_label",
        ]
    ].copy()

    return pd.concat(
        [
            metadata,
            result,
        ],
        axis=1,
    )

def run_baseline_selection(
    input_root: Path,
    output_root: Path,
    modality: str,
    top_k: int,
    variance_threshold: float,
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

    validate_metadata_alignment(
        train=train,
        validation=validation,
        test=test,
    )

    feature_columns = get_feature_columns(train)

    validation_feature_columns = get_feature_columns(validation)
    test_feature_columns = get_feature_columns(test)

    if feature_columns != validation_feature_columns:
        raise ValueError(
            "Train and validation feature columns differ"
        )

    if feature_columns != test_feature_columns:
        raise ValueError(
            "Train and test feature columns differ"
        )

    train_target = train["target_label"].astype(int)

    variance_selector, kbest_selector, selected_features = (
        fit_baseline_selector(
            train_features=train[feature_columns],
            train_target=train_target,
            top_k=top_k,
            variance_threshold=variance_threshold,
        )
    )

    transformed_train = transform_with_selectors(
        dataframe=train,
        feature_columns=feature_columns,
        variance_selector=variance_selector,
        kbest_selector=kbest_selector,
    )

    transformed_validation = transform_with_selectors(
        dataframe=validation,
        feature_columns=feature_columns,
        variance_selector=variance_selector,
        kbest_selector=kbest_selector,
    )

    transformed_test = transform_with_selectors(
        dataframe=test,
        feature_columns=feature_columns,
        variance_selector=variance_selector,
        kbest_selector=kbest_selector,
    )

    modality_output = output_root / modality
    modality_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    transformed_train.to_csv(
        modality_output / "train.csv",
        index=False,
    )

    transformed_validation.to_csv(
        modality_output / "validation.csv",
        index=False,
    )

    transformed_test.to_csv(
        modality_output / "test.csv",
        index=False,
    )

    selected_features_path = (
        modality_output / "selected_features.json"
    )

    with selected_features_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            selected_features,
            handle,
            indent=2,
        )

    selector_path = modality_output / "selectors.joblib"

    joblib.dump(
        {
            "variance_selector": variance_selector,
            "kbest_selector": kbest_selector,
            "input_feature_names": feature_columns,
            "selected_feature_names": selected_features,
            "fitted_on": "train",
        },
        selector_path,
    )

    variance_kept_count = int(
        variance_selector.get_support().sum()
    )

    summary = {
        "modality": modality,
        "input_feature_count": len(feature_columns),
        "variance_kept_feature_count": variance_kept_count,
        "selected_feature_count": len(selected_features),
        "requested_top_k": top_k,
        "effective_top_k": len(selected_features),
        "variance_threshold": variance_threshold,
        "fitted_on_split": "train",
        "selected_features": selected_features,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "output_directory": str(
            modality_output.resolve()
        ),
    }

    with (
        modality_output / "selection_summary.json"
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
