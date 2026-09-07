from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
}

VALID_SPLITS = {"train", "validation", "test"}

@dataclass
class FeaturePreprocessingResult:
    modality: str
    feature_count: int
    train_record_count: int
    validation_record_count: int
    test_record_count: int
    train_subject_count: int
    validation_subject_count: int
    test_subject_count: int
    output_directory: str
    feature_names_path: str
    imputer_path: str
    scaler_path: str
    summary_path: str

def load_multimodal_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Feature table is empty: {path}"
        )

    required_columns = META_COLUMNS

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return dataframe

def validate_table_identity(dataframe: pd.DataFrame) -> None:
    """Validate metadata and subject-level split integrity."""
    invalid_splits = sorted(
        set(dataframe["split"].astype(str))
        - VALID_SPLITS
    )

    if invalid_splits:
        raise ValueError(
            "Invalid split values: "
            + ", ".join(invalid_splits)
        )

    if dataframe["subject_id"].isna().any():
        raise ValueError("Feature table contains missing subject IDs")

    if dataframe["acquisition_group_id"].isna().any():
        raise ValueError(
            "Feature table contains missing acquisition group IDs"
        )

    duplicate_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "acquisition_group_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = dataframe.loc[
            duplicate_mask,
            [
                "subject_id",
                "acquisition_group_id",
            ],
        ]

        raise ValueError(
            "Feature table contains duplicate acquisition keys:\n"
            + duplicate_rows.to_string(index=False)
        )

    split_counts = (
        dataframe.groupby("subject_id")["split"]
        .nunique()
    )

    leaking_subjects = split_counts[split_counts > 1]

    if not leaking_subjects.empty:
        raise ValueError(
            "Subject-level split leakage detected for subjects: "
            + ", ".join(leaking_subjects.index.astype(str))
        )

    target_mapping = {
        "CN": 0,
        "MCI": 1,
        "AD": 2,
    }

    for index, row in dataframe.iterrows():
        diagnosis = str(
            row["diagnosis_xml"]
        ).strip().upper()

        if diagnosis not in target_mapping:
            raise ValueError(
                f"Invalid diagnosis at row {index}: {diagnosis}"
            )

        target = row["target_label"]

        if pd.isna(target):
            raise ValueError(
                f"Missing target_label at row {index}"
            )

        if int(float(target)) != target_mapping[diagnosis]:
            raise ValueError(
                f"Diagnosis/target mismatch at row {index}: "
                f"{diagnosis} versus {target}"
            )

def get_feature_columns(
    dataframe: pd.DataFrame,
    modality: str,
) -> list[str]:
    """
    Return numeric model feature columns.

    Clinical columns use the `clin_` prefix.

    Radiomics columns use `brain_` and `subcortical_` prefixes.

    Multimodal columns use both clinical and radiomics prefixes.
    """
    if modality == "clinical":
        prefixes = ("clin_",)

    elif modality == "radiomics":
        prefixes = (
            "brain_",
            "subcortical_",
        )

    elif modality == "multimodal":
        prefixes = (
            "clin_",
            "brain_",
            "subcortical_",
        )

    else:
        raise ValueError(
            "modality must be one of: clinical, radiomics, multimodal"
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(prefixes)
    ]

    if not feature_columns:
        raise ValueError(
            f"No feature columns found for modality={modality}"
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
            "Feature columns must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric_columns)
        )

    return sorted(feature_columns)

def make_metadata_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return dataframe[
        [
            "subject_id",
            "acquisition_group_id",
            "split",
            "diagnosis_xml",
            "target_label",
        ]
    ].copy()

def fit_train_only_transformer(
    train_features: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler using training features only.
    """
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    imputed_train = imputer.fit_transform(train_features)
    scaler.fit(imputed_train)

    return imputer, scaler

def transform_features(
    features: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    imputed = imputer.transform(features)
    scaled = scaler.transform(imputed)

    scaled = np.asarray(scaled, dtype=np.float64)

    if not np.isfinite(scaled).all():
        raise ValueError(
            "Transformed features contain NaN or infinite values"
        )

    return scaled

def save_split_table(
    output_dir: Path,
    split_name: str,
    metadata: pd.DataFrame,
    transformed_features: np.ndarray,
    feature_names: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dataframe = pd.DataFrame(
        transformed_features,
        columns=feature_names,
    )

    output_dataframe = pd.concat(
        [
            metadata.reset_index(drop=True),
            feature_dataframe.reset_index(drop=True),
        ],
        axis=1,
    )

    output_path = output_dir / f"{split_name}.csv"
    output_dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path

def process_modality(
    dataframe: pd.DataFrame,
    modality: str,
    output_root: Path,
) -> FeaturePreprocessingResult:
    feature_columns = get_feature_columns(
        dataframe=dataframe,
        modality=modality,
    )

    train_mask = dataframe["split"].astype(str) == "train"
    validation_mask = (
        dataframe["split"].astype(str) == "validation"
    )
    test_mask = dataframe["split"].astype(str) == "test"

    if not train_mask.any():
        raise ValueError(
            f"No training records found for modality={modality}"
        )

    if not validation_mask.any():
        raise ValueError(
            f"No validation records found for modality={modality}"
        )

    if not test_mask.any():
        raise ValueError(
            f"No test records found for modality={modality}"
        )

    train_features = dataframe.loc[
        train_mask,
        feature_columns,
    ].copy()

    validation_features = dataframe.loc[
        validation_mask,
        feature_columns,
    ].copy()

    test_features = dataframe.loc[
        test_mask,
        feature_columns,
    ].copy()

    imputer, scaler = fit_train_only_transformer(
        train_features=train_features,
    )

    train_transformed = transform_features(
        train_features,
        imputer,
        scaler,
    )

    validation_transformed = transform_features(
        validation_features,
        imputer,
        scaler,
    )

    test_transformed = transform_features(
        test_features,
        imputer,
        scaler,
    )

    modality_dir = output_root / modality
    modality_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_train = make_metadata_table(
        dataframe.loc[train_mask]
    )

    metadata_validation = make_metadata_table(
        dataframe.loc[validation_mask]
    )

    metadata_test = make_metadata_table(
        dataframe.loc[test_mask]
    )

    save_split_table(
        output_dir=modality_dir,
        split_name="train",
        metadata=metadata_train,
        transformed_features=train_transformed,
        feature_names=feature_columns,
    )

    save_split_table(
        output_dir=modality_dir,
        split_name="validation",
        metadata=metadata_validation,
        transformed_features=validation_transformed,
        feature_names=feature_columns,
    )

    save_split_table(
        output_dir=modality_dir,
        split_name="test",
        metadata=metadata_test,
        transformed_features=test_transformed,
        feature_names=feature_columns,
    )

    feature_names_path = modality_dir / "feature_names.json"

    with feature_names_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            feature_columns,
            handle,
            indent=2,
        )

    imputer_path = modality_dir / "imputer.joblib"
    scaler_path = modality_dir / "scaler.joblib"

    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)

    summary = {
        "modality": modality,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "record_counts": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "subject_counts": {
            "train": int(
                dataframe.loc[train_mask, "subject_id"].nunique()
            ),
            "validation": int(
                dataframe.loc[
                    validation_mask,
                    "subject_id",
                ].nunique()
            ),
            "test": int(
                dataframe.loc[test_mask, "subject_id"].nunique()
            ),
        },
        "fitted_on_split": "train",
        "imputation_strategy": "median",
        "scaling": "StandardScaler",
        "metadata_excluded_from_features": sorted(
            META_COLUMNS
        ),
    }

    summary_path = modality_dir / "preprocessing_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    return FeaturePreprocessingResult(
        modality=modality,
        feature_count=len(feature_columns),
        train_record_count=int(train_mask.sum()),
        validation_record_count=int(
            validation_mask.sum()
        ),
        test_record_count=int(test_mask.sum()),
        train_subject_count=int(
            dataframe.loc[
                train_mask,
                "subject_id",
            ].nunique()
        ),
        validation_subject_count=int(
            dataframe.loc[
                validation_mask,
                "subject_id",
            ].nunique()
        ),
        test_subject_count=int(
            dataframe.loc[
                test_mask,
                "subject_id",
            ].nunique()
        ),
        output_directory=str(modality_dir.resolve()),
        feature_names_path=str(
            feature_names_path.resolve()
        ),
        imputer_path=str(imputer_path.resolve()),
        scaler_path=str(scaler_path.resolve()),
        summary_path=str(summary_path.resolve()),
    )

def run_feature_preprocessing(
    multimodal_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    dataframe = load_multimodal_table(
        multimodal_path
    )

    validate_table_identity(dataframe)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for modality in (
        "clinical",
        "radiomics",
        "multimodal",
    ):
        result = process_modality(
            dataframe=dataframe,
            modality=modality,
            output_root=output_root,
        )

        results.append(result)

    summary = {
        "source_table": str(
            multimodal_path.resolve()
        ),
        "output_root": str(
            output_root.resolve()
        ),
        "modalities": {
            result.modality: {
                "feature_count": result.feature_count,
                "train_record_count": (
                    result.train_record_count
                ),
                "validation_record_count": (
                    result.validation_record_count
                ),
                "test_record_count": (
                    result.test_record_count
                ),
                "train_subject_count": (
                    result.train_subject_count
                ),
                "validation_subject_count": (
                    result.validation_subject_count
                ),
                "test_subject_count": (
                    result.test_subject_count
                ),
            }
            for result in results
        },
        "fitting_rule": (
            "Imputer and scaler were fitted on training records only."
        ),
        "subject_id_used_as_feature": False,
        "target_used_as_feature": False,
        "raw_data_modified": False,
    }

    summary_path = output_root / "feature_preprocessing_summary.json"

    with summary_path.open(
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
