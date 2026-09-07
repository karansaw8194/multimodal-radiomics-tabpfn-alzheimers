from __future__ import annotations

import argparse
import json
import sys

from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


KEY_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
}


TARGET_COLUMNS = {
    "diagnosis_xml",
    "target_label",
}


FORBIDDEN_FEATURE_COLUMNS = {
    "subject_id",
    "acquisition_group_id",
    "split",
    "diagnosis_xml",
    "target_label",
    "diagnosis_folder",
    "folder_class",
    "xml_path",
    "nifti_path",
    "raw_nifti_path",
    "preprocessed_nifti_path",
    "multiclass_mask_path",
    "brain_roi_mask_path",
    "subcortical_roi_mask_path",
}


def load_table(
    path: Path,
    table_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{table_name} does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"{table_name} is empty: {path}"
        )

    return dataframe


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )


def check_duplicate_keys(
    dataframe: pd.DataFrame,
    table_name: str,
) -> list[str]:
    duplicate_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "acquisition_group_id",
        ],
        keep=False,
    )

    duplicate_rows = dataframe.loc[duplicate_mask]

    if duplicate_rows.empty:
        return []

    duplicate_keys = (
        duplicate_rows[
            [
                "subject_id",
                "acquisition_group_id",
            ]
        ]
        .drop_duplicates()
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )

    return [
        (
            f"{table_name} has duplicate acquisition key: "
            f"{duplicate_key}"
        )
        for duplicate_key in duplicate_keys
    ]


def check_subject_split_consistency(
    dataframe: pd.DataFrame,
    table_name: str,
) -> list[str]:
    problems = []

    split_sets = (
        dataframe.groupby("subject_id")["split"]
        .apply(
            lambda values: sorted(
                set(values.astype(str))
            )
        )
    )

    for subject_id, splits in split_sets.items():
        if len(splits) != 1:
            problems.append(
                (
                    f"{table_name}: subject {subject_id} occurs "
                    f"in multiple splits: {splits}"
                )
            )

    return problems


def check_split_subject_overlap(
    dataframe: pd.DataFrame,
) -> list[str]:
    subjects_by_split: dict[str, set[str]] = defaultdict(set)

    for _, row in dataframe.iterrows():
        split = str(row["split"])
        subject_id = str(row["subject_id"])

        subjects_by_split[split].add(subject_id)

    problems = []

    split_names = sorted(subjects_by_split)

    for index, split_a in enumerate(split_names):
        for split_b in split_names[index + 1:]:
            overlap = sorted(
                subjects_by_split[split_a]
                & subjects_by_split[split_b]
            )

            if overlap:
                problems.append(
                    (
                        f"Subject overlap between {split_a} and "
                        f"{split_b}: "
                        + "|".join(overlap)
                    )
                )

    return problems


def check_target_consistency(
    dataframe: pd.DataFrame,
    table_name: str,
) -> list[str]:
    problems = []

    expected_targets = {
        "CN": 0,
        "MCI": 1,
        "AD": 2,
    }

    for index, row in dataframe.iterrows():
        diagnosis = str(
            row["diagnosis_xml"]
        ).strip().upper()

        if diagnosis not in expected_targets:
            problems.append(
                (
                    f"{table_name} row {index}: invalid "
                    f"diagnosis {diagnosis!r}"
                )
            )
            continue

        target_value = row["target_label"]

        if pd.isna(target_value):
            problems.append(
                (
                    f"{table_name} row {index}: missing "
                    "target_label"
                )
            )
            continue

        try:
            target_value = int(float(target_value))
        except (TypeError, ValueError):
            problems.append(
                (
                    f"{table_name} row {index}: target_label "
                    f"is not numeric: {target_value!r}"
                )
            )
            continue

        if target_value != expected_targets[diagnosis]:
            problems.append(
                (
                    f"{table_name} row {index}: diagnosis="
                    f"{diagnosis}, target_label={target_value}"
                )
            )

    return problems


def check_feature_columns(
    dataframe: pd.DataFrame,
    table_name: str,
) -> tuple[list[str], list[str]]:
    """
    Validate candidate feature columns.

    Metadata columns are allowed to exist in the table. They are only
    forbidden when accidentally included as model feature columns.
    Since this validator does not receive a separate feature list,
    it treats known metadata columns as reserved and checks suspicious
    additional columns.
    """

    reserved_columns = KEY_COLUMNS | TARGET_COLUMNS

    candidate_features = [
        column
        for column in dataframe.columns
        if column not in reserved_columns
    ]

    accidental_metadata_columns = sorted(
        set(candidate_features)
        & FORBIDDEN_FEATURE_COLUMNS
    )

    suspicious_terms = (
        "subject",
        "diagnosis",
        "target",
        "split",
        "acquisition",
        "image_id",
        "study_id",
        "series_id",
        "nifti",
        "xml",
        "mask_path",
        "preprocessed_path",
    )

    suspicious_feature_names = []

    for column in candidate_features:
        normalized = column.lower()

        if any(
            term in normalized
            for term in suspicious_terms
        ):
            suspicious_feature_names.append(column)

    if accidental_metadata_columns:
        return (
            accidental_metadata_columns,
            sorted(suspicious_feature_names),
        )

    return [], sorted(suspicious_feature_names)


def make_key_set(
    dataframe: pd.DataFrame,
) -> set[tuple[str, str]]:
    return {
        (
            str(row["subject_id"]),
            str(row["acquisition_group_id"]),
        )
        for _, row in dataframe.iterrows()
    }


def normalize_target_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "subject_id",
        "acquisition_group_id",
        "diagnosis_xml",
        "target_label",
    ]

    result = dataframe[columns].copy()

    result["subject_id"] = (
        result["subject_id"].astype(str)
    )

    result["acquisition_group_id"] = (
        result["acquisition_group_id"].astype(str)
    )

    result["diagnosis_xml"] = (
        result["diagnosis_xml"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result["target_label"] = pd.to_numeric(
        result["target_label"],
        errors="coerce",
    )

    result = result.sort_values(
        [
            "subject_id",
            "acquisition_group_id",
        ]
    ).reset_index(drop=True)

    return result


def validate_feature_tables(
    clinical_path: Path,
    radiomics_path: Path,
    multimodal_path: Path,
    output_path: Path,
) -> dict:

    clinical = load_table(
        clinical_path,
        "Clinical table",
    )

    radiomics = load_table(
        radiomics_path,
        "Radiomics table",
    )

    multimodal = load_table(
        multimodal_path,
        "Multimodal table",
    )

    # Clinical and multimodal tables require target_label.
    # Radiomics only requires diagnosis_xml.
    require_columns(
        clinical,
        KEY_COLUMNS | TARGET_COLUMNS,
        "Clinical table",
    )

    require_columns(
        radiomics,
        KEY_COLUMNS | {"diagnosis_xml"},
        "Radiomics table",
    )

    require_columns(
        multimodal,
        KEY_COLUMNS | TARGET_COLUMNS,
        "Multimodal table",
    )

    problems: list[str] = []
    warnings: list[str] = []

    # Radiomics is intentionally excluded because it does not
    # require target_label.
    table_items = (
        (clinical, "Clinical table"),
        (multimodal, "Multimodal table"),
    )

    for dataframe, table_name in table_items:

        problems.extend(
            check_duplicate_keys(
                dataframe,
                table_name,
            )
        )

        problems.extend(
            check_subject_split_consistency(
                dataframe,
                table_name,
            )
        )

        problems.extend(
            check_target_consistency(
                dataframe,
                table_name,
            )
        )

        forbidden, suspicious = check_feature_columns(
            dataframe,
            table_name,
        )

        if forbidden:
            problems.append(
                (
                    f"{table_name} contains reserved metadata "
                    f"columns among candidate features: {forbidden}"
                )
            )

        if suspicious:
            warnings.append(
                (
                    f"{table_name} contains suspicious feature "
                    f"names requiring review: {suspicious}"
                )
            )

    problems.extend(
        check_split_subject_overlap(multimodal)
    )

    clinical_keys = make_key_set(clinical)
    radiomics_keys = make_key_set(radiomics)
    multimodal_keys = make_key_set(multimodal)

    if clinical_keys != multimodal_keys:
        problems.append(
            "Clinical and multimodal acquisition keys do not match"
        )

    if radiomics_keys != multimodal_keys:
        problems.append(
            "Radiomics and multimodal acquisition keys do not match"
        )

    clinical_targets = normalize_target_table(
        clinical
    )

    multimodal_targets = normalize_target_table(
        multimodal
    )

    if not clinical_targets.equals(
        multimodal_targets
    ):
        problems.append(
            "Clinical and multimodal target values are inconsistent"
        )

    # Radiomics does not require target_label.
    # Compare only diagnosis_xml against clinical data.
    radiomics_diagnosis = radiomics[
        [
            "subject_id",
            "acquisition_group_id",
            "diagnosis_xml",
        ]
    ].copy()

    radiomics_diagnosis["subject_id"] = (
        radiomics_diagnosis[
            "subject_id"
        ].astype(str)
    )

    radiomics_diagnosis["acquisition_group_id"] = (
        radiomics_diagnosis[
            "acquisition_group_id"
        ].astype(str)
    )

    radiomics_diagnosis["diagnosis_xml"] = (
        radiomics_diagnosis[
            "diagnosis_xml"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    radiomics_diagnosis = (
        radiomics_diagnosis
        .sort_values(
            [
                "subject_id",
                "acquisition_group_id",
            ]
        )
        .reset_index(drop=True)
    )

    clinical_diagnosis = clinical_targets[
        [
            "subject_id",
            "acquisition_group_id",
            "diagnosis_xml",
        ]
    ].copy()

    if not radiomics_diagnosis.equals(
        clinical_diagnosis
    ):
        problems.append(
            "Radiomics and clinical diagnosis values are inconsistent"
        )

    split_counts = (
        multimodal.groupby("split")["subject_id"]
        .nunique()
        .to_dict()
    )

    record_counts = (
        multimodal.groupby("split")
        .size()
        .to_dict()
    )

    clinical_feature_columns = [
        column
        for column in clinical.columns
        if column.startswith("clin_")
    ]

    radiomics_feature_columns = [
        column
        for column in radiomics.columns
        if column not in FORBIDDEN_FEATURE_COLUMNS
    ]

    multimodal_feature_columns = [
        column
        for column in multimodal.columns
        if column not in FORBIDDEN_FEATURE_COLUMNS
    ]

    summary = {
        "clinical_path": str(
            clinical_path.resolve()
        ),
        "radiomics_path": str(
            radiomics_path.resolve()
        ),
        "multimodal_path": str(
            multimodal_path.resolve()
        ),
        "clinical_record_count": int(
            len(clinical)
        ),
        "radiomics_record_count": int(
            len(radiomics)
        ),
        "multimodal_record_count": int(
            len(multimodal)
        ),
        "clinical_unique_subject_count": int(
            clinical["subject_id"].nunique()
        ),
        "radiomics_unique_subject_count": int(
            radiomics["subject_id"].nunique()
        ),
        "multimodal_unique_subject_count": int(
            multimodal["subject_id"].nunique()
        ),
        "multimodal_subject_counts_by_split": {
            str(key): int(value)
            for key, value in split_counts.items()
        },
        "multimodal_record_counts_by_split": {
            str(key): int(value)
            for key, value in record_counts.items()
        },
        "clinical_feature_columns": (
            clinical_feature_columns
        ),
        "clinical_feature_count": len(
            clinical_feature_columns
        ),
        "radiomics_feature_count": len(
            radiomics_feature_columns
        ),
        "multimodal_feature_count": len(
            multimodal_feature_columns
        ),
        "problem_count": len(problems),
        "warning_count": len(warnings),
        "problems": problems,
        "warnings": warnings,
        "safe_for_next_stage": len(problems) == 0,
        "scaling_or_imputation_performed": False,
        "note": (
            "Scaling, imputation, and feature selection must be "
            "fitted using training data only in later stages."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate clinical, radiomics, and multimodal "
            "feature tables."
        )
    )

    parser.add_argument(
        "--clinical",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "clinical"
            / "clinical_features.csv"
        ),
    )

    parser.add_argument(
        "--radiomics",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "radiomics"
            / "radiomics_features.csv"
        ),
    )

    parser.add_argument(
        "--multimodal",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "multimodal_features.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "feature_table_validation.json"
        ),
    )

    args = parser.parse_args()

    summary = validate_feature_tables(
        clinical_path=args.clinical,
        radiomics_path=args.radiomics,
        multimodal_path=args.multimodal,
        output_path=args.output,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    if summary["problem_count"] > 0:
        raise SystemExit(
            "Feature-table validation failed. "
            "Review the reported problems."
        )


if __name__ == "__main__":
    main()