from pathlib import Path

import pytest

from scripts.create_subject_splits import (
    allocate_class_counts,
    create_splits,
)

def write_test_manifest(
    path: Path,
    records: list[tuple[str, str, str]],
) -> None:
    path.write_text(
        "nifti_path,xml_path,subject_id,diagnosis_xml,"
        "acquisition_date,processing_version\n"
        + "".join(
            (
                f"/tmp/{subject_id}_{visit}.nii,"
                f"/tmp/{subject_id}_{visit}.xml,"
                f"{subject_id},{diagnosis},"
                f"{visit},MPR__GradWarp\n"
            )
            for subject_id, diagnosis, visit in records
        ),
        encoding="utf-8",
    )

def test_allocate_class_counts() -> None:
    counts = allocate_class_counts(
        subject_count=13,
        proportions={
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
    )

    assert sum(counts.values()) == 13
    assert all(value >= 1 for value in counts.values())

def test_create_splits_keeps_subject_records_together(
    tmp_path: Path,
) -> None:
    records = []

    for class_name, prefix, count in (
        ("AD", "AD", 3),
        ("CN", "CN", 3),
        ("MCI", "MCI", 3),
    ):
        for index in range(count):
            subject_id = f"{prefix}_{index:03d}"

            records.append(
                (subject_id, class_name, "baseline")
            )
            records.append(
                (subject_id, class_name, "month_06")
            )

    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "splits"

    write_test_manifest(manifest_path, records)

    result = create_splits(
        manifest_path=manifest_path,
        output_dir=output_dir,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    assert result["record_count"] == 18
    assert result["unique_subject_count"] == 9
    assert result["subject_leakage_check_passed"] is True

    subject_split_path = output_dir / "subject_splits.csv"
    manifest_split_path = output_dir / "manifest_with_splits.csv"

    assert subject_split_path.exists()
    assert manifest_split_path.exists()

    import csv

    with subject_split_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        subject_rows = list(csv.DictReader(handle))

    with manifest_split_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))

    split_by_subject = {
        row["subject_id"]: row["split"]
        for row in subject_rows
    }

    for row in manifest_rows:
        assert row["split"] == split_by_subject[row["subject_id"]]

def test_split_rejects_conflicting_subject_labels(
    tmp_path: Path,
) -> None:
    records = [
        ("AD_001", "AD", "baseline"),
        ("AD_001", "CN", "month_06"),
        ("AD_002", "AD", "baseline"),
        ("AD_003", "AD", "baseline"),
        ("CN_001", "CN", "baseline"),
        ("CN_002", "CN", "baseline"),
        ("CN_003", "CN", "baseline"),
        ("MCI_001", "MCI", "baseline"),
        ("MCI_002", "MCI", "baseline"),
        ("MCI_003", "MCI", "baseline"),
    ]

    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "splits"

    write_test_manifest(manifest_path, records)

    with pytest.raises(ValueError, match="multiple XML diagnoses"):
        create_splits(
            manifest_path=manifest_path,
            output_dir=output_dir,
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            random_seed=42,
        )
