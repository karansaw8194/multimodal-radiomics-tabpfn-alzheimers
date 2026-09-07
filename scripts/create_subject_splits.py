from __future__ import annotations

import argparse
import csv
import json
import random

from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_ORDER = ("AD", "CN", "MCI")
SPLIT_ORDER = ("train", "validation", "test")
VALID_CLASSES = set(CLASS_ORDER)

def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("Manifest does not contain a header")

        required_columns = {
            "subject_id",
            "diagnosis_xml",
            "acquisition_date",
            "processing_version",
            "nifti_path",
        }

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Manifest contains no records")

    return rows

def validate_subject_labels(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    """
    Verify that each subject has exactly one XML diagnosis.

    The XML diagnosis is authoritative. Folder diagnosis is not used.
    """
    diagnoses_by_subject: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        subject_id = row["subject_id"].strip()
        diagnosis = row["diagnosis_xml"].strip().upper()

        if not subject_id:
            raise ValueError("A manifest record has an empty subject_id")

        if diagnosis not in VALID_CLASSES:
            raise ValueError(
                f"Invalid XML diagnosis for subject {subject_id}: "
                f"{diagnosis!r}"
            )

        diagnoses_by_subject[subject_id].add(diagnosis)

    conflicts = {
        subject_id: sorted(diagnoses)
        for subject_id, diagnoses in diagnoses_by_subject.items()
        if len(diagnoses) != 1
    }

    if conflicts:
        raise ValueError(
            "Subjects have multiple XML diagnoses: "
            + json.dumps(conflicts, sort_keys=True)
        )

    return {
        subject_id: next(iter(diagnoses))
        for subject_id, diagnoses in diagnoses_by_subject.items()
    }

def allocate_class_counts(
    subject_count: int,
    proportions: dict[str, float],
) -> dict[str, int]:
    """
    Allocate subjects to train, validation, and test.

    Each class receives at least one subject in each split when the class
    contains at least three subjects. Remaining subjects are allocated using
    the configured proportions and largest-remainder rounding.
    """
    if subject_count < 3:
        raise ValueError(
            "Each class must contain at least three subjects for this "
            "three-way split."
        )

    counts = {split_name: 1 for split_name in SPLIT_ORDER}
    remaining = subject_count - len(SPLIT_ORDER)

    ideal_remaining = {
        split_name: remaining * proportions[split_name]
        for split_name in SPLIT_ORDER
    }

    floor_remaining = {
        split_name: int(ideal_remaining[split_name] // 1)
        for split_name in SPLIT_ORDER
    }

    for split_name in SPLIT_ORDER:
        counts[split_name] += floor_remaining[split_name]

    assigned = sum(counts.values())
    leftover = subject_count - assigned

    remainders = sorted(
        SPLIT_ORDER,
        key=lambda split_name: (
            -(
                ideal_remaining[split_name]
                - floor_remaining[split_name]
            ),
            SPLIT_ORDER.index(split_name),
        ),
    )

    for index in range(leftover):
        counts[remainders[index % len(remainders)]] += 1

    if sum(counts.values()) != subject_count:
        raise RuntimeError(
            "Internal split allocation error: counts do not sum to the "
            "class subject count."
        )

    return counts

def split_subjects_for_class(
    subjects: list[str],
    class_name: str,
    proportions: dict[str, float],
    random_seed: int,
) -> dict[str, list[str]]:
    """Shuffle and split subjects from one diagnosis class."""
    shuffled = sorted(subjects)

    class_seed = random_seed + sum(
        ord(character) for character in class_name
    )

    generator = random.Random(class_seed)
    generator.shuffle(shuffled)

    counts = allocate_class_counts(
        subject_count=len(shuffled),
        proportions=proportions,
    )

    result: dict[str, list[str]] = {}
    start = 0

    for split_name in SPLIT_ORDER:
        stop = start + counts[split_name]
        result[split_name] = shuffled[start:stop]
        start = stop

    return result

def create_subject_split_mapping(
    subject_to_class: dict[str, str],
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> dict[str, str]:
    """Create a subject-to-split mapping with class stratification."""
    proportions = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }

    total = sum(proportions.values())

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test proportions must sum to 1.0"
        )

    subjects_by_class: dict[str, list[str]] = defaultdict(list)

    for subject_id, class_name in subject_to_class.items():
        subjects_by_class[class_name].append(subject_id)

    subject_to_split: dict[str, str] = {}

    for class_name in CLASS_ORDER:
        class_subjects = subjects_by_class.get(class_name, [])

        if len(class_subjects) < 3:
            raise ValueError(
                f"Class {class_name} has {len(class_subjects)} subjects. "
                "At least three subjects are required."
            )

        class_split_subjects = split_subjects_for_class(
            subjects=class_subjects,
            class_name=class_name,
            proportions=proportions,
            random_seed=random_seed,
        )

        for split_name, split_subjects in class_split_subjects.items():
            for subject_id in split_subjects:
                if subject_id in subject_to_split:
                    raise RuntimeError(
                        f"Subject assigned more than once: {subject_id}"
                    )

                subject_to_split[subject_id] = split_name

    if set(subject_to_split) != set(subject_to_class):
        raise RuntimeError(
            "Some subjects were not assigned to a split."
        )

    return subject_to_split

def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

def validate_no_subject_leakage(
    subject_rows: list[dict[str, str]],
) -> list[str]:
    """Check that each subject occurs in exactly one split."""
    splits_by_subject: dict[str, set[str]] = defaultdict(set)

    for row in subject_rows:
        splits_by_subject[row["subject_id"]].add(row["split"])

    problems = []

    for subject_id, splits in sorted(splits_by_subject.items()):
        if len(splits) != 1:
            problems.append(
                f"Subject {subject_id} occurs in splits: "
                + "|".join(sorted(splits))
            )

    return problems

def create_splits(
    manifest_path: Path,
    output_dir: Path,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> dict:
    rows = load_manifest(manifest_path)
    subject_to_class = validate_subject_labels(rows)

    subject_to_split = create_subject_split_mapping(
        subject_to_class=subject_to_class,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_seed=random_seed,
    )

    subject_rows = []

    for subject_id in sorted(subject_to_split):
        subject_rows.append(
            {
                "subject_id": subject_id,
                "diagnosis_xml": subject_to_class[subject_id],
                "split": subject_to_split[subject_id],
            }
        )

    leakage_problems = validate_no_subject_leakage(subject_rows)

    if leakage_problems:
        raise RuntimeError(
            "Subject leakage detected:\n"
            + "\n".join(leakage_problems)
        )

    manifest_split_rows = []

    for row in rows:
        subject_id = row["subject_id"].strip()
        split_name = subject_to_split[subject_id]

        output_row = dict(row)
        output_row["split"] = split_name

        manifest_split_rows.append(output_row)

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "subject_splits.csv",
        subject_rows,
    )

    write_csv(
        output_dir / "manifest_with_splits.csv",
        manifest_split_rows,
    )

    split_subject_counts = {
        split_name: Counter(
            row["diagnosis_xml"]
            for row in subject_rows
            if row["split"] == split_name
        )
        for split_name in SPLIT_ORDER
    }

    split_record_counts = {
        split_name: Counter(
            row["diagnosis_xml"]
            for row in manifest_split_rows
            if row["split"] == split_name
        )
        for split_name in SPLIT_ORDER
    }

    subject_count_by_split = Counter(
        row["split"]
        for row in subject_rows
    )

    record_count_by_split = Counter(
        row["split"]
        for row in manifest_split_rows
    )

    result = {
        "manifest_path": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "random_seed": random_seed,
        "proportions": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "record_count": len(rows),
        "unique_subject_count": len(subject_to_class),
        "subject_count_by_split": dict(subject_count_by_split),
        "record_count_by_split": dict(record_count_by_split),
        "subject_class_counts_by_split": {
            split_name: dict(
                sorted(split_subject_counts[split_name].items())
            )
            for split_name in SPLIT_ORDER
        },
        "record_class_counts_by_split": {
            split_name: dict(
                sorted(split_record_counts[split_name].items())
            )
            for split_name in SPLIT_ORDER
        },
        "subject_leakage_problems": leakage_problems,
        "subject_leakage_check_passed": not leakage_problems,
        "target_column": "diagnosis_xml",
        "group_column": "subject_id",
        "folder_diagnosis_used": False,
        "subject_id_used_as_feature": False,
    }

    with (
        output_dir / "split_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic subject-level train/validation/test "
            "splits from an ADNI manifest."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "adni_mri_xml_manifest.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "subject_splits"
        ),
    )

    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    result = create_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
