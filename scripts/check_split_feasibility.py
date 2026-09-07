from __future__ import annotations

import argparse
import csv
import json

from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_CLASSES = {"CN", "MCI", "AD"}

def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("Manifest does not contain a header")

        required_columns = {"subject_id", "diagnosis_xml"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        return list(reader)

def check_split_feasibility(
    manifest_path: Path,
    output_path: Path,
) -> dict:
    rows = load_manifest(manifest_path)

    subjects_by_class: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        subject_id = row["subject_id"].strip()
        diagnosis = row["diagnosis_xml"].strip().upper()

        if not subject_id:
            continue

        if diagnosis not in VALID_CLASSES:
            continue

        subjects_by_class[diagnosis].add(subject_id)

    subject_counts = {
        class_name: len(subjects_by_class.get(class_name, set()))
        for class_name in ("AD", "CN", "MCI")
    }

    minimum_subject_count = min(subject_counts.values())

    errors: list[str] = []
    warnings: list[str] = []

    for class_name, count in subject_counts.items():
        if count < 3:
            errors.append(
                f"Class {class_name} has only {count} subject(s). "
                "At least 3 subjects are required for a basic "
                "train/validation/test feasibility check."
            )

    if len({row["subject_id"].strip() for row in rows}) == 0:
        errors.append("No valid subject IDs were found.")

    if errors:
        warnings.append(
            "Do not create train/validation/test splits yet."
        )
    else:
        warnings.append(
            "Basic subject-level three-way splitting is feasible."
        )
        warnings.append(
            "The final splitter must still verify class presence in "
            "every split."
        )

    result = {
        "manifest_path": str(manifest_path.resolve()),
        "record_count": len(rows),
        "unique_subject_count": len(
            {
                row["subject_id"].strip()
                for row in rows
                if row["subject_id"].strip()
            }
        ),
        "subject_counts_by_xml_diagnosis": subject_counts,
        "minimum_subject_count_in_any_class": minimum_subject_count,
        "three_way_subject_stratification_feasible": not errors,
        "errors": errors,
        "warnings": warnings,
        "recommended_action": (
            "Proceed to the subject-level split implementation."
            if not errors
            else (
                "Collect more subjects, especially from classes with "
                "fewer than three subjects."
            )
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether subject-level train/validation/test "
            "splitting is feasible."
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
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "split_feasibility.json"
        ),
    )

    args = parser.parse_args()

    result = check_split_feasibility(
        manifest_path=args.manifest,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
