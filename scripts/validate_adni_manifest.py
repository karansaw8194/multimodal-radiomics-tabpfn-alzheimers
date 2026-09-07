from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REQUIRED_COLUMNS = {
    "nifti_path",
    "xml_path",
    "mapping_status",
    "folder_class",
    "subject_id",
    "diagnosis_xml",
    "diagnosis_folder",
    "processing_version",
    "acquisition_date",
    "image_directory_id",
    "image_id_xml",
    "study_id",
    "series_id",
    "visit",
    "age",
    "sex",
    "apoe_a1",
    "apoe_a2",
    "mmse",
    "gds",
    "cdr",
    "weight",
    "modality",
    "date_acquired",
    "warnings_json",
    "flattened_metadata_json",
}

VALID_CLASSES = {"CN", "MCI", "AD"}

def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("Manifest does not contain a header")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        return list(reader)

def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

def validate_manifest(
    manifest_path: Path,
    output_dir: Path,
) -> dict:
    rows = load_manifest(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    subjects_by_diagnosis: dict[str, set[str]] = defaultdict(set)
    diagnoses_by_subject: dict[str, set[str]] = defaultdict(set)
    records_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, row in enumerate(rows, start=2):
        nifti_path = Path(row["nifti_path"])
        xml_path = Path(row["xml_path"]) if row["xml_path"] else None

        subject_id = row["subject_id"].strip()
        diagnosis_xml = row["diagnosis_xml"].strip().upper()
        diagnosis_folder = row["diagnosis_folder"].strip().upper()
        mapping_status = row["mapping_status"].strip()

        if not nifti_path.exists():
            errors.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "NIfTI file does not exist",
                }
            )

        if xml_path is None or not xml_path.exists():
            errors.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "XML file does not exist",
                }
            )

        if not subject_id:
            errors.append(
                {
                    "row": str(index),
                    "subject_id": "",
                    "nifti_path": str(nifti_path),
                    "problem": "Missing subject ID",
                }
            )

        if diagnosis_xml not in VALID_CLASSES:
            errors.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": (
                        "XML diagnosis is missing or outside CN/MCI/AD"
                    ),
                }
            )

        if diagnosis_folder not in VALID_CLASSES:
            warnings.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "Folder diagnosis is missing or invalid",
                }
            )

        if diagnosis_xml and diagnosis_folder:
            if diagnosis_xml != diagnosis_folder:
                warnings.append(
                    {
                        "row": str(index),
                        "subject_id": subject_id,
                        "nifti_path": str(nifti_path),
                        "problem": (
                            "Folder diagnosis differs from XML diagnosis: "
                            f"folder={diagnosis_folder}, "
                            f"xml={diagnosis_xml}"
                        ),
                    }
                )

        if mapping_status != "complete_flat_xml_mapping":
            warnings.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": (
                        "Mapping status is not complete_flat_xml_mapping: "
                        f"{mapping_status}"
                    ),
                }
            )

        if not row["processing_version"].strip():
            warnings.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "Missing processing version",
                }
            )

        if not row["acquisition_date"].strip():
            warnings.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "Missing acquisition date",
                }
            )

        if not row["image_id_xml"].strip():
            warnings.append(
                {
                    "row": str(index),
                    "subject_id": subject_id,
                    "nifti_path": str(nifti_path),
                    "problem": "Missing XML image ID",
                }
            )

        if subject_id:
            diagnoses_by_subject[subject_id].add(diagnosis_xml)
            records_by_subject[subject_id].append(row)

        if subject_id and diagnosis_xml in VALID_CLASSES:
            subjects_by_diagnosis[diagnosis_xml].add(subject_id)

    for subject_id, diagnoses in diagnoses_by_subject.items():
        valid_diagnoses = {
            diagnosis
            for diagnosis in diagnoses
            if diagnosis in VALID_CLASSES
        }

        if len(valid_diagnoses) > 1:
            warnings.append(
                {
                    "row": "",
                    "subject_id": subject_id,
                    "nifti_path": "",
                    "problem": (
                        "Subject has multiple XML diagnoses: "
                        + "|".join(sorted(valid_diagnoses))
                    ),
                }
            )

    duplicate_subject_rows = []

    for diagnosis, subjects in subjects_by_diagnosis.items():
        for subject_id in sorted(subjects):
            duplicate_subject_rows.append(
                {
                    "subject_id": subject_id,
                    "diagnosis_xml": diagnosis,
                    "record_count": str(
                        len(records_by_subject[subject_id])
                    ),
                }
            )

    write_csv(
        output_dir / "manifest_validation_errors.csv",
        errors,
    )

    write_csv(
        output_dir / "manifest_validation_warnings.csv",
        warnings,
    )

    write_csv(
        output_dir / "subject_record_counts.csv",
        duplicate_subject_rows,
    )

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "record_count": len(rows),
        "unique_subject_count": len(records_by_subject),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "xml_diagnosis_record_counts": {
            diagnosis: sum(
                1
                for row in rows
                if row["diagnosis_xml"].strip().upper() == diagnosis
            )
            for diagnosis in sorted(VALID_CLASSES)
        },
        "xml_diagnosis_subject_counts": {
            diagnosis: len(subjects)
            for diagnosis, subjects in sorted(
                subjects_by_diagnosis.items()
            )
        },
        "subjects_with_multiple_xml_diagnoses": sorted(
            subject_id
            for subject_id, diagnoses in diagnoses_by_subject.items()
            if len(
                {
                    diagnosis
                    for diagnosis in diagnoses
                    if diagnosis in VALID_CLASSES
                }
            )
            > 1
        ),
        "raw_data_validation_passed": len(errors) == 0,
        "safe_for_next_stage": len(errors) == 0,
    }

    with (
        output_dir / "manifest_validation_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ADNI MRI/XML manifest."
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
            / "manifest_validation"
        ),
    )

    args = parser.parse_args()

    summary = validate_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["error_count"] > 0:
        raise SystemExit(
            "Manifest validation failed. Review manifest_validation_errors.csv."
        )

if __name__ == "__main__":
    main()
