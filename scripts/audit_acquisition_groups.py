from __future__ import annotations

import argparse
import csv
import json

from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUP_COLUMNS = (
    "subject_id",
    "study_id",
    "series_id",
    "date_acquired",
    "acquisition_date",
)

def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        return list(reader)

def choose_group_key(row: dict[str, str]) -> tuple[str, ...]:
    """
    Build a stable acquisition key.

    Preferred identifiers are subject, study, and series. When an identifier
    is unavailable, the acquisition date is retained. This function never
    merges records from different subjects.
    """
    subject_id = row["subject_id"].strip()
    study_id = row.get("study_id", "").strip()
    series_id = row.get("series_id", "").strip()
    date_acquired = row.get("date_acquired", "").strip()
    acquisition_date = row.get("acquisition_date", "").strip()

    if not subject_id:
        raise ValueError("A manifest record has no subject_id")

    if study_id and series_id:
        return (
            "subject_study_series",
            subject_id,
            study_id,
            series_id,
        )

    if series_id:
        return (
            "subject_series",
            subject_id,
            series_id,
        )

    if study_id and date_acquired:
        return (
            "subject_study_date",
            subject_id,
            study_id,
            date_acquired,
        )

    if acquisition_date:
        return (
            "subject_acquisition_date",
            subject_id,
            acquisition_date,
        )

    image_id = (
        row.get("image_directory_id", "").strip()
        or row.get("image_id_xml", "").strip()
    )

    if image_id:
        return (
            "subject_image",
            subject_id,
            image_id,
        )

    raise ValueError(
        "Could not construct a safe acquisition key for record: "
        + row.get("nifti_path", "")
    )

def group_records(
    rows: list[dict[str, str]],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[
        tuple[str, ...],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        groups[choose_group_key(row)].append(row)

    return dict(groups)

def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
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

def audit_groups(
    manifest_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
) -> dict:
    manifest_rows = load_csv(manifest_path)
    split_rows = load_csv(split_manifest_path)

    manifest_by_nifti = {
        row["nifti_path"]: row
        for row in manifest_rows
    }

    split_by_nifti = {
        row["nifti_path"]: row.get("split", "").strip()
        for row in split_rows
    }

    missing_from_split_manifest = sorted(
        set(manifest_by_nifti) - set(split_by_nifti)
    )

    if missing_from_split_manifest:
        raise ValueError(
            "Some manifest records are missing from the split manifest: "
            + "; ".join(missing_from_split_manifest[:5])
        )

    groups = group_records(manifest_rows)

    group_rows: list[dict[str, str]] = []
    problems: list[dict[str, str]] = []

    processing_version_counts = Counter()
    group_size_counts = Counter()

    for group_number, (group_key, records) in enumerate(
        sorted(groups.items(), key=lambda item: str(item[0])),
        start=1,
    ):
        group_id = f"acquisition_{group_number:05d}"

        subject_ids = sorted(
            {
                row["subject_id"].strip()
                for row in records
            }
        )

        diagnoses = sorted(
            {
                row["diagnosis_xml"].strip().upper()
                for row in records
                if row["diagnosis_xml"].strip()
            }
        )

        splits = sorted(
            {
                split_by_nifti[row["nifti_path"]]
                for row in records
                if split_by_nifti[row["nifti_path"]]
            }
        )

        processing_versions = sorted(
            {
                row["processing_version"].strip()
                for row in records
                if row["processing_version"].strip()
            }
        )

        nifti_paths = sorted(
            row["nifti_path"]
            for row in records
        )

        group_size_counts[len(records)] += 1

        for version in processing_versions:
            processing_version_counts[version] += 1

        if len(subject_ids) != 1:
            problems.append(
                {
                    "group_id": group_id,
                    "problem": (
                        "Acquisition group contains multiple subjects: "
                        + "|".join(subject_ids)
                    ),
                    "nifti_paths": "|".join(nifti_paths),
                }
            )

        if len(diagnoses) > 1:
            problems.append(
                {
                    "group_id": group_id,
                    "problem": (
                        "Acquisition group contains multiple XML "
                        "diagnoses: "
                        + "|".join(diagnoses)
                    ),
                    "nifti_paths": "|".join(nifti_paths),
                }
            )

        if len(splits) != 1:
            problems.append(
                {
                    "group_id": group_id,
                    "problem": (
                        "Acquisition group occurs in multiple or missing "
                        "splits: "
                        + "|".join(splits)
                    ),
                    "nifti_paths": "|".join(nifti_paths),
                }
            )

        if not processing_versions:
            problems.append(
                {
                    "group_id": group_id,
                    "problem": "Acquisition group has no processing version",
                    "nifti_paths": "|".join(nifti_paths),
                }
            )

        group_rows.append(
            {
                "group_id": group_id,
                "group_key": "|".join(group_key),
                "subject_id": subject_ids[0] if subject_ids else "",
                "diagnosis_xml": diagnoses[0] if len(diagnoses) == 1 else "|".join(diagnoses),
                "split": splits[0] if len(splits) == 1 else "|".join(splits),
                "record_count": str(len(records)),
                "processing_version_count": str(len(processing_versions)),
                "processing_versions": "|".join(processing_versions),
                "nifti_paths": "|".join(nifti_paths),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "acquisition_groups.csv",
        group_rows,
    )

    write_csv(
        output_dir / "acquisition_group_problems.csv",
        problems,
    )

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "split_manifest_path": str(split_manifest_path.resolve()),
        "record_count": len(manifest_rows),
        "acquisition_group_count": len(groups),
        "group_size_counts": {
            str(size): count
            for size, count in sorted(group_size_counts.items())
        },
        "processing_version_group_counts": dict(
            sorted(processing_version_counts.items())
        ),
        "problem_count": len(problems),
        "safe_for_processing_version_selection": len(problems) == 0,
        "processing_version_selection_performed": False,
        "raw_data_modified": False,
        "important_note": (
            "Multiple processing versions in one acquisition group "
            "must not be treated as independent samples. A processing "
            "version must be selected explicitly before preprocessing."
        ),
    }

    with (
        output_dir / "acquisition_group_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit acquisition groups and processing versions without "
            "selecting or modifying MRI files."
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
        "--split-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "subject_splits"
            / "manifest_with_splits.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "acquisition_groups"
        ),
    )

    args = parser.parse_args()

    summary = audit_groups(
        manifest_path=args.manifest,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
