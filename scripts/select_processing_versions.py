from __future__ import annotations

import argparse
import csv
import json

from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

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

def choose_record(
    records: list[dict[str, str]],
    preferred_versions: list[str],
    allow_singleton: bool,
) -> tuple[dict[str, str] | None, str]:
    """
    Select exactly one record from an acquisition group.

    Preferred versions are evaluated in the supplied order. A singleton
    acquisition may only be selected when allow_singleton is explicitly true.
    """
    if not records:
        return None, "empty_group"

    versions = {
        row["processing_version"].strip()
        for row in records
    }

    for preferred_version in preferred_versions:
        matching = [
            row
            for row in records
            if row["processing_version"].strip() == preferred_version
        ]

        if len(matching) == 1:
            return matching[0], "preferred_version"

        if len(matching) > 1:
            return None, (
                "multiple_records_for_same_processing_version"
            )

    if len(records) == 1 and allow_singleton:
        return records[0], "explicit_singleton"

    return None, (
        "no_preferred_version_available:"
        + "|".join(sorted(versions))
    )

def select_processing_versions(
    acquisition_groups_path: Path,
    manifest_path: Path,
    output_dir: Path,
    preferred_versions: list[str],
    allow_singleton: bool,
) -> dict:
    group_rows = load_csv(acquisition_groups_path)
    manifest_rows = load_csv(manifest_path)

    manifest_by_path = {
        row["nifti_path"]: row
        for row in manifest_rows
    }

    records_by_group: dict[str, list[dict[str, str]]] = {}

    for group_row in group_rows:
        group_id = group_row["group_id"]
        nifti_paths = [
            path
            for path in group_row["nifti_paths"].split("|")
            if path
        ]

        records = []

        for nifti_path in nifti_paths:
            if nifti_path not in manifest_by_path:
                raise ValueError(
                    f"NIfTI path from acquisition group is missing from "
                    f"the manifest: {nifti_path}"
                )

            records.append(manifest_by_path[nifti_path])

        records_by_group[group_id] = records

    selected_rows: list[dict[str, str]] = []
    unresolved_rows: list[dict[str, str]] = []
    selection_rows: list[dict[str, str]] = []

    for group_row in group_rows:
        group_id = group_row["group_id"]
        records = records_by_group[group_id]

        selected, selection_reason = choose_record(
            records=records,
            preferred_versions=preferred_versions,
            allow_singleton=allow_singleton,
        )

        if selected is None:
            unresolved_rows.append(
                {
                    "group_id": group_id,
                    "subject_id": group_row["subject_id"],
                    "split": group_row["split"],
                    "available_versions": group_row[
                        "processing_versions"
                    ],
                    "record_count": group_row["record_count"],
                    "reason": selection_reason,
                }
            )
            continue

        selected_copy = dict(selected)
        selected_copy["acquisition_group_id"] = group_id
        selected_copy["selection_reason"] = selection_reason
        selected_rows.append(selected_copy)

        selection_rows.append(
            {
                "group_id": group_id,
                "subject_id": selected["subject_id"],
                "split": selected.get("split", group_row["split"]),
                "selected_processing_version": selected[
                    "processing_version"
                ],
                "selection_reason": selection_reason,
                "selected_nifti_path": selected["nifti_path"],
                "available_versions": group_row[
                    "processing_versions"
                ],
                "available_record_count": group_row["record_count"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "selected_manifest.csv",
        selected_rows,
    )

    write_csv(
        output_dir / "processing_version_selection.csv",
        selection_rows,
    )

    write_csv(
        output_dir / "unresolved_acquisition_groups.csv",
        unresolved_rows,
    )

    split_counts = Counter(
        row.get("split", "")
        for row in selected_rows
    )

    diagnosis_counts = Counter(
        row.get("diagnosis_xml", "")
        for row in selected_rows
    )

    selected_version_counts = Counter(
        row.get("processing_version", "")
        for row in selected_rows
    )

    summary = {
        "acquisition_group_count": len(group_rows),
        "selected_record_count": len(selected_rows),
        "unresolved_group_count": len(unresolved_rows),
        "preferred_versions": preferred_versions,
        "allow_singleton": allow_singleton,
        "selection_completed": len(unresolved_rows) == 0,
        "raw_data_modified": False,
        "selected_split_counts": dict(sorted(split_counts.items())),
        "selected_diagnosis_counts": dict(
            sorted(diagnosis_counts.items())
        ),
        "selected_processing_version_counts": dict(
            sorted(selected_version_counts.items())
        ),
        "selection_reasons": dict(
            Counter(
                row["selection_reason"]
                for row in selection_rows
            )
        ),
        "important_note": (
            "The selected manifest contains at most one MRI record per "
            "acquisition group. Subject ID and diagnosis remain metadata "
            "columns and must not be used as model features."
        ),
    }

    with (
        output_dir / "processing_version_selection_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select one explicit processing version per acquisition group."
        )
    )

    parser.add_argument(
        "--acquisition-groups",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "acquisition_groups"
            / "acquisition_groups.csv"
        ),
    )

    parser.add_argument(
        "--manifest",
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
            / "selected_processing"
        ),
    )

    parser.add_argument(
        "--preferred-version",
        action="append",
        dest="preferred_versions",
        default=[],
        help=(
            "Processing version to prefer. Repeat this argument to define "
            "priority order."
        ),
    )

    parser.add_argument(
        "--allow-singleton",
        action="store_true",
        help=(
            "Allow groups containing exactly one available MRI record to "
            "be selected even when their version is not in the preference "
            "list."
        ),
    )

    args = parser.parse_args()

    if not args.preferred_versions and not args.allow_singleton:
        parser.error(
            "Provide at least one --preferred-version or explicitly use "
            "--allow-singleton."
        )

    summary = select_processing_versions(
        acquisition_groups_path=args.acquisition_groups,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        preferred_versions=args.preferred_versions,
        allow_singleton=args.allow_singleton,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["unresolved_group_count"] > 0:
        raise SystemExit(
            "Some acquisition groups are unresolved. Review "
            "unresolved_acquisition_groups.csv."
        )

if __name__ == "__main__":
    main()
