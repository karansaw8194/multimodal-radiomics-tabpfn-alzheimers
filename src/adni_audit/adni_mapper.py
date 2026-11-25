from __future__ import annotations

import csv
import json

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .audit import KNOWN_CLASSES, is_nifti, parse_xml

@dataclass
class ADNIRecord:
    nifti_path: str
    xml_path: Optional[str]
    mapping_status: str
    folder_class: Optional[str]
    subject_id: Optional[str]
    diagnosis_xml: Optional[str]
    diagnosis_folder: Optional[str]
    processing_version: Optional[str]
    acquisition_date: Optional[str]
    image_directory_id: Optional[str]
    image_id_xml: Optional[str]
    study_id: Optional[str]
    series_id: Optional[str]
    visit: Optional[str]
    age: Optional[str]
    sex: Optional[str]
    apoe_a1: Optional[str]
    apoe_a2: Optional[str]
    mmse: Optional[str]
    gds: Optional[str]
    cdr: Optional[str]
    weight: Optional[str]
    modality: Optional[str]
    date_acquired: Optional[str]
    warnings_json: str
    flattened_metadata_json: str

def identify_class_from_path(
    path: Path,
    root: Path,
) -> Optional[str]:
    """Find CN, MCI, or AD in a path."""
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return None

    for part in relative_parts:
        if part.upper() in KNOWN_CLASSES:
            return part.upper()

    return None

def identify_subject_id(
    nifti_path: Path,
    mri_root: Path,
) -> Optional[str]:
    """
    Supported structures:

    mri_root/
    └── CLASS/
        └── SUBJECT_ID/
            └── PROCESSING_VERSION/
                └── ACQUISITION_DATE/
                        └── IMAGE_DIRECTORY_ID/
                            └── MRI_FILE

    or the subject-first layout:

    mri_root/
    └── SUBJECT_ID/
        └── PROCESSING_VERSION/
            └── ACQUISITION_DATE/
                └── IMAGE_DIRECTORY_ID/
                    └── MRI_FILE
    """
    try:
        relative_parts = nifti_path.relative_to(mri_root).parts
    except ValueError:
        return None

    structure_offset = _mri_structure_offset(relative_parts)

    if len(relative_parts) >= structure_offset + 5:
        return relative_parts[structure_offset]

    return None

def identify_processing_version(
    nifti_path: Path,
    mri_root: Path,
) -> Optional[str]:
    """Return the processing-version directory name."""
    try:
        relative_parts = nifti_path.relative_to(mri_root).parts
    except ValueError:
        return None

    structure_offset = _mri_structure_offset(relative_parts)

    if len(relative_parts) >= structure_offset + 5:
        return relative_parts[structure_offset + 1]

    return None

def identify_acquisition_date(
    nifti_path: Path,
    mri_root: Path,
) -> Optional[str]:
    """Return the acquisition-date directory name."""
    try:
        relative_parts = nifti_path.relative_to(mri_root).parts
    except ValueError:
        return None

    structure_offset = _mri_structure_offset(relative_parts)

    if len(relative_parts) >= structure_offset + 5:
        return relative_parts[structure_offset + 2]

    return None

def identify_image_directory_id(
    nifti_path: Path,
    mri_root: Path,
) -> Optional[str]:
    """Return the image directory identifier, for example I48588."""
    try:
        relative_parts = nifti_path.relative_to(mri_root).parts
    except ValueError:
        return None

    structure_offset = _mri_structure_offset(relative_parts)

    if len(relative_parts) >= structure_offset + 5:
        return relative_parts[structure_offset + 3]

    return None


def _mri_structure_offset(relative_parts: tuple[str, ...]) -> int:
    """Return the leading class-directory offset for a legacy MRI path."""
    if relative_parts and relative_parts[0].upper() in KNOWN_CLASSES:
        return 1

    return 0

def find_class_metadata_directory(
    clinical_root: Path,
    class_name: str,
) -> Optional[Path]:
    """
    Find the class-specific metadata directory.

    This supports names such as:

    ad_metadata
    cn_metadata
    mci_metadata
    mci_metdata
    """
    if not clinical_root.exists():
        return None

    normalized_class = class_name.lower()
    candidates: list[Path] = []

    for directory in clinical_root.iterdir():
        if not directory.is_dir():
            continue

        normalized_name = directory.name.lower().replace("_", "")

        if normalized_class in normalized_name:
            candidates.append(directory)

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.name.lower())

    return candidates[0]

def is_complete_flat_xml(
    xml_path: Path,
    class_metadata_directory: Path,
) -> bool:
    """
    Identify flat ADNI XML files.

    Complete files are normally directly inside ad_metadata, cn_metadata,
    or the corresponding MCI metadata directory. Nested XML files are
    technical summary files and are not preferred.
    """
    return xml_path.parent == class_metadata_directory

def xml_contains_image_token(
    xml_path: Path,
    image_directory_id: str,
) -> bool:
    """
    Match I48588 in a filename without accidentally creating II48588.
    """
    image_token = image_directory_id.upper()

    return image_token in xml_path.name.upper()

def find_xml_for_nifti(
    nifti_path: Path,
    mri_root: Path,
    clinical_root: Path,
) -> tuple[Optional[Path], str, Optional[str]]:
    """
    Find the complete flat XML corresponding to one NIfTI file.

    Example NIfTI:

    raw/mri/AD/141_S_1152/
    MPR__GradWarp__B1_Correction__Mask/
    2006-12-26_15_23_11.0/
    I48588/
    image.nii

    Expected complete XML:

    raw/clinical/ad_metadata/
    ADNI_141_S_1152_MPR__GradWarp__B1_Correction__Mask_S24487_I48588.xml
    """
    class_name = identify_class_from_path(
        path=nifti_path,
        root=mri_root,
    )

    subject_id = identify_subject_id(
        nifti_path=nifti_path,
        mri_root=mri_root,
    )

    processing_version = identify_processing_version(
        nifti_path=nifti_path,
        mri_root=mri_root,
    )

    image_directory_id = identify_image_directory_id(
        nifti_path=nifti_path,
        mri_root=mri_root,
    )

    if subject_id is None:
        return None, "subject_id_not_found", class_name

    if image_directory_id is None:
        return None, "image_directory_id_not_found", class_name

    if class_name is not None:
        class_metadata_directory = find_class_metadata_directory(
            clinical_root=clinical_root,
            class_name=class_name,
        )

        if class_metadata_directory is None:
            return (
                None,
                "class_metadata_directory_not_found",
                class_name,
            )

        all_xml_files = sorted(
            path
            for path in class_metadata_directory.rglob("*.xml")
            if path.is_file()
        )
        complete_xml_files = [
            path
            for path in all_xml_files
            if is_complete_flat_xml(
                xml_path=path,
                class_metadata_directory=class_metadata_directory,
            )
        ]
    else:
        # Subject-first MRI exports store their complete IDAXS XML files
        # directly in clinical_root; nested XML files are technical summaries.
        complete_xml_files = sorted(
            path
            for path in clinical_root.glob("*.xml")
            if path.is_file()
        )

    if not complete_xml_files:
        return None, "no_complete_xml_files_found", class_name

    subject_candidates = [
        xml_path
        for xml_path in complete_xml_files
        if subject_id.upper() in xml_path.name.upper()
    ]

    image_candidates = [
        xml_path
        for xml_path in subject_candidates
        if xml_contains_image_token(
            xml_path=xml_path,
            image_directory_id=image_directory_id,
        )
    ]

    if not image_candidates:
        return (
            None,
            "complete_flat_xml_not_found",
            class_name,
        )

    if processing_version is not None:
        processing_candidates = [
            xml_path
            for xml_path in image_candidates
            if processing_version.upper() in xml_path.name.upper()
        ]

        if processing_candidates:
            image_candidates = processing_candidates

    image_candidates.sort(
        key=lambda path: (
            len(path.parts),
            str(path),
        )
    )

    if len(image_candidates) > 1:
        return (
            image_candidates[0],
            "multiple_complete_xml_candidates",
            class_name,
        )

    return (
        image_candidates[0],
        "complete_flat_xml_mapping",
        class_name,
    )

def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a list of dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

def audit_adni_layout(
    mri_root: Path,
    clinical_root: Path,
    output_dir: Path,
) -> dict:
    """
    Audit the separate ADNI MRI and clinical XML directory trees.

    Raw data is read only. Generated reports are written to output_dir.
    """
    mri_root = mri_root.expanduser().resolve()
    clinical_root = clinical_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not mri_root.exists():
        raise FileNotFoundError(
            f"MRI root does not exist: {mri_root}"
        )

    if not clinical_root.exists():
        raise FileNotFoundError(
            f"Clinical XML root does not exist: {clinical_root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nifti_files = sorted(
        path
        for path in mri_root.rglob("*")
        if path.is_file() and is_nifti(path)
    )

    records: list[ADNIRecord] = []

    for nifti_path in nifti_files:
        (
            xml_path,
            mapping_status,
            folder_class,
        ) = find_xml_for_nifti(
            nifti_path=nifti_path,
            mri_root=mri_root,
            clinical_root=clinical_root,
        )

        warnings: list[str] = []
        canonical: dict[str, str] = {}
        flattened: dict[str, list[str]] = {}

        if xml_path is None:
            warnings.append("Complete XML file was not found")
        else:
            (
                canonical,
                flattened,
                xml_error,
            ) = parse_xml(xml_path)

            if xml_error is not None:
                warnings.append(xml_error)

        folder_subject_id = identify_subject_id(
            nifti_path=nifti_path,
            mri_root=mri_root,
        )

        image_directory_id = identify_image_directory_id(
            nifti_path=nifti_path,
            mri_root=mri_root,
        )

        xml_subject_id = canonical.get("subject_id")
        subject_id = xml_subject_id or folder_subject_id

        if xml_subject_id is None:
            warnings.append(
                "Subject ID was obtained from the MRI directory structure"
            )
        elif folder_subject_id is not None:
            if xml_subject_id != folder_subject_id:
                warnings.append(
                    "XML subject ID and folder subject ID do not match"
                )

        diagnosis_xml = canonical.get("diagnosis")
        diagnosis_folder = folder_class

        if diagnosis_xml is None:
            warnings.append(
                "Diagnosis was not found or was not recognized in XML"
            )

        if diagnosis_xml and diagnosis_folder:
            if diagnosis_xml != diagnosis_folder:
                warnings.append(
                    "Folder class and XML diagnosis do not match"
                )

        if mapping_status != "complete_flat_xml_mapping":
            warnings.append(mapping_status)

        records.append(
            ADNIRecord(
                nifti_path=str(nifti_path),
                xml_path=(
                    str(xml_path)
                    if xml_path is not None
                    else None
                ),
                mapping_status=mapping_status,
                folder_class=folder_class,
                subject_id=subject_id,
                diagnosis_xml=diagnosis_xml,
                diagnosis_folder=diagnosis_folder,
                processing_version=identify_processing_version(
                    nifti_path=nifti_path,
                    mri_root=mri_root,
                ),
                acquisition_date=identify_acquisition_date(
                    nifti_path=nifti_path,
                    mri_root=mri_root,
                ),
                image_directory_id=image_directory_id,
                image_id_xml=canonical.get("image_id"),
                study_id=canonical.get("study_id"),
                series_id=canonical.get("series_id"),
                visit=canonical.get("visit"),
                age=canonical.get("age"),
                sex=canonical.get("sex"),
                apoe_a1=canonical.get("apoe_a1"),
                apoe_a2=canonical.get("apoe_a2"),
                mmse=canonical.get("mmse"),
                gds=canonical.get("gds"),
                cdr=canonical.get("cdr"),
                weight=canonical.get("weight"),
                modality=canonical.get("modality"),
                date_acquired=canonical.get("date_acquired"),
                warnings_json=json.dumps(
                    warnings,
                    sort_keys=True,
                ),
                flattened_metadata_json=json.dumps(
                    {
                        "canonical_fields": canonical,
                        "flattened_fields": flattened,
                    },
                    sort_keys=True,
                ),
            )
        )

    manifest_rows = [
        asdict(record)
        for record in records
    ]

    write_csv(
        output_dir / "adni_mri_xml_manifest.csv",
        manifest_rows,
    )

    warning_rows = []

    for record in records:
        warnings = json.loads(
            record.warnings_json
        )

        for warning in warnings:
            warning_rows.append(
                {
                    "subject_id": record.subject_id,
                    "nifti_path": record.nifti_path,
                    "xml_path": record.xml_path,
                    "warning": warning,
                }
            )

    write_csv(
        output_dir / "adni_mapping_warnings.csv",
        warning_rows,
    )

    subject_groups: dict[
        str,
        list[ADNIRecord],
    ] = defaultdict(list)

    for record in records:
        subject_key = (
            record.subject_id
            if record.subject_id is not None
            else "MISSING_SUBJECT_ID"
        )

        subject_groups[subject_key].append(record)

    subject_rows = []

    for subject_id, group in sorted(
        subject_groups.items()
    ):
        diagnoses = sorted(
            {
                record.diagnosis_xml
                for record in group
                if record.diagnosis_xml is not None
            }
        )

        folder_classes = sorted(
            {
                record.folder_class
                for record in group
                if record.folder_class is not None
            }
        )

        processing_versions = sorted(
            {
                record.processing_version
                for record in group
                if record.processing_version is not None
            }
        )

        acquisition_dates = sorted(
            {
                record.acquisition_date
                for record in group
                if record.acquisition_date is not None
            }
        )

        subject_warnings = []

        if len(diagnoses) > 1:
            subject_warnings.append(
                "Subject has multiple XML diagnosis values"
            )

        if len(folder_classes) > 1:
            subject_warnings.append(
                "Subject occurs under multiple folder classes"
            )

        subject_rows.append(
            {
                "subject_id": subject_id,
                "nifti_count": len(group),
                "acquisition_date_count": len(acquisition_dates),
                "xml_diagnoses": "|".join(diagnoses),
                "folder_classes": "|".join(folder_classes),
                "processing_versions": "|".join(
                    processing_versions
                ),
                "warnings": "|".join(subject_warnings),
            }
        )

    write_csv(
        output_dir / "adni_subject_summary.csv",
        subject_rows,
    )

    processing_groups: dict[
        tuple,
        list[ADNIRecord],
    ] = defaultdict(list)

    for record in records:
        group_key = (
            record.subject_id,
            record.acquisition_date,
            record.study_id,
            record.series_id,
        )

        processing_groups[group_key].append(record)

    processing_rows = []

    for group_key, group in sorted(
        processing_groups.items(),
        key=lambda item: str(item[0]),
    ):
        versions = sorted(
            {
                record.processing_version or "UNKNOWN"
                for record in group
            }
        )

        processing_rows.append(
            {
                "subject_id": group[0].subject_id,
                "acquisition_date": group[0].acquisition_date,
                "study_id": group[0].study_id,
                "series_id": group[0].series_id,
                "file_count": len(group),
                "processing_versions": "|".join(versions),
                "nifti_paths": "|".join(
                    record.nifti_path
                    for record in group
                ),
            }
        )

    write_csv(
        output_dir / "adni_processing_groups.csv",
        processing_rows,
    )

    summary = {
        "mri_root": str(mri_root),
        "clinical_root": str(clinical_root),
        "nifti_count": len(nifti_files),
        "manifest_record_count": len(records),
        "unique_subject_count": len(subject_groups),
        "mapping_status_counts": dict(
            Counter(
                record.mapping_status
                for record in records
            )
        ),
        "folder_class_counts": dict(
            Counter(
                record.folder_class
                for record in records
                if record.folder_class is not None
            )
        ),
        "xml_diagnosis_counts": dict(
            Counter(
                record.diagnosis_xml
                for record in records
                if record.diagnosis_xml is not None
            )
        ),
        "processing_version_counts": dict(
            Counter(
                record.processing_version or "UNKNOWN"
                for record in records
            )
        ),
        "warning_count": len(warning_rows),
    }

    with (
        output_dir / "adni_mapping_summary.json"
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
