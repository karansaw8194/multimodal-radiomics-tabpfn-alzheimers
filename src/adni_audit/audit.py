from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

NIFTI_SUFFIXES = (".nii", ".nii.gz")
KNOWN_CLASSES = {"CN", "MCI", "AD"}

KNOWN_PROCESSING_VERSIONS = (
    "MPR__GradWarp__B1_Correction__N3__Scaled",
    "MPR__GradWarp__B1_Correction__Mask",
    "MPR__GradWarp__B1_Correction__N3",
    "MPR__GradWarp__B1_Correction",
    "MPR__GradWarp",
    "MPR__GradWarp__N3__Scaled_2",
    "MPR__GradWarp__N3__Scaled",
    "MPR__GradWarp__N3",
    "MPR__GradWarp__Mask",
    "MT1__GradWarp__N3m",
    "MT1__N3m",
)

@dataclass
class MRIRecord:
    nifti_path: str
    xml_path: Optional[str]
    xml_mapping_status: str
    subject_id: Optional[str]
    diagnosis_xml: Optional[str]
    diagnosis_folder: Optional[str]
    visit: Optional[str]
    study_id: Optional[str]
    series_id: Optional[str]
    image_id: Optional[str]
    processing_version: Optional[str]
    acquisition_key: str
    nifti_size_bytes: int
    metadata_json: str
    warnings_json: str

def local_name(tag: str) -> str:
    """Remove an XML namespace from a tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]

    return tag

def normalize_key(value: str) -> str:
    """Normalize XML names for reliable comparison."""
    value = local_name(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)

def clean_value(value: Optional[str]) -> Optional[str]:
    """Strip whitespace and return None for empty values."""
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None

def normalize_diagnosis(value: Optional[str]) -> Optional[str]:
    """
    Normalize the ADNI researchGroup value.

    Only explicit diagnosis labels are accepted. Diagnosis is not inferred
    from MMSE, GDS, CDR, or any other clinical score.
    """
    if value is None:
        return None

    normalized = value.strip().upper()

    aliases = {
        "CN": "CN",
        "NORMAL": "CN",
        "CONTROL": "CN",
        "COGNITIVELY NORMAL": "CN",
        "MCI": "MCI",
        "MILD COGNITIVE IMPAIRMENT": "MCI",
        "AD": "AD",
        "ALZHEIMERS": "AD",
        "ALZHEIMER'S": "AD",
        "ALZHEIMER DISEASE": "AD",
        "ALZHEIMER'S DISEASE": "AD",
    }

    return aliases.get(normalized)

def is_nifti(path: Path) -> bool:
    """Return True for .nii and .nii.gz files."""
    return path.name.lower().endswith(NIFTI_SUFFIXES)

def _add_flattened(
    flattened: dict[str, list[str]],
    key: str,
    value: Optional[str],
) -> None:
    """Add a non-empty XML value to the flattened metadata dictionary."""
    value = clean_value(value)

    if value is None:
        return

    normalized_key = normalize_key(key)

    if normalized_key:
        flattened[normalized_key].append(value)

def parse_xml(
    xml_path: Path,
) -> tuple[dict[str, str], dict[str, list[str]], Optional[str]]:
    """
    Parse ADNI IDAXS XML metadata.

    Supported fields include:

    - subjectIdentifier
    - researchGroup
    - subjectSex
    - subjectInfo item="APOE A1"
    - subjectInfo item="APOE A2"
    - visitIdentifier
    - assessment name="MMSE"
    - assessmentScore attribute="MMSCORE"
    - assessmentScore attribute="GDTOTAL"
    - assessmentScore attribute="CDGLOBAL"
    - studyIdentifier
    - subjectAge
    - weightKg
    - seriesIdentifier
    - modality
    - dateAcquired
    - derivedProduct imageUID
    """
    flattened: dict[str, list[str]] = defaultdict(list)
    canonical: dict[str, str] = {}

    try:
        root = ET.parse(xml_path).getroot()

        for element in root.iter():
            tag = local_name(element.tag)
            tag_key = normalize_key(tag)
            text = clean_value(element.text)

            if text is not None:
                _add_flattened(flattened, tag_key, text)

            for attribute_name, attribute_value in element.attrib.items():
                attribute_key = normalize_key(attribute_name)
                attribute_value = clean_value(attribute_value)

                if attribute_value is None:
                    continue

                _add_flattened(
                    flattened,
                    f"{tag_key}_{attribute_key}",
                    attribute_value,
                )

                _add_flattened(
                    flattened,
                    attribute_key,
                    attribute_value,
                )

            if tag_key == "subjectinfo":
                item_value = None

                for attribute_name, attribute_value in element.attrib.items():
                    if normalize_key(attribute_name) == "item":
                        item_value = clean_value(attribute_value)
                        break

                if item_value is not None and text is not None:
                    item_key = normalize_key(item_value)

                    _add_flattened(
                        flattened,
                        f"subjectinfo_{item_key}",
                        text,
                    )

            if tag_key == "assessmentscore":
                score_value = text

                for attribute_name, attribute_value in element.attrib.items():
                    if normalize_key(attribute_name) == "attribute":
                        attribute_key = normalize_key(attribute_value)

                        if score_value is not None:
                            _add_flattened(
                                flattened,
                                f"assessmentscore_{attribute_key}",
                                score_value,
                            )

                            _add_flattened(
                                flattened,
                                attribute_key,
                                score_value,
                            )

        def first_flattened(*keys: str) -> Optional[str]:
            for key in keys:
                normalized_key = normalize_key(key)
                values = flattened.get(normalized_key)

                if values:
                    return values[0]

            return None

        def assign(
            field_name: str,
            *candidate_keys: str,
        ) -> None:
            value = first_flattened(*candidate_keys)

            if value is not None:
                canonical[field_name] = value

        assign(
            "subject_id",
            "subjectIdentifier",
            "subjectId",
            "subject_id",
        )

        assign(
            "diagnosis",
            "researchGroup",
            "diagnosis",
            "diagnosisLabel",
        )

        assign(
            "sex",
            "subjectSex",
            "sex",
            "gender",
        )

        assign(
            "age",
            "subjectAge",
            "age",
            "ageAtVisit",
        )

        assign(
            "weight",
            "weightKg",
            "weight",
        )

        assign(
            "visit",
            "visitIdentifier",
            "visit",
            "visitCode",
        )

        assign(
            "study_id",
            "studyIdentifier",
            "studyId",
        )

        assign(
            "series_id",
            "seriesIdentifier",
            "seriesId",
        )

        assign(
            "image_id",
            "imageUID",
            "imageId",
        )

        assign(
            "date_acquired",
            "dateAcquired",
            "acquisitionDate",
        )

        assign(
            "modality",
            "modality",
        )

        assign(
            "apoe_a1",
            "subjectInfo_APOE_A1",
            "subjectInfo_APOEA1",
            "APOE_A1",
            "apoeA1",
        )

        assign(
            "apoe_a2",
            "subjectInfo_APOE_A2",
            "subjectInfo_APOEA2",
            "APOE_A2",
            "apoeA2",
        )

        assign(
            "mmse",
            "assessmentScore_MMSCORE",
            "MMSCORE",
            "MMSE",
        )

        assign(
            "gds",
            "assessmentScore_GDTOTAL",
            "GDTOTAL",
            "GDS",
        )

        assign(
            "cdr",
            "assessmentScore_CDGLOBAL",
            "CDGLOBAL",
            "CDR",
        )

        if "diagnosis" in canonical:
            normalized_diagnosis = normalize_diagnosis(
                canonical["diagnosis"]
            )

            if normalized_diagnosis is not None:
                canonical["diagnosis"] = normalized_diagnosis

        return canonical, dict(flattened), None

    except ET.ParseError as exc:
        return {}, {}, f"XML parse error: {exc}"

    except OSError as exc:
        return {}, {}, f"XML read error: {exc}"

def infer_folder_diagnosis(
    dataset_root: Path,
    nifti_path: Path,
) -> Optional[str]:
    """Find CN, MCI, or AD in the NIfTI path."""
    try:
        relative_parts = nifti_path.relative_to(dataset_root).parts
    except ValueError:
        return None

    for part in relative_parts:
        normalized = part.upper()

        if normalized in KNOWN_CLASSES:
            return normalized

    return None

def infer_subject_folder(
    dataset_root: Path,
    nifti_path: Path,
) -> Optional[str]:
    """Infer a subject ID from common class/subject directory layouts."""
    try:
        relative_parts = nifti_path.relative_to(dataset_root).parts
    except ValueError:
        return None

    if len(relative_parts) >= 2:
        if relative_parts[0].upper() in KNOWN_CLASSES:
            return relative_parts[1]

    if relative_parts:
        return relative_parts[0]

    return None

def infer_processing_version(
    nifti_path: Path,
) -> Optional[str]:
    """Detect the processing version from the path."""
    path_text = str(nifti_path)

    for version in KNOWN_PROCESSING_VERSIONS:
        if version in path_text:
            return version

    for part in nifti_path.parts:
        if "__" in part or part.startswith("MT1"):
            return part

    return None

def remove_processing_components(path_text: str) -> str:
    """Remove processing-version components for fallback grouping."""
    parts = []

    for part in Path(path_text).parts:
        if part in KNOWN_PROCESSING_VERSIONS:
            continue

        if "__" in part or part.startswith("MT1"):
            continue

        parts.append(part)

    return "/".join(parts)

def make_acquisition_key(
    dataset_root: Path,
    nifti_path: Path,
    canonical: dict[str, str],
    subject_id: Optional[str],
) -> str:
    """
    Create a deterministic acquisition key.

    XML identifiers are preferred. A path-based hash is used only when
    sufficient XML identifiers are unavailable.
    """
    identifiers = [
        subject_id,
        canonical.get("date_acquired"),
        canonical.get("study_id"),
        canonical.get("series_id"),
    ]

    identifiers = [value for value in identifiers if value]

    if len(identifiers) >= 3:
        return "xml:" + "|".join(identifiers)

    relative_path = nifti_path.relative_to(dataset_root)
    cleaned_path = remove_processing_components(str(relative_path))
    raw_key = f"{cleaned_path}|{nifti_path.name}"

    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]

    return f"pathhash:{digest}"

def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
) -> dict:
    """
    Generic audit for layouts where NIfTI and XML files share a directory tree.

    The ADNI-specific mapper should be used for the current project because
    the MRI and clinical XML trees are separate.
    """
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and is_nifti(path)
    )

    xml_files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xml"
    )

    xml_by_parent: dict[Path, list[Path]] = defaultdict(list)

    for xml_path in xml_files:
        xml_by_parent[xml_path.parent].append(xml_path)

    records: list[MRIRecord] = []

    for nifti_path in nifti_files:
        subject_id = infer_subject_folder(dataset_root, nifti_path)
        folder_diagnosis = infer_folder_diagnosis(
            dataset_root,
            nifti_path,
        )
        processing_version = infer_processing_version(nifti_path)

        xml_path: Optional[Path] = None
        mapping_status = "no_xml_candidates"

        same_directory_xml = xml_by_parent.get(nifti_path.parent, [])

        if len(same_directory_xml) == 1:
            xml_path = same_directory_xml[0]
            mapping_status = "same_directory"

        elif len(same_directory_xml) > 1:
            xml_path = sorted(same_directory_xml)[0]
            mapping_status = "ambiguous_same_directory"

        canonical: dict[str, str] = {}
        flattened: dict[str, list[str]] = {}
        warnings: list[str] = []

        if xml_path is not None:
            canonical, flattened, xml_error = parse_xml(xml_path)

            if xml_error:
                warnings.append(xml_error)
        else:
            warnings.append("XML metadata was not mapped")

        xml_subject_id = canonical.get("subject_id")

        if xml_subject_id is not None:
            if subject_id is not None and xml_subject_id != subject_id:
                warnings.append(
                    "XML subject ID and folder subject ID do not match"
                )

            subject_id = xml_subject_id
        else:
            warnings.append(
                "Subject ID was obtained from the directory structure"
            )

        diagnosis_xml = canonical.get("diagnosis")

        if diagnosis_xml is None:
            warnings.append(
                "Diagnosis was not found or was not recognized in XML"
            )

        if diagnosis_xml and folder_diagnosis:
            if diagnosis_xml != folder_diagnosis:
                warnings.append(
                    "Folder diagnosis and XML diagnosis do not match"
                )

        acquisition_key = make_acquisition_key(
            dataset_root=dataset_root,
            nifti_path=nifti_path,
            canonical=canonical,
            subject_id=subject_id,
        )

        records.append(
            MRIRecord(
                nifti_path=str(nifti_path),
                xml_path=str(xml_path) if xml_path else None,
                xml_mapping_status=mapping_status,
                subject_id=subject_id,
                diagnosis_xml=diagnosis_xml,
                diagnosis_folder=folder_diagnosis,
                visit=canonical.get("visit"),
                study_id=canonical.get("study_id"),
                series_id=canonical.get("series_id"),
                image_id=canonical.get("image_id"),
                processing_version=processing_version,
                acquisition_key=acquisition_key,
                nifti_size_bytes=nifti_path.stat().st_size,
                metadata_json=json.dumps(
                    {
                        "canonical_fields": canonical,
                        "flattened_fields": flattened,
                    },
                    sort_keys=True,
                ),
                warnings_json=json.dumps(warnings),
            )
        )

    acquisition_groups: dict[str, list[MRIRecord]] = defaultdict(list)

    for record in records:
        acquisition_groups[record.acquisition_key].append(record)

    manifest_rows = [asdict(record) for record in records]

    write_csv(
        output_dir / "mri_manifest.csv",
        manifest_rows,
    )

    subject_groups: dict[str, list[MRIRecord]] = defaultdict(list)

    for record in records:
        subject_groups[record.subject_id or "MISSING_SUBJECT_ID"].append(
            record
        )

    subject_rows = []

    for subject_id, group in sorted(subject_groups.items()):
        diagnoses = sorted(
            {
                record.diagnosis_xml
                for record in group
                if record.diagnosis_xml
            }
        )

        folder_diagnoses = sorted(
            {
                record.diagnosis_folder
                for record in group
                if record.diagnosis_folder
            }
        )

        processing_versions = sorted(
            {
                record.processing_version
                for record in group
                if record.processing_version
            }
        )

        subject_rows.append(
            {
                "subject_id": subject_id,
                "nifti_count": len(group),
                "xml_diagnoses": "|".join(diagnoses),
                "folder_diagnoses": "|".join(folder_diagnoses),
                "processing_versions": "|".join(processing_versions),
            }
        )

    write_csv(
        output_dir / "subject_summary.csv",
        subject_rows,
    )

    processing_rows = []

    for acquisition_key, group in sorted(acquisition_groups.items()):
        processing_rows.append(
            {
                "acquisition_key": acquisition_key,
                "subject_id": group[0].subject_id,
                "visit": group[0].visit,
                "study_id": group[0].study_id,
                "series_id": group[0].series_id,
                "file_count": len(group),
                "processing_versions": "|".join(
                    sorted(
                        {
                            record.processing_version or "UNKNOWN"
                            for record in group
                        }
                    )
                ),
                "nifti_paths": "|".join(
                    record.nifti_path for record in group
                ),
            }
        )

    write_csv(
        output_dir / "acquisition_processing_groups.csv",
        processing_rows,
    )

    warning_rows = []

    for record in records:
        for warning in json.loads(record.warnings_json):
            warning_rows.append(
                {
                    "subject_id": record.subject_id,
                    "nifti_path": record.nifti_path,
                    "xml_path": record.xml_path,
                    "warning": warning,
                }
            )

    write_csv(
        output_dir / "warnings.csv",
        warning_rows,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "nifti_count": len(nifti_files),
        "xml_count": len(xml_files),
        "record_count": len(records),
        "unique_subject_count": len(subject_groups),
        "unique_acquisition_count": len(acquisition_groups),
        "diagnosis_xml_counts": dict(
            Counter(
                record.diagnosis_xml
                for record in records
                if record.diagnosis_xml
            )
        ),
        "diagnosis_folder_counts": dict(
            Counter(
                record.diagnosis_folder
                for record in records
                if record.diagnosis_folder
            )
        ),
        "processing_version_counts": dict(
            Counter(
                record.processing_version or "UNKNOWN"
                for record in records
            )
        ),
        "xml_mapping_status_counts": dict(
            Counter(
                record.xml_mapping_status
                for record in records
            )
        ),
        "warning_count": len(warning_rows),
    }

    with (output_dir / "audit_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary
