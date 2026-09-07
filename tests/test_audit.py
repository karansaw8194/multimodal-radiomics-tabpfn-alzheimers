from pathlib import Path

from adni_audit.audit import (
    audit_dataset,
    infer_folder_diagnosis,
    infer_processing_version,
    normalize_diagnosis,
    parse_xml,
)

def test_normalize_diagnosis() -> None:
    assert normalize_diagnosis("CN") == "CN"
    assert normalize_diagnosis("Mild Cognitive Impairment") == "MCI"
    assert normalize_diagnosis("Alzheimer's Disease") == "AD"
    assert normalize_diagnosis("unknown diagnosis") is None

def test_processing_version_detection() -> None:
    path = Path(
        "/data/CN/S001/MPR__GradWarp__B1_Correction__N3/image.nii.gz"
    )

    assert (
        infer_processing_version(path)
        == "MPR__GradWarp__B1_Correction__N3"
    )

def test_folder_diagnosis_detection(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    nifti_path = dataset_root / "MCI" / "S001" / "image.nii.gz"

    assert infer_folder_diagnosis(dataset_root, nifti_path) == "MCI"

def test_xml_parsing(tmp_path: Path) -> None:
    xml_path = tmp_path / "metadata.xml"

    xml_path.write_text(
        """
        <Metadata>
            <SubjectID>S001</SubjectID>
            <Diagnosis>MCI</Diagnosis>
            <Age>72</Age>
            <Sex>F</Sex>
            <Visit>m06</Visit>
        </Metadata>
        """,
        encoding="utf-8",
    )

    canonical, flattened, error = parse_xml(xml_path)

    assert error is None
    assert canonical["subject_id"] == "S001"
    assert canonical["diagnosis"] == "MCI"
    assert canonical["age"] == "72"
    assert canonical["sex"] == "F"
    assert canonical["visit"] == "m06"
    assert "diagnosis" in flattened

def test_audit_does_not_modify_raw_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    subject_dir = dataset_root / "CN" / "S001" / "MPR__GradWarp"

    subject_dir.mkdir(parents=True)

    nifti_path = subject_dir / "scan.nii.gz"
    xml_path = dataset_root / "CN" / "S001" / "metadata.xml"

    nifti_path.write_bytes(b"test-nifti-placeholder")

    xml_path.write_text(
        """
        <Metadata>
            <SubjectID>S001</SubjectID>
            <Diagnosis>CN</Diagnosis>
            <Visit>bl</Visit>
            <StudyID>STUDY001</StudyID>
            <SeriesID>SERIES001</SeriesID>
        </Metadata>
        """,
        encoding="utf-8",
    )

    original_nifti_bytes = nifti_path.read_bytes()

    output_dir = tmp_path / "audit"

    summary = audit_dataset(dataset_root, output_dir)

    assert summary["nifti_count"] == 1
    assert summary["xml_count"] == 1
    assert summary["unique_subject_count"] == 1
    assert nifti_path.read_bytes() == original_nifti_bytes
    assert (output_dir / "mri_manifest.csv").exists()
    assert (output_dir / "subject_summary.csv").exists()
    assert (output_dir / "audit_summary.json").exists()
