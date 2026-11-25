from pathlib import Path

from adni_audit.audit import (
    audit_dataset,
    infer_folder_diagnosis,
    infer_processing_version,
    normalize_diagnosis,
    parse_xml,
)
from adni_audit.adni_mapper import (
    audit_adni_layout,
    find_xml_for_nifti,
    identify_acquisition_date,
    identify_image_directory_id,
    identify_processing_version,
    identify_subject_id,
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


def test_subject_first_layout_maps_root_complete_xml(tmp_path: Path) -> None:
    mri_root = tmp_path / "mri"
    clinical_root = tmp_path / "metadata"
    nifti_path = (
        mri_root
        / "941_S_1311"
        / "MPR__GradWarp__B1_Correction__N3"
        / "2009-03-31_14_45_23.0"
        / "I143860"
        / "scan.nii"
    )
    nifti_path.parent.mkdir(parents=True)
    nifti_path.write_bytes(b"test-nifti-placeholder")

    clinical_root.mkdir()
    xml_path = clinical_root / (
        "ADNI_941_S_1311_MPR__GradWarp__B1_Correction__N3_"
        "S65346_I143860.xml"
    )
    xml_path.write_text(
        """
        <idaxs>
          <subject><subjectIdentifier>941_S_1311</subjectIdentifier>
          <researchGroup>MCI</researchGroup></subject>
          <study><studyIdentifier>20664</studyIdentifier>
          <series><seriesIdentifier>65346</seriesIdentifier>
          <derivedProduct><imageUID>143860</imageUID></derivedProduct>
          </series></study>
        </idaxs>
        """,
        encoding="utf-8",
    )

    assert identify_subject_id(nifti_path, mri_root) == "941_S_1311"
    assert (
        identify_processing_version(nifti_path, mri_root)
        == "MPR__GradWarp__B1_Correction__N3"
    )
    assert (
        identify_acquisition_date(nifti_path, mri_root)
        == "2009-03-31_14_45_23.0"
    )
    assert identify_image_directory_id(nifti_path, mri_root) == "I143860"

    mapped_xml, status, folder_class = find_xml_for_nifti(
        nifti_path,
        mri_root,
        clinical_root,
    )

    assert mapped_xml == xml_path
    assert status == "complete_flat_xml_mapping"
    assert folder_class is None

    output_dir = tmp_path / "audit"
    summary = audit_adni_layout(mri_root, clinical_root, output_dir)

    assert summary["mapping_status_counts"] == {
        "complete_flat_xml_mapping": 1
    }
    manifest = (output_dir / "adni_mri_xml_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert "941_S_1311" in manifest
    assert "MCI" in manifest
    assert "143860" in manifest
