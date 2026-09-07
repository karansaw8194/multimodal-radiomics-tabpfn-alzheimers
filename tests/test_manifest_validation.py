from pathlib import Path

from scripts.validate_adni_manifest import validate_manifest

def test_manifest_validation_accepts_valid_manifest(
    tmp_path: Path,
) -> None:
    nifti_path = tmp_path / "scan.nii"
    xml_path = tmp_path / "metadata.xml"

    nifti_path.write_bytes(b"nifti")
    xml_path.write_text("<xml />", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"

    manifest_path.write_text(
        f"""nifti_path,xml_path,mapping_status,folder_class,subject_id,diagnosis_xml,diagnosis_folder,processing_version,acquisition_date,image_directory_id,image_id_xml,study_id,series_id,visit,age,sex,apoe_a1,apoe_a2,mmse,gds,cdr,weight,modality,date_acquired,warnings_json,flattened_metadata_json
{nifti_path},{xml_path},complete_flat_xml_mapping,AD,S001,AD,AD,MPR__GradWarp,2020-01-01,I1,1,STUDY1,SERIES1,baseline,70,F,3,4,23,2,1,65,MRI,2020-01-01,[],{{}}
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "validation"

    summary = validate_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )

    assert summary["record_count"] == 1
    assert summary["error_count"] == 0
    assert summary["safe_for_next_stage"] is True
