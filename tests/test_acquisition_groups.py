from pathlib import Path

from scripts.audit_acquisition_groups import audit_groups

def test_processing_versions_are_grouped(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    split_manifest_path = tmp_path / "manifest_with_splits.csv"
    output_dir = tmp_path / "groups"

    header = (
        "nifti_path,xml_path,subject_id,diagnosis_xml,"
        "processing_version,acquisition_date,date_acquired,"
        "study_id,series_id,image_directory_id,image_id_xml\n"
    )

    rows = [
        (
            "/data/a.nii,/data/a.xml,S001,AD,"
            "MPR__GradWarp,DATE1,2020-01-01,STUDY1,SERIES1,I1,1\n"
        ),
        (
            "/data/b.nii,/data/b.xml,S001,AD,"
            "MPR__GradWarp__B1_Correction,DATE1,2020-01-01,"
            "STUDY1,SERIES1,I2,2\n"
        ),
        (
            "/data/c.nii,/data/c.xml,S002,CN,"
            "MPR__GradWarp,DATE2,2020-01-02,STUDY2,SERIES2,I3,3\n"
        ),
    ]

    manifest_path.write_text(
        header + "".join(rows),
        encoding="utf-8",
    )

    split_manifest_path.write_text(
        header.replace(
            "image_id_xml\n",
            "image_id_xml,split\n",
        )
        + "".join(
            row.replace("\n", ",train\n")
            if "S002" not in row
            else row.replace("\n", ",test\n")
            for row in rows
        ),
        encoding="utf-8",
    )

    summary = audit_groups(
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        output_dir=output_dir,
    )

    assert summary["record_count"] == 3
    assert summary["acquisition_group_count"] == 2
    assert summary["problem_count"] == 0

    groups_path = output_dir / "acquisition_groups.csv"
    assert groups_path.exists()

    text = groups_path.read_text(encoding="utf-8")
    assert "MPR__GradWarp|MPR__GradWarp__B1_Correction" in text
