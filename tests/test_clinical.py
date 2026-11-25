from pathlib import Path
import pandas as pd
from adni_pipeline.clinical_processor import (
    DIAGNOSIS_LABEL_MAP,
    attach_acquisition_metadata,
    parse_numeric,
    parse_sex_binary,
    process_clinical_manifest,
)

def test_parse_helpers() -> None:
    assert parse_numeric("71.5") == 71.5
    assert pd.isna(parse_numeric(""))
    assert parse_sex_binary("M") == 0.0
    assert parse_sex_binary("F") == 1.0
    assert pd.isna(parse_sex_binary("unknown"))

def test_process_clinical_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "test_manifest.csv"
    out_dir = tmp_path / "clinical"

    data = {
        "subject_id": ["S001", "S002"],
        "acquisition_group_id": ["g1", "g2"],
        "split": ["train", "test"],
        "diagnosis_xml": ["CN", "AD"],
        "age": ["70.0", "75.0"],
        "sex": ["M", "F"],
        "apoe_a1": ["3", "4"],
        "apoe_a2": ["3", "4"],
        "mmse": ["29.0", "18.0"],
        "gds": ["1.0", "5.0"],
        "cdr": ["0.0", "1.0"],
        "weight": ["70.0", "65.0"],
    }
    pd.DataFrame(data).to_csv(manifest_path, index=False)

    df = process_clinical_manifest(manifest_path, out_dir)

    assert len(df) == 2
    assert "target_label" in df.columns
    assert df.loc[0, "target_label"] == 0  # CN
    assert df.loc[1, "target_label"] == 2  # AD
    assert df.loc[0, "clin_sex"] == 0.0
    assert df.loc[1, "clin_sex"] == 1.0


def test_attach_acquisition_metadata_uses_canonical_nifti_path(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "subject_id": ["S001", "S002"],
            "diagnosis_xml": ["CN", "AD"],
            "nifti_path": ["/raw/second.nii", "/raw/first.nii"],
        }
    )
    catalog_path = tmp_path / "preprocessing_catalog.csv"
    pd.DataFrame(
        {
            "subject_id": ["S002", "S001"],
            "diagnosis_xml": ["AD", "CN"],
            "raw_nifti_path": ["/raw/first.nii", "/raw/second.nii"],
            "acquisition_group_id": ["group_0001", "group_0002"],
            "split": ["train", "test"],
        }
    ).to_csv(catalog_path, index=False)

    aligned = attach_acquisition_metadata(manifest, catalog_path)

    assert aligned["acquisition_group_id"].tolist() == ["group_0002", "group_0001"]
    assert aligned["split"].tolist() == ["test", "train"]
