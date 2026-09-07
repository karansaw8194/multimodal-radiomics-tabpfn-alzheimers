from pathlib import Path

import pandas as pd

from adni_pipeline.demo_pipeline import (
    choose_demo_record,
    write_demo_outputs,
)

def test_choose_demo_record(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"

    dataframe = pd.DataFrame(
        {
            "subject_id": ["S001"],
            "acquisition_group_id": ["G001"],
            "split": ["test"],
            "diagnosis_xml": ["AD"],
            "nifti_path": ["/tmp/example.nii"],
            "xml_path": ["/tmp/example.xml"],
            "processing_version": [
                "MPR__GradWarp"
            ],
        }
    )

    dataframe.to_csv(
        manifest_path,
        index=False,
    )

    result = choose_demo_record(
        manifest_path=manifest_path,
        subject_id="S001",
        acquisition_group_id="G001",
        random_seed=42,
    )

    assert result["subject_id"] == "S001"
    assert result["acquisition_group_id"] == "G001"

def test_write_demo_outputs(tmp_path: Path) -> None:
    result = {
        "demo_mode": "clinical",
        "record": {
            "subject_id": "S001",
            "acquisition_group_id": "G001",
            "split": "test",
            "diagnosis_xml": "AD",
            "processing_version": "MPR__GradWarp",
        },
        "stages": [
            {
                "stage": "Test stage",
                "status": "COMPLETE",
                "details": {"value": 1},
            }
        ],
        "completed_stage_count": 1,
        "total_stage_count": 1,
        "raw_data_modified": False,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
    }

    paths = write_demo_outputs(
        result=result,
        output_dir=tmp_path,
    )

    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["html"]).exists()
