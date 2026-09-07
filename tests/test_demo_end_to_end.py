from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from adni_pipeline.demo_end_to_end import (
    run_end_to_end_demo,
)

def test_end_to_end_demo_creates_report(
    tmp_path: Path,
) -> None:
    project_root = tmp_path

    manifest_dir = (
        project_root
        / "reports"
        / "data_audit"
        / "selected_processing"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_mri = project_root / "raw_demo.nii.gz"
    raw_xml = project_root / "raw_demo.xml"

    image = nib.Nifti1Image(
        np.ones((10, 10, 10), dtype=np.float32),
        np.eye(4),
    )

    nib.save(image, raw_mri)
    raw_xml.write_text(
        "<metadata><researchGroup>CN</researchGroup></metadata>",
        encoding="utf-8",
    )

    manifest = pd.DataFrame(
        {
            "subject_id": ["S001"],
            "acquisition_group_id": ["G001"],
            "split": ["test"],
            "diagnosis_xml": ["CN"],
            "nifti_path": [str(raw_mri)],
            "xml_path": [str(raw_xml)],
            "processing_version": [
                "MPR__GradWarp"
            ],
        }
    )

    manifest.to_csv(
        manifest_dir / "selected_manifest.csv",
        index=False,
    )

    result = run_end_to_end_demo(
        project_root=project_root,
        output_dir=project_root / "demo",
        mode="multimodal",
        subject_id="S001",
        acquisition_group_id="G001",
        random_seed=42,
    )

    assert result["record"]["subject_id"] == "S001"
    assert (
        project_root
        / "demo"
        / "demo_report.html"
    ).exists()

    assert (
        project_root
        / "demo"
        / "images"
        / "axial.png"
    ).exists()
