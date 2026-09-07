from pathlib import Path

import nibabel as nib
import numpy as np

from adni_pipeline.mri_visualization import (
    create_mri_html_report,
    create_mri_visualization,
)

def test_create_mri_visualization(
    tmp_path: Path,
) -> None:
    data = np.zeros(
        (20, 24, 28),
        dtype=np.float32,
    )

    data[5:15, 6:18, 7:21] = 100.0

    image = nib.Nifti1Image(
        data,
        np.diag(
            [
                2.0,
                2.0,
                2.0,
                1.0,
            ]
        ),
    )

    nifti_path = tmp_path / "test.nii.gz"
    output_dir = tmp_path / "visualization"

    nib.save(
        image,
        nifti_path,
    )

    result = create_mri_visualization(
        nifti_path=nifti_path,
        output_dir=output_dir,
    )

    assert result["shape"] == [20, 24, 28]
    assert result["orientation"] == "RAS"
    assert len(result["base64_images"]) == 3

    for plane in (
        "axial",
        "coronal",
        "sagittal",
    ):
        assert (
            output_dir / f"{plane}.png"
        ).exists()

    html_path = output_dir / "report.html"

    create_mri_html_report(
        visualization=result,
        output_path=html_path,
    )

    assert html_path.exists()
    assert "MRI Visualization" in html_path.read_text(
        encoding="utf-8"
    )

def test_mri_visualization_with_mask(
    tmp_path: Path,
) -> None:
    data = np.ones(
        (12, 12, 12),
        dtype=np.float32,
    )

    mask = np.zeros(
        (12, 12, 12),
        dtype=np.uint8,
    )

    mask[3:9, 3:9, 3:9] = 1

    affine = np.eye(4)

    nifti_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_dir = tmp_path / "overlay"

    nib.save(
        nib.Nifti1Image(data, affine),
        nifti_path,
    )

    nib.save(
        nib.Nifti1Image(mask, affine),
        mask_path,
    )

    result = create_mri_visualization(
        nifti_path=nifti_path,
        mask_path=mask_path,
        output_dir=output_dir,
    )

    assert result["mask_voxel_count"] == int(mask.sum())
    assert result["mask_path"] is not None
