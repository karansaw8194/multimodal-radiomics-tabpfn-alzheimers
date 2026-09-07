from pathlib import Path
import nibabel as nib
import numpy as np

from adni_pipeline.preprocessing import (
    get_orientation_codes,
    reorient_to_ras,
    zscore_normalize_nonzero,
    preprocess_single_nifti,
)

def test_zscore_normalize_nonzero() -> None:
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:8, 2:8, 2:8] = np.random.uniform(10.0, 50.0, size=(6, 6, 6))

    norm_data, mean_val, std_val = zscore_normalize_nonzero(data)

    assert mean_val > 0
    assert std_val > 0
    assert np.all(norm_data[data == 0] == 0)
    assert np.isclose(np.mean(norm_data[data > 0]), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm_data[data > 0]), 1.0, atol=1e-5)

def test_preprocess_single_nifti_flow(tmp_path: Path) -> None:
    # Create synthetic 3D NIfTI
    array_data = np.zeros((20, 20, 20), dtype=np.float32)
    array_data[5:15, 5:15, 5:15] = 100.0
    affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm spacing
    synthetic_img = nib.Nifti1Image(array_data, affine)

    raw_path = tmp_path / "synthetic.nii.gz"
    out_path = tmp_path / "output.nii.gz"
    nib.save(synthetic_img, str(raw_path))

    out_img, meta = preprocess_single_nifti(
        nifti_path=raw_path,
        output_path=out_path,
        target_spacing=(1.0, 1.0, 1.0),
    )

    assert out_path.exists()
    assert meta["preprocessed_orientation"] == "RAS"
    assert meta["preprocessed_zooms"] == "(1.0, 1.0, 1.0)"
    assert out_img.shape[0] == 40  # 20 voxels @ 2mm -> 40 voxels @ 1mm
