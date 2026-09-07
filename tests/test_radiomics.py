from pathlib import Path
import numpy as np
import SimpleITK as sitk

from adni_pipeline.radiomics_extractor import (
    RadiomicsConfig,
    extract_features_for_pair,
)

def test_extract_features_for_pair(tmp_path: Path) -> None:
    # Synthetic 3D Image
    img_arr = np.random.uniform(10.0, 100.0, size=(15, 15, 15)).astype(np.float32)
    mask_arr = np.zeros((15, 15, 15), dtype=np.uint8)
    mask_arr[3:12, 3:12, 3:12] = 1

    img_sitk = sitk.GetImageFromArray(img_arr)
    mask_sitk = sitk.GetImageFromArray(mask_arr)

    img_path = tmp_path / "test_img.nii.gz"
    mask_path = tmp_path / "test_mask.nii.gz"

    sitk.WriteImage(img_sitk, str(img_path))
    sitk.WriteImage(mask_sitk, str(mask_path))

    config = RadiomicsConfig()
    features = extract_features_for_pair(
        image_path=img_path,
        mask_path=mask_path,
        prefix="test",
        config=config,
    )

    assert len(features) >= 10
    assert any("firstorder" in k for k in features.keys())
    assert any("shape" in k for k in features.keys())
