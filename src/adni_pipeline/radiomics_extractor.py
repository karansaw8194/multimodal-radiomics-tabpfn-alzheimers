from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

PYRADIOMICS_AVAILABLE = False
try:
    from radiomics import featureextractor
    PYRADIOMICS_AVAILABLE = True
except Exception:
    PYRADIOMICS_AVAILABLE = False

@dataclass
class RadiomicsConfig:
    bin_width: float = 25.0
    enable_shape_3d: bool = True
    enable_first_order: bool = True
    enable_texture: bool = True

DEFAULT_FEATURE_KEYS = [
    "shape_Volume",
    "shape_VoxelCount",
    "shape_SurfaceArea",
    "shape_Elongation",
    "shape_Flatness",
    "shape_Sphericity",
    "firstorder_Mean",
    "firstorder_StandardDeviation",
    "firstorder_Minimum",
    "firstorder_Maximum",
    "firstorder_Variance",
    "firstorder_Skewness",
    "firstorder_Kurtosis",
    "firstorder_Median",
    "firstorder_P10",
    "firstorder_P90",
    "firstorder_IQR",
    "firstorder_RMS",
    "firstorder_Energy",
    "firstorder_Entropy",
    "glcm_TextureMean",
    "glcm_TextureStd",
]

def extract_features_simpleitk_native(
    image_path: Path,
    mask_path: Path,
    prefix: str = "",
    config: Optional[RadiomicsConfig] = None,
) -> dict[str, float]:
    """
    Native SimpleITK/NumPy 3D Radiomics Feature Extractor with empty-mask safety.
    """
    if config is None:
        config = RadiomicsConfig()

    # Pre-initialize all feature keys with 0.0
    features: dict[str, float] = {
        f"{prefix}_{key}": 0.0 for key in DEFAULT_FEATURE_KEYS
    }

    image_sitk = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    mask_sitk = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)

    # 1. Check if mask is non-empty
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_sitk)

    if 1 not in shape_filter.GetLabels():
        # Mask is empty; return zeroed features safely
        return features

    # Shape Features
    if config.enable_shape_3d:
        vol = float(shape_filter.GetPhysicalSize(1))
        area = float(shape_filter.GetPerimeter(1))

        features[f"{prefix}_shape_Volume"] = vol
        features[f"{prefix}_shape_VoxelCount"] = float(shape_filter.GetNumberOfPixels(1))
        features[f"{prefix}_shape_SurfaceArea"] = area
        features[f"{prefix}_shape_Elongation"] = float(shape_filter.GetElongation(1))
        features[f"{prefix}_shape_Flatness"] = float(shape_filter.GetFlatness(1))

        if vol > 0 and area > 0:
            sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * vol) ** (2.0 / 3.0)) / area
            features[f"{prefix}_shape_Sphericity"] = float(sphericity)

    # 2. First Order Features
    intensity_filter = sitk.LabelIntensityStatisticsImageFilter()
    intensity_filter.Execute(mask_sitk, image_sitk)

    if 1 in intensity_filter.GetLabels() and config.enable_first_order:
        features[f"{prefix}_firstorder_Mean"] = float(intensity_filter.GetMean(1))
        features[f"{prefix}_firstorder_StandardDeviation"] = float(intensity_filter.GetStandardDeviation(1))
        features[f"{prefix}_firstorder_Minimum"] = float(intensity_filter.GetMinimum(1))
        features[f"{prefix}_firstorder_Maximum"] = float(intensity_filter.GetMaximum(1))
        features[f"{prefix}_firstorder_Variance"] = float(intensity_filter.GetVariance(1))
        features[f"{prefix}_firstorder_Skewness"] = float(intensity_filter.GetSkewness(1))
        features[f"{prefix}_firstorder_Kurtosis"] = float(intensity_filter.GetKurtosis(1))

        img_arr = sitk.GetArrayFromImage(image_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk) > 0
        voxels = img_arr[mask_arr]

        if len(voxels) > 0:
            features[f"{prefix}_firstorder_Median"] = float(np.median(voxels))
            features[f"{prefix}_firstorder_P10"] = float(np.percentile(voxels, 10))
            features[f"{prefix}_firstorder_P90"] = float(np.percentile(voxels, 90))
            features[f"{prefix}_firstorder_IQR"] = float(
                np.percentile(voxels, 75) - np.percentile(voxels, 25)
            )
            features[f"{prefix}_firstorder_RMS"] = float(np.sqrt(np.mean(voxels ** 2)))
            features[f"{prefix}_firstorder_Energy"] = float(np.sum(voxels ** 2))

            hist, _ = np.histogram(voxels, bins=32)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            features[f"{prefix}_firstorder_Entropy"] = float(-np.sum(probs * np.log2(probs)))

    # 3. GLCM Texture Features
    if config.enable_texture:
        try:
            img_arr = sitk.GetArrayFromImage(image_sitk)
            mask_arr = sitk.GetArrayFromImage(mask_sitk) > 0

            if np.any(mask_arr):
                min_v, max_v = np.min(img_arr[mask_arr]), np.max(img_arr[mask_arr])
                if max_v > min_v:
                    scaled_arr = np.zeros_like(img_arr, dtype=np.uint8)
                    scaled_arr[mask_arr] = (
                        (img_arr[mask_arr] - min_v) / (max_v - min_v) * 31 + 1
                    ).astype(np.uint8)

                    scaled_sitk = sitk.GetImageFromArray(scaled_arr)
                    scaled_sitk.CopyInformation(image_sitk)

                    glcm = sitk.ScalarImageToCooccurrenceMatrixFilter()
                    glcm.SetInput(scaled_sitk)
                    glcm.SetMaskImage(mask_sitk)
                    glcm.Execute()

                    features[f"{prefix}_glcm_TextureMean"] = float(
                        np.mean(sitk.GetArrayFromImage(glcm.GetOutput()))
                    )
                    features[f"{prefix}_glcm_TextureStd"] = float(
                        np.std(sitk.GetArrayFromImage(glcm.GetOutput()))
                    )
        except Exception:
            pass

    return features

def extract_features_pyradiomics(
    image_path: Path,
    mask_path: Path,
    prefix: str = "",
    config: Optional[RadiomicsConfig] = None,
) -> dict[str, float]:
    """PyRadiomics Feature Extraction Wrapper."""
    if config is None:
        config = RadiomicsConfig()

    settings = {
        "binWidth": config.bin_width,
        "correctMask": True,
        "additionalInfo": False,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    extractor.disableAllFeatureClasses()

    if config.enable_shape_3d:
        extractor.enableFeatureClassByName("shape")
    if config.enable_first_order:
        extractor.enableFeatureClassByName("firstorder")
    if config.enable_texture:
        extractor.enableFeatureClassByName("glcm")
        extractor.enableFeatureClassByName("glrlm")
        extractor.enableFeatureClassByName("glszm")
        extractor.enableFeatureClassByName("gldm")
        extractor.enableFeatureClassByName("ngtdm")

    image_sitk = sitk.ReadImage(str(image_path))
    mask_sitk = sitk.ReadImage(str(mask_path))

    raw_features = extractor.execute(image_sitk, mask_sitk, label=1)

    cleaned_features: dict[str, float] = {}
    for key, val in raw_features.items():
        if key.startswith("diagnostics_"):
            continue
        feat_name = f"{prefix}_{key}" if prefix else key
        try:
            cleaned_features[feat_name] = float(val)
        except (ValueError, TypeError):
            continue

    return cleaned_features

def extract_features_for_pair(
    image_path: Path,
    mask_path: Path,
    prefix: str = "",
    config: Optional[RadiomicsConfig] = None,
) -> dict[str, float]:
    """Extract 3D Radiomics Features with fallback support."""
    if PYRADIOMICS_AVAILABLE:
        try:
            return extract_features_pyradiomics(
                image_path=image_path,
                mask_path=mask_path,
                prefix=prefix,
                config=config,
            )
        except Exception:
            pass

    return extract_features_simpleitk_native(
        image_path=image_path,
        mask_path=mask_path,
        prefix=prefix,
        config=config,
    )

def run_radiomics_extraction_pipeline(
    seg_catalog_path: Path,
    output_dir: Path,
    config: Optional[RadiomicsConfig] = None,
) -> pd.DataFrame:
    """Run 3D Radiomics Feature Extraction over all scans in segmentation catalog."""
    if not seg_catalog_path.exists():
        raise FileNotFoundError(f"Segmentation catalog missing: {seg_catalog_path}")

    if config is None:
        config = RadiomicsConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_df = pd.read_csv(seg_catalog_path)
    extracted_rows: list[dict[str, Any]] = []

    for _, row in catalog_df.iterrows():
        subject_id = str(row["subject_id"])
        group_id = str(row["acquisition_group_id"])
        split = str(row["split"])
        dx = str(row["diagnosis_xml"])

        prep_img_path = Path(row["preprocessed_nifti_path"])
        brain_mask_path = Path(row["brain_roi_mask_path"])
        subcortical_mask_path = Path(row["subcortical_roi_mask_path"])

        meta_row: dict[str, Any] = {
            "subject_id": subject_id,
            "acquisition_group_id": group_id,
            "split": split,
            "diagnosis_xml": dx,
        }

        # 1. Extract Whole Brain ROI Features
        brain_feats = extract_features_for_pair(
            image_path=prep_img_path,
            mask_path=brain_mask_path,
            prefix="brain",
            config=config,
        )

        # 2. Extract Subcortical ROI Features
        subcortical_feats = extract_features_for_pair(
            image_path=prep_img_path,
            mask_path=subcortical_mask_path,
            prefix="subcortical",
            config=config,
        )

        meta_row.update(brain_feats)
        meta_row.update(subcortical_feats)
        extracted_rows.append(meta_row)

    features_df = pd.DataFrame(extracted_rows)

    # Fill any remaining residual NaNs with 0.0 if present
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[num_cols] = features_df[num_cols].fillna(0.0)

    # Save CSV
    out_csv_path = output_dir / "radiomics_features.csv"
    features_df.to_csv(out_csv_path, index=False)

    num_feature_cols = len(
        [
            c
            for c in features_df.columns
            if c not in ["subject_id", "acquisition_group_id", "split", "diagnosis_xml"]
        ]
    )

    summary = {
        "total_scans_extracted": len(features_df),
        "total_radiomic_features": num_feature_cols,
        "engine_used": "PyRadiomics" if PYRADIOMICS_AVAILABLE else "SimpleITK_Native",
        "feature_csv_path": str(out_csv_path.resolve()),
    }

    with (output_dir / "radiomics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return features_df
