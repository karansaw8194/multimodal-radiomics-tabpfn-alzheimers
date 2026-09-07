from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

@dataclass
class PreprocessingRecord:
    subject_id: str
    acquisition_group_id: str
    split: str
    diagnosis_xml: str
    raw_nifti_path: str
    preprocessed_nifti_path: str
    original_shape: str
    preprocessed_shape: str
    original_zooms: str
    preprocessed_zooms: str
    original_orientation: str
    preprocessed_orientation: str
    nonzero_mean: float
    nonzero_std: float

def get_orientation_codes(img: nib.Nifti1Image) -> str:
    """Return 3-letter orientation string (e.g. RAS, LPI)."""
    return "".join(nib.orientations.aff2axcodes(img.affine))

def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient NIfTI volume to standard RAS coordinate system."""
    canonical_img = nib.as_closest_canonical(img)
    return canonical_img

def resample_volume(
    img: nib.Nifti1Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> nib.Nifti1Image:
    """
    Resample NIfTI volume to target isotropic voxel spacing in mm.

    Interpolation order 1 is trilinear, suitable for intensity images.
    """
    data = img.get_fdata(dtype=np.float32)
    current_zooms = img.header.get_zooms()[:3]

    if np.allclose(current_zooms, target_spacing, atol=1e-2):
        return img

    zoom_factors = [
        cur / tgt for cur, tgt in zip(current_zooms, target_spacing)
    ]

    # Resample 3D array data
    from scipy.ndimage import zoom

    resampled_data = zoom(data, zoom_factors, order=order)

    # Adjust affine matrix for new voxel spacing
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[:3, i] = new_affine[:3, i] * (
            current_zooms[i] / target_spacing[i]
        )

    resampled_img = nib.Nifti1Image(resampled_data, new_affine, img.header)
    resampled_img.header.set_zooms(target_spacing)

    return resampled_img

def zscore_normalize_nonzero(
    data: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Apply per-volume z-score normalization on non-zero tissue voxels.

    Background zeros are preserved to avoid shifting empty space.
    """
    nonzero_mask = data > 0
    if not np.any(nonzero_mask):
        # Fallback if volume is all zeros
        return data.astype(np.float32), 0.0, 1.0

    mean_val = float(np.mean(data[nonzero_mask]))
    std_val = float(np.std(data[nonzero_mask]))

    if std_val < 1e-8:
        std_val = 1.0

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[nonzero_mask] = (
        data[nonzero_mask] - mean_val
    ) / std_val

    return normalized, mean_val, std_val

def preprocess_single_nifti(
    nifti_path: Path,
    output_path: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[nib.Nifti1Image, dict]:
    """Preprocess a single NIfTI file: Orient to RAS -> Resample -> Normalize."""
    img = nib.load(str(nifti_path))

    orig_shape = img.shape[:3]
    orig_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    orig_orient = get_orientation_codes(img)

    # 1. Reorient to RAS
    ras_img = reorient_to_ras(img)

    # 2. Resample to Target Spacing (1.0, 1.0, 1.0)
    resampled_img = resample_volume(
        ras_img, target_spacing=target_spacing, order=1
    )

    # 3. Z-score normalize non-zero voxels
    data = resampled_img.get_fdata(dtype=np.float32)
    norm_data, nonzero_mean, nonzero_std = zscore_normalize_nonzero(data)

    out_img = nib.Nifti1Image(
        norm_data, resampled_img.affine, resampled_img.header
    )

    out_shape = out_img.shape[:3]
    out_zooms = tuple(float(z) for z in out_img.header.get_zooms()[:3])
    out_orient = get_orientation_codes(out_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))

    metadata = {
        "original_shape": str(orig_shape),
        "preprocessed_shape": str(out_shape),
        "original_zooms": str(orig_zooms),
        "preprocessed_zooms": str(out_zooms),
        "original_orientation": orig_orient,
        "preprocessed_orientation": out_orient,
        "nonzero_mean": nonzero_mean,
        "nonzero_std": nonzero_std,
    }

    return out_img, metadata

def run_preprocessing_pipeline(
    manifest_path: Path,
    output_dir: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[PreprocessingRecord]:
    """Process all scans listed in selected_manifest.csv."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    records: list[PreprocessingRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    for i, row in enumerate(rows, start=1):
        raw_path = Path(row["nifti_path"]).resolve()
        subject_id = row["subject_id"]
        group_id = row.get("acquisition_group_id", f"group_{i:04d}")
        split = row.get("split", "unassigned")
        dx = row.get("diagnosis_xml", "UNKNOWN")

        # Destination filename
        out_filename = f"{subject_id}_{group_id}_preprocessed.nii.gz"
        out_path = output_dir / split / out_filename

        _, meta = preprocess_single_nifti(
            nifti_path=raw_path,
            output_path=out_path,
            target_spacing=target_spacing,
        )

        rec = PreprocessingRecord(
            subject_id=subject_id,
            acquisition_group_id=group_id,
            split=split,
            diagnosis_xml=dx,
            raw_nifti_path=str(raw_path),
            preprocessed_nifti_path=str(out_path.resolve()),
            original_shape=meta["original_shape"],
            preprocessed_shape=meta["preprocessed_shape"],
            original_zooms=meta["original_zooms"],
            preprocessed_zooms=meta["preprocessed_zooms"],
            original_orientation=meta["original_orientation"],
            preprocessed_orientation=meta["preprocessed_orientation"],
            nonzero_mean=meta["nonzero_mean"],
            nonzero_std=meta["nonzero_std"],
        )
        records.append(rec)

    # Save Catalog CSV
    catalog_path = output_dir / "preprocessing_catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(records[0]).keys())
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    # Save Summary JSON
    summary = {
        "total_scans_processed": len(records),
        "target_spacing": list(target_spacing),
        "output_directory": str(output_dir.resolve()),
        "catalog_csv": str(catalog_path.resolve()),
    }
    with (output_dir / "preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return records
