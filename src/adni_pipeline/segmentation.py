from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

@dataclass
class SegmentationRecord:
    subject_id: str
    acquisition_group_id: str
    split: str
    diagnosis_xml: str
    preprocessed_nifti_path: str
    multiclass_mask_path: str
    brain_roi_mask_path: str
    subcortical_roi_mask_path: str
    brain_volume_mm3: float
    subcortical_volume_mm3: float
    ventricle_volume_mm3: float

@dataclass
class SegmentationMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    hd95: float

def compute_hd95(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance (HD95) in voxel units."""
    coords_gt = np.argwhere(mask_gt > 0)
    coords_pred = np.argwhere(mask_pred > 0)

    if len(coords_gt) == 0 or len(coords_pred) == 0:
        return 0.0

    # Directed Hausdorff distances
    d_gt_pred = [
        np.min(np.linalg.norm(pt - coords_pred, axis=1))
        for pt in coords_gt[:: max(1, len(coords_gt) // 500)]
    ]
    d_pred_gt = [
        np.min(np.linalg.norm(pt - coords_gt, axis=1))
        for pt in coords_pred[:: max(1, len(coords_pred) // 500)]
    ]

    all_distances = np.concatenate([d_gt_pred, d_pred_gt])
    return float(np.percentile(all_distances, 95))

def compute_segmentation_metrics(
    mask_gt: np.ndarray, mask_pred: np.ndarray
) -> SegmentationMetrics:
    """Compute Dice, IoU, Precision, Recall, Sensitivity, Specificity, and HD95."""
    gt_bin = (mask_gt > 0).astype(bool)
    pred_bin = (mask_pred > 0).astype(bool)

    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(~gt_bin, pred_bin).sum()
    fn = np.logical_and(gt_bin, ~pred_bin).sum()
    tn = np.logical_and(~gt_bin, ~pred_bin).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    hd95_val = compute_hd95(gt_bin, pred_bin)

    return SegmentationMetrics(
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        hd95=float(hd95_val),
    )

class BrainSegmentationBackend(ABC):
    """Abstract Base Class for Brain Segmentation Backends."""

    @abstractmethod
    def segment(self, image_data: np.ndarray) -> np.ndarray:
        """
        Input: 3D float image array.
        Output: 3D int array with ROI label masks:
            0: Background
            1: Brain Parenchyma / Cortex
            2: Subcortical / Hippocampal ROI candidate
            3: Ventricles / CSF Space
        """
        pass

class OtsuMorphologicalBackend(BrainSegmentationBackend):
    """Robust 3D Skull-Stripping and Tissue Segmentation Backend."""

    def segment(self, image_data: np.ndarray) -> np.ndarray:
        # Non-zero tissue mask
        tissue_mask = image_data != 0
        if not np.any(tissue_mask):
            return np.zeros_like(image_data, dtype=np.int16)

        # 3D Morphological skull stripping
        cleaned_tissue = ndimage.binary_opening(tissue_mask, iterations=2)
        cleaned_tissue = ndimage.binary_closing(cleaned_tissue, iterations=2)

        # Extract largest connected component (Brain Volume)
        labeled, num_features = ndimage.label(cleaned_tissue)
        if num_features == 0:
            return np.zeros_like(image_data, dtype=np.int16)

        sizes = ndimage.sum(cleaned_tissue, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        brain_mask = labeled == largest_label

        # Fill holes in brain mask
        brain_mask = ndimage.binary_fill_holes(brain_mask)

        # Tissue intensity thresholds within brain mask
        brain_intensities = image_data[brain_mask]
        q30, q85 = np.percentile(brain_intensities, [30, 85])

        out_mask = np.zeros_like(image_data, dtype=np.int16)
        # Label 1: Main Brain Cortex/Parenchyma
        out_mask[brain_mask] = 1

        # Label 2: High-intensity Subcortical/Deep Grey structure candidate
        out_mask[brain_mask & (image_data >= q85)] = 2

        # Label 3: Low-intensity Ventricle / Inner CSF candidate
        out_mask[brain_mask & (image_data <= q30)] = 3

        return out_mask

class DeepLearningBackend(BrainSegmentationBackend):
    """Interface Wrapper for external nnU-Net / MedSAM Models."""

    def __init__(self, model_weights_path: Optional[Path] = None):
        self.model_weights_path = model_weights_path

    def segment(self, image_data: np.ndarray) -> np.ndarray:
        if self.model_weights_path is None or not self.model_weights_path.exists():
            # Fallback to morphological backend if weights are missing
            fallback = OtsuMorphologicalBackend()
            return fallback.segment(image_data)

        # Placeholder for external model inference call
        # e.g., predictions = medsam_predict(image_data, self.model_weights_path)
        raise NotImplementedError("Deep learning weights path must be loaded.")

def run_segmentation_pipeline(
    catalog_path: Path,
    seg_output_dir: Path,
    roi_output_dir: Path,
    backend: Optional[BrainSegmentationBackend] = None,
) -> list[SegmentationRecord]:
    """Execute segmentation and ROI mask extraction for all preprocessed scans."""
    if backend is None:
        backend = OtsuMorphologicalBackend()

    seg_output_dir.mkdir(parents=True, exist_ok=True)
    roi_output_dir.mkdir(parents=True, exist_ok=True)

    with catalog_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    records: list[SegmentationRecord] = []

    for row in rows:
        subject_id = row["subject_id"]
        group_id = row["acquisition_group_id"]
        split = row["split"]
        dx = row["diagnosis_xml"]
        prep_path = Path(row["preprocessed_nifti_path"])

        img = nib.load(str(prep_path))
        data = img.get_fdata(dtype=np.float32)

        # Voxel volume in mm3
        zooms = img.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(zooms))

        # Run Backend Segmentation
        multiclass_mask = backend.segment(data)

        # Compute volumes
        brain_vol_mm3 = float(np.sum(multiclass_mask > 0) * voxel_vol_mm3)
        subcortical_vol_mm3 = float(np.sum(multiclass_mask == 2) * voxel_vol_mm3)
        ventricle_vol_mm3 = float(np.sum(multiclass_mask == 3) * voxel_vol_mm3)

        # Save Multiclass Mask
        mask_filename = f"{subject_id}_{group_id}_multiclass_mask.nii.gz"
        mask_path = seg_output_dir / split / mask_filename
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        mask_img = nib.Nifti1Image(multiclass_mask.astype(np.int16), img.affine, img.header)
        nib.save(mask_img, str(mask_path))

        # Save Binary Whole Brain ROI Mask
        brain_roi = (multiclass_mask > 0).astype(np.uint8)
        brain_roi_filename = f"{subject_id}_{group_id}_brain_roi.nii.gz"
        brain_roi_path = roi_output_dir / split / brain_roi_filename
        brain_roi_path.parent.mkdir(parents=True, exist_ok=True)

        brain_roi_img = nib.Nifti1Image(brain_roi, img.affine, img.header)
        nib.save(brain_roi_img, str(brain_roi_path))

        # Save Binary Subcortical ROI Mask
        subcortical_roi = (multiclass_mask == 2).astype(np.uint8)
        subcortical_roi_filename = f"{subject_id}_{group_id}_subcortical_roi.nii.gz"
        subcortical_roi_path = roi_output_dir / split / subcortical_roi_filename

        subcortical_roi_img = nib.Nifti1Image(subcortical_roi, img.affine, img.header)
        nib.save(subcortical_roi_img, str(subcortical_roi_path))

        rec = SegmentationRecord(
            subject_id=subject_id,
            acquisition_group_id=group_id,
            split=split,
            diagnosis_xml=dx,
            preprocessed_nifti_path=str(prep_path.resolve()),
            multiclass_mask_path=str(mask_path.resolve()),
            brain_roi_mask_path=str(brain_roi_path.resolve()),
            subcortical_roi_mask_path=str(subcortical_roi_path.resolve()),
            brain_volume_mm3=brain_vol_mm3,
            subcortical_volume_mm3=subcortical_vol_mm3,
            ventricle_volume_mm3=ventricle_vol_mm3,
        )
        records.append(rec)

    # Save Segmentation Catalog CSV
    catalog_out = seg_output_dir / "segmentation_catalog.csv"
    with catalog_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    # Save Summary JSON
    summary = {
        "total_scans_segmented": len(records),
        "segmentation_catalog": str(catalog_out.resolve()),
        "multiclass_mask_dir": str(seg_output_dir.resolve()),
        "roi_mask_dir": str(roi_output_dir.resolve()),
    }
    with (seg_output_dir / "segmentation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return records
