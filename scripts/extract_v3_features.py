#!/usr/bin/env python3
"""
scripts/extract_v3_features.py
------------------------------
Parallel feature extraction engine across 415 deduplicated scans from
reports/data_audit/v3_clean_manifest.csv.

Extracts:
1. Brain & Subcortical segmentation via OtsuMorphologicalBackend
2. 44 3D Radiomics features (Whole Brain ROI + Subcortical ROI)
3. 2 Added Volumetric Biomarkers:
   - ratio_ventricle_to_brain
   - ratio_subcortical_to_brain
4. Merges with 8 clinical XML variables
5. Produces data/clean_v3/multimodal_features.csv
"""

from __future__ import annotations
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adni_pipeline.preprocessing import preprocess_single_nifti
from adni_pipeline.segmentation import OtsuMorphologicalBackend
from adni_pipeline.radiomics_extractor import extract_features_for_pair, RadiomicsConfig

MANIFEST_PATH = PROJECT_ROOT / "reports" / "data_audit" / "v3_clean_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean_v3"
CACHE_DIR = OUTPUT_DIR / "feature_cache"


def process_single_scan(row_dict: dict) -> dict | None:
    image_id = str(row_dict["image_id"])
    cache_file = CACHE_DIR / f"{image_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    nifti_path = Path(row_dict["nifti_path"])
    if not nifti_path.exists():
        print(f"Error: {nifti_path} not found.")
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        prep_path = tmp_path / "prep.nii.gz"

        try:
            out_img, meta = preprocess_single_nifti(
                nifti_path=nifti_path,
                output_path=prep_path,
                target_spacing=(1.0, 1.0, 1.0),
            )
            prep_data = out_img.get_fdata(dtype=np.float32)

            backend = OtsuMorphologicalBackend()
            multiclass_mask = backend.segment(prep_data)

            # Volumetric Ratios
            total_brain_voxels = float(np.sum(multiclass_mask > 0))
            ventricle_voxels = float(np.sum(multiclass_mask == 3))
            subcortical_voxels = float(np.sum(multiclass_mask == 2))

            ratio_ventricle = float(ventricle_voxels / max(1.0, total_brain_voxels))
            ratio_subcortical = float(subcortical_voxels / max(1.0, total_brain_voxels))

            # Save ROI masks with identical header to avoid SimpleITK physical space mismatch
            brain_mask = (multiclass_mask > 0).astype(np.uint8)
            subc_mask = (multiclass_mask == 2).astype(np.uint8)

            brain_mask_path = tmp_path / "brain_roi.nii.gz"
            subc_mask_path = tmp_path / "subcortical_roi.nii.gz"

            nib.save(nib.Nifti1Image(brain_mask, out_img.affine, out_img.header), str(brain_mask_path))
            nib.save(nib.Nifti1Image(subc_mask, out_img.affine, out_img.header), str(subc_mask_path))

            rad_config = RadiomicsConfig()
            brain_feats = extract_features_for_pair(
                image_path=prep_path,
                mask_path=brain_mask_path,
                prefix="brain",
                config=rad_config,
            )
            subc_feats = extract_features_for_pair(
                image_path=prep_path,
                mask_path=subc_mask_path,
                prefix="subcortical",
                config=rad_config,
            )

            record = {
                "subject_id": row_dict["subject_id"],
                "image_id": image_id,
                "date_acquired": row_dict.get("date_acquired", ""),
                "diagnosis": row_dict["diagnosis"],
                "target_label": int(row_dict["target_label"]),
                "split": row_dict["split"],
                # Added Volume Ratios
                "ratio_ventricle_to_brain": ratio_ventricle,
                "ratio_subcortical_to_brain": ratio_subcortical,
                # Clinical variables
                "clin_age": row_dict.get("clin_age"),
                "clin_sex": row_dict.get("clin_sex"),
                "clin_weight": row_dict.get("clin_weight"),
                "clin_apoe_a1": row_dict.get("clin_apoe_a1"),
                "clin_apoe_a2": row_dict.get("clin_apoe_a2"),
                "clin_mmse": row_dict.get("clin_mmse"),
                "clin_cdr": row_dict.get("clin_cdr"),
                "clin_gds": row_dict.get("clin_gds"),
                # 44 Radiomics features
                **brain_feats,
                **subc_feats,
            }

            with open(cache_file, "w") as f:
                json.dump(record, f)

            return record
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            return None


def main():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest loaded: {len(df_manifest)} scans across {df_manifest['subject_id'].nunique()} subjects.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = df_manifest.to_dict("records")

    print(f"Starting parallel feature extraction on 8 worker processes...")
    t0 = time.time()
    extracted_records = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_scan, r): r["image_id"] for r in rows}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                extracted_records.append(res)
            completed += 1
            if completed % 25 == 0 or completed == len(rows):
                elapsed = time.time() - t0
                pct = completed / len(rows) * 100
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{len(rows)} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} scans/sec")

    print(f"Extracted features for {len(extracted_records)} / {len(rows)} scans in {time.time() - t0:.1f}s.")

    df_features = pd.DataFrame(extracted_records)
    out_csv = OUTPUT_DIR / "multimodal_features.csv"
    df_features.to_csv(out_csv, index=False)
    print(f"Saved complete multimodal features table to: {out_csv}")
    print(f"Table dimensions: {df_features.shape[0]} rows x {df_features.shape[1]} columns")


if __name__ == "__main__":
    main()
