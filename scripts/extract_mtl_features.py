#!/usr/bin/env python3
"""
scripts/extract_mtl_features.py
-------------------------------
Extracts focal Medial Temporal Lobe (MTL) / Hippocampal ROI features and
Hippocampal-to-Ventricle Ratio (HVR) across all deduplicated scans in parallel.
"""

from __future__ import annotations
import concurrent.futures
import time
import tempfile
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adni_pipeline.preprocessing import preprocess_single_nifti
from adni_pipeline.segmentation import OtsuMorphologicalBackend

MANIFEST_PATH = PROJECT_ROOT / "reports" / "data_audit" / "v3_clean_manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "clean_v3" / "mtl_features.csv"


def extract_mtl_for_scan(row_dict: dict) -> dict:
    image_id = str(row_dict["image_id"])
    nifti_path = Path(row_dict["nifti_path"])

    if not nifti_path.exists():
        return {"image_id": image_id, "mtl_hvr": np.nan, "mtl_ratio_to_brain": np.nan, "mtl_ventricle_vol": np.nan, "mtl_parenchyma_vol": np.nan}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_prep = Path(tmp_dir) / "prep.nii.gz"
        try:
            out_img, meta = preprocess_single_nifti(
                nifti_path=nifti_path,
                output_path=tmp_prep,
                target_spacing=(1.0, 1.0, 1.0),
            )
            data = out_img.get_fdata(dtype=np.float32)
            backend = OtsuMorphologicalBackend()
            mask = backend.segment(data)

            brain_pts = np.argwhere(mask > 0)
            if len(brain_pts) == 0:
                return {"image_id": image_id, "mtl_hvr": np.nan, "mtl_ratio_to_brain": np.nan, "mtl_ventricle_vol": np.nan, "mtl_parenchyma_vol": np.nan}

            c = (brain_pts.min(axis=0) + brain_pts.max(axis=0)) / 2.0

            # Medial temporal lobe bilateral bounding box (in 1.0mm RAS+ coordinates)
            x_min_l, x_max_l = int(c[0] - 38), int(c[0] - 16)
            x_min_r, x_max_r = int(c[0] + 16), int(c[0] + 38)
            y_min, y_max = int(c[1] - 32), int(c[1] + 8)
            z_min, z_max = int(c[2] - 35), int(c[2] - 5)

            mtl_mask = np.zeros_like(mask, dtype=bool)
            mtl_mask[x_min_l:x_max_l, y_min:y_max, z_min:z_max] = True
            mtl_mask[x_min_r:x_max_r, y_min:y_max, z_min:z_max] = True

            mtl_par = float(np.sum((mask > 0) & (mask != 3) & mtl_mask))
            mtl_vent = float(np.sum((mask == 3) & mtl_mask))
            tot_brain = float(np.sum(mask > 0))
            tot_vent = float(np.sum(mask == 3))

            hvr = float(mtl_par / max(1.0, tot_vent))
            mtl_ratio = float(mtl_par / max(1.0, tot_brain))

            return {
                "image_id": image_id,
                "mtl_hvr": hvr,
                "mtl_ratio_to_brain": mtl_ratio,
                "mtl_ventricle_vol": mtl_vent,
                "mtl_parenchyma_vol": mtl_par,
            }
        except Exception as e:
            return {"image_id": image_id, "mtl_hvr": np.nan, "mtl_ratio_to_brain": np.nan, "mtl_ventricle_vol": np.nan, "mtl_parenchyma_vol": np.nan}


def main():
    df_manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df_manifest)} scans.")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = df_manifest.to_dict("records")
    print("Starting parallel MTL feature extraction on 8 workers...")
    t0 = time.time()
    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_mtl_for_scan, r): r["image_id"] for r in rows}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(rows):
                elapsed = time.time() - t0
                pct = completed / len(rows) * 100
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{len(rows)} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} scans/s")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved MTL features to {OUTPUT_CSV} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
