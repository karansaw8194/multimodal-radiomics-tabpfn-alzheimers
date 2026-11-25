"""Extract 3D Vision Mamba (Med-Mamba) Embeddings and Merge with Enhanced Clinical Features.

Replaces coarse Otsu morphological segmentation and generic radiomics with
frozen 3D Vision Mamba spatial embeddings (24 dimensions) per MRI scan.
Merges with granular clinical features (CDR-SB, Memory, Orientation, Judgment,
APOE4 carrier/dose, MMSE, FAQ, Age, Sex, Weight) and engineered non-linear terms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adni_pipeline.clinical_parser_enhanced import parse_enhanced_clinical_xml
from adni_pipeline.vision_mamba_3d import extract_mamba_features, get_frozen_mamba_model

DATA_DIR = PROJECT_ROOT / "data" / "clean_v3"
INPUT_CSV = DATA_DIR / "multimodal_features.csv"
CACHE_CSV = DATA_DIR / "mamba_embeddings_cache.csv"
OUTPUT_CSV = DATA_DIR / "mamba_multimodal_dataset.csv"


def extract_or_load_mamba_cache(
    df: pd.DataFrame,
    cache_path: Path,
    dry_run: bool = False,
    sample_size: int = 10,
    embedding_dim: int = 24,
) -> pd.DataFrame:
    """Extract or load cached 3D Vision Mamba embeddings for all scans."""
    if dry_run:
        df = df.head(sample_size).copy()
        print(f"[Smoke Test] Running dry-run on {len(df)} sample scans...")

    # Load existing cache if available
    cached_df = pd.DataFrame()
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path)
            print(f"Loaded existing Mamba embeddings cache with {len(cached_df)} records.")
        except Exception as e:
            print(f"Failed to read cache {cache_path}: {e}")

    cached_ids = set()
    if not cached_df.empty and "image_id" in cached_df.columns:
        cached_ids = set(cached_df["image_id"].astype(str))

    model = get_frozen_mamba_model(embedding_dim=embedding_dim, device="cpu")
    feat_cols = [f"mamba_feat_{i:02d}" for i in range(embedding_dim)]

    new_rows = []
    total = len(df)
    t_start = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        img_id = str(row["image_id"])
        nii_path = Path(str(row["nifti_path"]))

        if img_id in cached_ids:
            continue

        if not nii_path.exists():
            print(f"[Warning] NIfTI not found: {nii_path}")
            continue

        t0 = time.time()
        try:
            feats = extract_mamba_features(
                nii_path,
                embedding_dim=embedding_dim,
                device="cpu",
                model=model,
            )
            feats["image_id"] = img_id
            feats["subject_id"] = str(row["subject_id"])
            feats["nifti_path"] = str(nii_path)
            new_rows.append(feats)
        except Exception as e:
            print(f"[Error] Failed to extract from {nii_path}: {e}")
            continue

        if (len(new_rows) + len(cached_ids)) % 25 == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            done = len(cached_ids) + len(new_rows)
            rate = done / (elapsed + 1e-6)
            print(f"[{done}/{total}] Mamba Embeddings Extracted ({rate:.1f} scans/sec)")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not cached_df.empty:
            full_cache = pd.concat([cached_df, new_df], ignore_index=True)
            full_cache = full_cache.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
        else:
            full_cache = new_df
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full_cache.to_csv(cache_path, index=False)
        print(f"Saved updated Mamba embeddings cache ({len(full_cache)} scans) to {cache_path}")
        return full_cache
    else:
        return cached_df


def build_mamba_multimodal_dataset(dry_run: bool = False, sample_size: int = 10) -> pd.DataFrame:
    """Build complete multimodal dataset combining 3D Vision Mamba embeddings and clinical features."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Source features CSV missing: {INPUT_CSV}")

    df_base = pd.read_csv(INPUT_CSV)
    if dry_run:
        df_base = df_base.head(sample_size).copy()

    print(f"Processing {len(df_base)} scans for Mamba Multimodal Dataset...")

    # 1. Extract / retrieve 3D Vision Mamba embeddings
    mamba_cache = extract_or_load_mamba_cache(
        df_base,
        CACHE_CSV,
        dry_run=dry_run,
        sample_size=sample_size,
        embedding_dim=24,
    )
    mamba_cache["image_id"] = mamba_cache["image_id"].astype(str)
    df_base["image_id"] = df_base["image_id"].astype(str)

    mamba_feat_cols = [f"mamba_feat_{i:02d}" for i in range(24)]
    mamba_sub = mamba_cache[["image_id"] + [c for c in mamba_feat_cols if c in mamba_cache.columns]].copy()

    # 2. Parse granular clinical features from XMLs
    print("Parsing granular clinical dementia staging metrics from XMLs...")
    clin_rows = []
    for _, row in df_base.iterrows():
        xml_p = Path(str(row["xml_path"]))
        parsed = parse_enhanced_clinical_xml(xml_p)
        parsed["image_id"] = str(row["image_id"])
        clin_rows.append(parsed)

    df_clin = pd.DataFrame(clin_rows)

    # 3. Merge base metadata with enhanced clinical features
    meta_cols = ["nifti_path", "image_id", "subject_id", "date_acquired", "xml_path", "series_id", "diagnosis", "target_label", "split"]
    # Retain longitudinal rate of change features from df_base if present
    longitudinal_cols = [
        "time_from_baseline_years",
        "delta_mmse_annualized",
        "delta_ventricle_annualized",
        "delta_cdr_annualized",
    ]
    avail_long = [c for c in longitudinal_cols if c in df_base.columns]

    df_merged = df_base[meta_cols + avail_long].copy()

    # Drop duplicate image_id in clinical and merge
    clin_features_only = [
        c for c in df_clin.columns
        if c not in ["subject_id", "series_id", "date_acquired", "diagnosis", "target_label"]
    ]
    df_merged = df_merged.merge(df_clin[clin_features_only], on="image_id", how="left")

    # 4. Merge with 3D Vision Mamba deep embeddings via 1:1 image_id match
    df_merged = df_merged.merge(mamba_sub, on="image_id", how="inner")

    # 5. Compute cross-modal interaction terms
    # Non-linear interaction between Mamba deep spatial representation and cognitive decline
    mamba_spatial_norm = np.sqrt(np.sum(df_merged[mamba_feat_cols].values ** 2, axis=1) + 1e-6)
    df_merged["mamba_norm"] = mamba_spatial_norm
    df_merged["feature_mamba_cdr_product"] = df_merged["mamba_norm"] * (df_merged["clin_cdr"] + 0.1)
    df_merged["feature_mamba_cognitive_interaction"] = (
        df_merged["mamba_norm"] * df_merged["feature_cognitive_deficit"]
    )
    df_merged["feature_hacd"] = (
        df_merged["clin_cdrsb"] / (df_merged["clin_mmse"].clip(0, 30) + 1.0)
    )
    df_merged["feature_mci_retention_index"] = (
        (df_merged["clin_mmse"] / (df_merged["clin_cdrsb"] + 0.5))
        * (1.0 / (df_merged["mamba_norm"] + 1e-4))
    )

    print(f"Dataset assembled: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns.")
    if not dry_run:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved full Mamba multimodal dataset to {OUTPUT_CSV}")

    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D Vision Mamba embeddings and build multimodal dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry-run smoke test on a small subset")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size for dry-run smoke test")
    args = parser.parse_args()

    df = build_mamba_multimodal_dataset(dry_run=args.dry_run, sample_size=args.sample_size)
    print("Execution complete. Sample data preview:")
    cols_preview = ["subject_id", "diagnosis", "clin_cdr", "clin_cdrsb", "clin_apoe4_carrier", "mamba_feat_00", "mamba_feat_01", "mamba_norm"]
    avail_preview = [c for c in cols_preview if c in df.columns]
    print(df[avail_preview].head(5))
