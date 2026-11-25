#!/usr/bin/env python3
"""
scripts/build_v3_multimodal_dataset.py
--------------------------------------
Unifies:
1. 44 3D Radiomics features (from data/clean_v3/feature_cache/*.json)
2. Granular Clinical XML assessments (clin_faq, clin_npi, clin_gds, clin_mmse, clin_cdr, clin_age, clin_sex, clin_weight)
3. Focal Medial Temporal Lobe (MTL) / Hippocampal ROI features & HVR (from data/clean_v3/mtl_features.csv)
4. Longitudinal annualized rate-of-change delta features (delta_mmse_annualized, delta_faq_annualized, delta_ventricle_annualized, delta_cdr_annualized)
5. Cross-modal biological interaction ratios

Outputs:
  data/clean_v3/multimodal_features.csv (557 scans across 256 unique subjects)
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reports" / "data_audit" / "v3_clean_manifest.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "clean_v3" / "feature_cache"
MTL_CSV = PROJECT_ROOT / "data" / "clean_v3" / "mtl_features.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "clean_v3" / "multimodal_features.csv"


def build_dataset():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest loaded: {len(df_manifest)} scans across {df_manifest['subject_id'].nunique()} subjects.")

    # 1. Load radiomics from feature_cache
    radiomics_records = []
    for img_id in df_manifest["image_id"]:
        c_path = CACHE_DIR / f"{img_id}.json"
        if c_path.exists():
            try:
                with open(c_path, "r") as f:
                    radiomics_records.append(json.load(f))
            except Exception:
                pass

    print(f"Loaded {len(radiomics_records)} radiomics cache records.")
    df_rad = pd.DataFrame(radiomics_records)
    # Ensure image_id is string
    df_rad["image_id"] = df_rad["image_id"].astype(str)
    df_manifest["image_id"] = df_manifest["image_id"].astype(str)

    # 2. Merge Manifest metadata & clinical fields
    # Exclude metadata columns from df_rad that are refreshed from manifest
    cols_to_drop_from_rad = [
        c for c in ["diagnosis", "target_label", "split", "clin_age", "clin_sex", 
                    "clin_weight", "clin_apoe_a1", "clin_apoe_a2", "clin_mmse", 
                    "clin_cdr", "clin_faq", "clin_npi", "clin_gds", "date_acquired", "subject_id"]
        if c in df_rad.columns
    ]
    df_rad = df_rad.drop(columns=cols_to_drop_from_rad)

    df_merged = pd.merge(df_manifest, df_rad, on="image_id", how="inner")
    print(f"Merged manifest and radiomics: {df_merged.shape}")

    # 3. Merge MTL / Hippocampal features if available
    if MTL_CSV.exists():
        df_mtl = pd.read_csv(MTL_CSV)
        df_mtl["image_id"] = df_mtl["image_id"].astype(str)
        df_merged = pd.merge(df_merged, df_mtl, on="image_id", how="left")
        print(f"Merged MTL features: {df_merged.shape}")
    else:
        print("MTL features CSV not yet found; skipping MTL merge for now.")

    # 4. Longitudinal Rate-of-Change / Delta Features
    df_merged["date_dt"] = pd.to_datetime(df_merged["date_acquired"], errors="coerce")
    df_merged = df_merged.sort_values(["subject_id", "date_dt"]).reset_index(drop=True)

    delta_mmse = []
    delta_faq = []
    delta_vent = []
    delta_cdr = []
    time_from_base = []

    for subj_id, grp in df_merged.groupby("subject_id", sort=False):
        base_row = grp.iloc[0]
        base_t = base_row["date_dt"]
        base_mmse = base_row.get("clin_mmse")
        base_faq = base_row.get("clin_faq")
        base_vent = base_row.get("ratio_ventricle_to_brain")
        base_cdr = base_row.get("clin_cdr")

        for _, row in grp.iterrows():
            if pd.notnull(row["date_dt"]) and pd.notnull(base_t) and (row["date_dt"] > base_t):
                dt_years = (row["date_dt"] - base_t).total_seconds() / (365.25 * 86400)
            else:
                dt_years = 0.0

            time_from_base.append(dt_years)

            if dt_years >= 0.1:  # At least ~36 days between visits
                d_m = (row["clin_mmse"] - base_mmse) / dt_years if pd.notnull(row.get("clin_mmse")) and pd.notnull(base_mmse) else 0.0
                d_f = (row["clin_faq"] - base_faq) / dt_years if pd.notnull(row.get("clin_faq")) and pd.notnull(base_faq) else 0.0
                d_v = (row["ratio_ventricle_to_brain"] - base_vent) / dt_years if pd.notnull(row.get("ratio_ventricle_to_brain")) and pd.notnull(base_vent) else 0.0
                d_c = (row["clin_cdr"] - base_cdr) / dt_years if pd.notnull(row.get("clin_cdr")) and pd.notnull(base_cdr) else 0.0
            else:
                d_m, d_f, d_v, d_c = 0.0, 0.0, 0.0, 0.0

            delta_mmse.append(d_m)
            delta_faq.append(d_f)
            delta_vent.append(d_v)
            delta_cdr.append(d_c)

    df_merged["time_from_baseline_years"] = time_from_base
    df_merged["delta_mmse_annualized"] = delta_mmse
    df_merged["delta_faq_annualized"] = delta_faq
    df_merged["delta_ventricle_annualized"] = delta_vent
    df_merged["delta_cdr_annualized"] = delta_cdr

    # 5. Cross-Modal Biological Interaction Ratios
    df_merged["ratio_ventricle_to_subcortical"] = df_merged["ratio_ventricle_to_brain"] / (df_merged["ratio_subcortical_to_brain"] + 1e-6)
    df_merged["clin_mmse_age_ratio"] = df_merged["clin_mmse"] / (df_merged["clin_age"] + 1e-6)
    df_merged["clin_cdr_mmse_ratio"] = df_merged["clin_cdr"] / (df_merged["clin_mmse"] + 1.0)
    df_merged["biomarker_atrophy_cognition_index"] = df_merged["ratio_ventricle_to_brain"] * (df_merged["clin_cdr"] + 0.1) / (df_merged["clin_mmse"] + 1.0)

    # Added FAQ & NPI ratios
    df_merged["ratio_faq_to_mmse"] = (df_merged["clin_faq"] + 0.1) / (df_merged["clin_mmse"] + 1.0)
    df_merged["clin_faq_cdr_product"] = df_merged["clin_faq"] * (df_merged["clin_cdr"] + 0.05)

    if "mtl_hvr" in df_merged.columns:
        df_merged["ratio_hvr_to_cdr"] = df_merged["mtl_hvr"] / (df_merged["clin_cdr"] + 0.1)
        df_merged["ratio_hvr_to_faq"] = df_merged["mtl_hvr"] / (df_merged["clin_faq"] + 1.0)

    # 6. Asymmetric Non-Linear Interaction Features (User Action 3)
    # Hippocampal Atrophy to Cognitive Decline Interaction (HACD)
    ratio_v_s = df_merged["ratio_ventricle_to_brain"] / (df_merged["ratio_subcortical_to_brain"] + 1e-6)
    cog_deficit = (30.0 - df_merged["clin_mmse"].clip(0, 30)) / (df_merged["clin_mmse"].clip(0, 30) + 1.0)
    df_merged["feature_hacd"] = ratio_v_s * cog_deficit

    # CDR-Squared Severity Multiplier
    df_merged["feature_cdr_severity"] = (df_merged["clin_cdr"] ** 2) * df_merged["clin_age"]

    # Drop temporary datetime column
    df_merged = df_merged.drop(columns=["date_dt"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved enriched multimodal features table to: {OUTPUT_CSV}")
    print(f"Table dimensions: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")


if __name__ == "__main__":
    build_dataset()
