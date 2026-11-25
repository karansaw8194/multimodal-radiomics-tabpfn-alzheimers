from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.feature_preprocessing import run_feature_preprocessing


def prepare_v2_clean_dataset(project_root: Path = PROJECT_ROOT) -> dict:
    manifest_path = project_root / "reports" / "data_audit" / "new_dataset_fixed" / "adni_mri_xml_manifest_valid.csv"
    features_path = project_root / "data" / "new_dataset_derived" / "multimodal_v1" / "multimodal_features.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(features_path)

    if len(manifest) != len(features):
        raise ValueError(f"Manifest length ({len(manifest)}) != Features length ({len(features)})")

    # Combine for filtering
    combined = features.copy()
    combined["nifti_path"] = manifest["nifti_path"]
    combined["xml_path"] = manifest["xml_path"]
    combined["processing_version"] = manifest["processing_version"]
    combined["date_acquired"] = manifest["date_acquired"]
    combined["image_id_xml"] = manifest["image_id_xml"]
    combined["series_id"] = manifest["series_id"]
    combined["visit"] = manifest["visit"]
    combined["age"] = manifest["age"]
    combined["sex"] = manifest["sex"]

    # 1. Exclude masks
    is_mask = combined["processing_version"].str.contains("Mask", case=False, na=False)
    mask_count = int(is_mask.sum())
    non_mask = combined[~is_mask].copy()

    # 2. Priority ranking for processing descriptions
    priority_map = {
        "MPR__GradWarp__B1_Correction__N3__Scaled": 10,
        "MPR__GradWarp__B1_Correction__N3__Scaled_2": 10,
        "MT1__GradWarp__N3m": 10,
        "MPR__GradWarp__B1_Correction__N3": 8,
        "MT1__N3m": 8,
        "MPR__GradWarp__B1_Correction": 6,
        "MPR__GradWarp": 4,
    }
    non_mask["priority"] = non_mask["processing_version"].map(priority_map).fillna(0)

    # Sort deterministically
    sorted_df = non_mask.sort_values(
        by=["subject_id", "date_acquired", "priority", "image_id_xml"],
        ascending=[True, True, False, False],
    )

    # Deduplicate to exactly 1 best scan per session
    dedup = sorted_df.drop_duplicates(
        subset=["subject_id", "date_acquired"],
        keep="first",
    ).copy()

    # Sort cleanly by split and subject
    split_order = {"train": 0, "validation": 1, "test": 2}
    dedup["split_rank"] = dedup["split"].map(split_order)
    dedup = dedup.sort_values(by=["split_rank", "subject_id", "date_acquired"]).reset_index(drop=True)
    dedup.drop(columns=["split_rank", "priority"], inplace=True)

    # Validate zero subject leakage
    train_subjs = set(dedup.loc[dedup["split"] == "train", "subject_id"])
    val_subjs = set(dedup.loc[dedup["split"] == "validation", "subject_id"])
    test_subjs = set(dedup.loc[dedup["split"] == "test", "subject_id"])

    assert not (train_subjs & val_subjs), "Train/Val leakage!"
    assert not (train_subjs & test_subjs), "Train/Test leakage!"
    assert not (val_subjs & test_subjs), "Val/Test leakage!"

    # Create output directories
    clean_data_dir = project_root / "data" / "clean_v2"
    clean_data_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = project_root / "reports" / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Save clean manifest
    manifest_cols = [
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "target_label",
        "nifti_path",
        "xml_path",
        "processing_version",
        "date_acquired",
        "image_id_xml",
        "series_id",
        "visit",
        "age",
        "sex",
    ]
    clean_manifest_df = dedup[manifest_cols].copy()
    clean_manifest_path = audit_dir / "v2_clean_manifest.csv"
    clean_manifest_df.to_csv(clean_manifest_path, index=False)

    # Extract feature column subsets
    meta_cols = ["subject_id", "acquisition_group_id", "split", "diagnosis_xml", "target_label"]
    clin_cols = [c for c in dedup.columns if c.startswith("clin_")]
    rad_cols = [c for c in dedup.columns if c.startswith(("brain_", "subcortical_"))]
    multi_cols = clin_cols + rad_cols

    # Save feature tables
    multimodal_table = dedup[meta_cols + multi_cols].copy()
    clinical_table = dedup[meta_cols + clin_cols].copy()
    radiomics_table = dedup[meta_cols + rad_cols].copy()

    multimodal_csv = clean_data_dir / "multimodal_features.csv"
    clinical_csv = clean_data_dir / "clinical_features.csv"
    radiomics_csv = clean_data_dir / "radiomics_features.csv"

    multimodal_table.to_csv(multimodal_csv, index=False)
    clinical_table.to_csv(clinical_csv, index=False)
    radiomics_table.to_csv(radiomics_csv, index=False)

    # Run train-only imputation and scaling into model_inputs
    model_inputs_dir = clean_data_dir / "model_inputs"
    preprocessing_summary = run_feature_preprocessing(
        multimodal_path=multimodal_csv,
        output_root=model_inputs_dir,
    )

    stats = {
        "raw_total_records": len(manifest),
        "masks_excluded": mask_count,
        "duplicate_processing_excluded": len(non_mask) - len(dedup),
        "clean_total_records": len(dedup),
        "unique_subjects": int(dedup["subject_id"].nunique()),
        "splits": {
            s: {
                "scans": int(len(dedup[dedup["split"] == s])),
                "subjects": int(dedup.loc[dedup["split"] == s, "subject_id"].nunique()),
                "class_scans": dedup.loc[dedup["split"] == s, "diagnosis_xml"].value_counts().to_dict(),
                "class_subjects": dedup.loc[dedup["split"] == s].groupby("subject_id")["diagnosis_xml"].first().value_counts().to_dict(),
            }
            for s in ["train", "validation", "test"]
        },
        "preprocessing_summary": preprocessing_summary,
    }

    with (clean_data_dir / "v2_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("==================================================")
    print("      V2 CLEAN DATASET PREPARATION COMPLETE")
    print("==================================================")
    print(f"Total Raw Rows:        {len(manifest)}")
    print(f"Masks Excluded:        {mask_count}")
    print(f"Deduplicated Sessions: {len(dedup)} across {dedup['subject_id'].nunique()} subjects")
    for s, data in stats["splits"].items():
        print(f"  [{s.upper()}]: {data['scans']} scans | {data['subjects']} subjects | Subj Classes: {data['class_subjects']}")
    print(f"Clean Manifest:        {clean_manifest_path}")
    print(f"Multimodal Features:   {multimodal_csv}")
    print(f"Model Inputs Root:     {model_inputs_dir}")

    return stats


if __name__ == "__main__":
    prepare_v2_clean_dataset()
