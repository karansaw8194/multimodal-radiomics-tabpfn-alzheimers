from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DIAGNOSIS_LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}

REVERSE_DIAGNOSIS_MAP = {v: k for k, v in DIAGNOSIS_LABEL_MAP.items()}

@dataclass
class ClinicalConfig:
    age_col: str = "age"
    sex_col: str = "sex"
    apoe_a1_col: str = "apoe_a1"
    apoe_a2_col: str = "apoe_a2"
    mmse_col: str = "mmse"
    gds_col: str = "gds"
    cdr_col: str = "cdr"
    weight_col: str = "weight"

def parse_numeric(val: Any) -> float:
    """Safely convert string/object to float or np.nan."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def parse_sex_binary(val: Any) -> float:
    """Encode Sex: M -> 0.0, F -> 1.0, else np.nan."""
    if pd.isna(val) or val is None:
        return np.nan

    s = str(val).strip().upper()
    if s in ["M", "MALE", "0", "0.0"]:
        return 0.0
    elif s in ["F", "FEMALE", "1", "1.0"]:
        return 1.0
    return np.nan


def attach_acquisition_metadata(
    manifest_df: pd.DataFrame,
    preprocessing_catalog_path: Path,
) -> pd.DataFrame:
    """Attach acquisition IDs and splits through the canonical raw NIfTI path.

    The clinical manifest and preprocessing catalog describe the same acquisitions,
    but only the catalog owns ``acquisition_group_id``.  This deliberately uses a
    one-to-one path join rather than subject ID or input row position, since a
    subject can have multiple acquisitions.
    """
    if not preprocessing_catalog_path.exists():
        raise FileNotFoundError(
            f"Preprocessing catalog missing: {preprocessing_catalog_path}"
        )

    required_manifest = {"nifti_path", "subject_id", "diagnosis_xml"}
    required_catalog = {
        "raw_nifti_path",
        "acquisition_group_id",
        "split",
        "subject_id",
        "diagnosis_xml",
    }
    missing_manifest = required_manifest.difference(manifest_df.columns)
    if missing_manifest:
        raise ValueError(
            "Manifest cannot be aligned to acquisitions; missing columns: "
            f"{sorted(missing_manifest)}"
        )

    catalog_df = pd.read_csv(preprocessing_catalog_path)
    missing_catalog = required_catalog.difference(catalog_df.columns)
    if missing_catalog:
        raise ValueError(
            "Preprocessing catalog cannot provide acquisition metadata; missing "
            f"columns: {sorted(missing_catalog)}"
        )
    if manifest_df["nifti_path"].isna().any() or manifest_df["nifti_path"].duplicated().any():
        raise ValueError("Manifest NIfTI paths must be present and unique for alignment.")
    if catalog_df["raw_nifti_path"].isna().any() or catalog_df["raw_nifti_path"].duplicated().any():
        raise ValueError("Catalog raw NIfTI paths must be present and unique for alignment.")

    catalog_metadata = catalog_df[
        [
            "raw_nifti_path",
            "acquisition_group_id",
            "split",
            "subject_id",
            "diagnosis_xml",
        ]
    ].rename(
        columns={
            "subject_id": "catalog_subject_id",
            "diagnosis_xml": "catalog_diagnosis_xml",
        }
    )
    aligned = manifest_df.merge(
        catalog_metadata,
        how="left",
        left_on="nifti_path",
        right_on="raw_nifti_path",
        validate="one_to_one",
    )
    if aligned["acquisition_group_id"].isna().any():
        count = int(aligned["acquisition_group_id"].isna().sum())
        raise ValueError(
            f"Could not map {count} manifest acquisition(s) to the preprocessing catalog."
        )

    subject_mismatch = aligned["subject_id"].astype(str).str.strip() != aligned[
        "catalog_subject_id"
    ].astype(str).str.strip()
    diagnosis_mismatch = aligned["diagnosis_xml"].astype(str).str.strip().str.upper() != aligned[
        "catalog_diagnosis_xml"
    ].astype(str).str.strip().str.upper()
    if subject_mismatch.any() or diagnosis_mismatch.any():
        raise ValueError(
            "Catalog metadata disagrees with the clinical manifest for one or more "
            "canonical raw NIfTI paths."
        )
    return aligned.drop(
        columns=["raw_nifti_path", "catalog_subject_id", "catalog_diagnosis_xml"]
    )


def process_clinical_manifest(
    manifest_path: Path,
    output_dir: Path,
    config: Optional[ClinicalConfig] = None,
    preprocessing_catalog_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract and standardize clinical features from the selected manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    if config is None:
        config = ClinicalConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_path)
    if "acquisition_group_id" not in df.columns:
        if preprocessing_catalog_path is None:
            raise ValueError(
                "Manifest has no acquisition_group_id. Provide preprocessing_catalog_path "
                "to align it through canonical raw NIfTI paths."
            )
        df = attach_acquisition_metadata(df, preprocessing_catalog_path)

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        subject_id = str(row["subject_id"]).strip()
        group_id = str(row["acquisition_group_id"]).strip()
        split = str(row.get("split", "unassigned")).strip()
        dx_str = str(row.get("diagnosis_xml", "")).strip().upper()

        if dx_str not in DIAGNOSIS_LABEL_MAP:
            continue

        target_label = DIAGNOSIS_LABEL_MAP[dx_str]

        # Standardize Clinical Features
        age_val = parse_numeric(row.get(config.age_col))
        sex_val = parse_sex_binary(row.get(config.sex_col))
        apoe1_val = parse_numeric(row.get(config.apoe_a1_col))
        apoe2_val = parse_numeric(row.get(config.apoe_a2_col))
        mmse_val = parse_numeric(row.get(config.mmse_col))
        gds_val = parse_numeric(row.get(config.gds_col))
        cdr_val = parse_numeric(row.get(config.cdr_col))
        weight_val = parse_numeric(row.get(config.weight_col))

        record = {
            "subject_id": subject_id,
            "acquisition_group_id": group_id,
            "split": split,
            "diagnosis_xml": dx_str,
            "target_label": target_label,
            "clin_age": age_val,
            "clin_sex": sex_val,
            "clin_apoe_a1": apoe1_val,
            "clin_apoe_a2": apoe2_val,
            "clin_mmse": mmse_val,
            "clin_gds": gds_val,
            "clin_cdr": cdr_val,
            "clin_weight": weight_val,
        }
        records.append(record)

    clinical_df = pd.DataFrame(records)

    # Save CSV
    out_csv = output_dir / "clinical_features.csv"
    clinical_df.to_csv(out_csv, index=False)

    # Summary
    feature_cols = [c for c in clinical_df.columns if c.startswith("clin_")]
    summary = {
        "total_clinical_records": len(clinical_df),
        "unique_subjects": int(clinical_df["subject_id"].nunique()),
        "clinical_feature_count": len(feature_cols),
        "target_distribution": {
            dx: int((clinical_df["diagnosis_xml"] == dx).sum())
            for dx in ["CN", "MCI", "AD"]
        },
        "missing_value_summary": {
            col: int(clinical_df[col].isnull().sum()) for col in feature_cols
        },
        "output_csv_path": str(out_csv.resolve()),
    }

    with (output_dir / "clinical_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return clinical_df

def merge_multimodal_features(
    clinical_csv_path: Path,
    radiomics_csv_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge clinical and radiomics features on (subject_id, acquisition_group_id)."""
    if not clinical_csv_path.exists():
        raise FileNotFoundError(f"Clinical CSV missing: {clinical_csv_path}")

    if not radiomics_csv_path.exists():
        raise FileNotFoundError(f"Radiomics CSV missing: {radiomics_csv_path}")

    clin_df = pd.read_csv(clinical_csv_path)
    rad_df = pd.read_csv(radiomics_csv_path)

    # Identify metadata columns to avoid duplication during merge
    metadata_cols = ["subject_id", "acquisition_group_id", "split", "diagnosis_xml"]
    for name, dataframe in (("Clinical", clin_df), ("Radiomics", rad_df)):
        missing = set(metadata_cols).difference(dataframe.columns)
        if missing:
            raise ValueError(f"{name} table missing merge columns: {sorted(missing)}")
        if dataframe[metadata_cols].isna().any().any():
            raise ValueError(f"{name} table has missing multimodal merge keys.")
        if dataframe.duplicated(metadata_cols).any():
            raise ValueError(f"{name} table has duplicate multimodal merge keys.")

    # Select feature columns from radiomics
    rad_feature_cols = [c for c in rad_df.columns if c not in metadata_cols]

    # Merge on keys
    merged_df = pd.merge(
        clin_df,
        rad_df[metadata_cols + rad_feature_cols],
        on=metadata_cols,
        how="inner",
        validate="one_to_one",
    )

    if len(merged_df) != len(clin_df) or len(merged_df) != len(rad_df):
        raise ValueError(
            "Clinical and radiomics acquisitions do not match exactly; refusing to "
            "write a partial multimodal table."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    return merged_df
