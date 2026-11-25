#!/usr/bin/env python3
"""
scripts/prepare_v3_dataset.py
-----------------------------
Ingests the newly organized 203-subject ADNI cohort from Downloads, performs
deterministic session-level deduplication, applies zero-leakage subject-stratified
splitting, and produces reports/data_audit/v3_clean_manifest.csv.
"""

from __future__ import annotations
import glob
import os
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Support multiple dataset source directories
_base_candidates = [
    Path("/home/karan/Downloads/NEW DATA SET "),
    Path("/home/karan/Downloads/NEW DATA SET"),
    Path("/home/karan/Downloads/new 2"),
]
BASE_DIRS = [d for d in _base_candidates if d.exists()]

MRI_DIRS = [d / "mri" for d in BASE_DIRS if (d / "mri").exists()]
XML_DIRS = [d / "xml" for d in BASE_DIRS if (d / "xml").exists()]
OUTPUT_MANIFEST = PROJECT_ROOT / "reports" / "data_audit" / "v3_clean_manifest.csv"


def parse_clinical_xml(xml_path: Path) -> dict:
    """Extracts diagnosis, subject ID, image ID, series ID, and clinical assessment values."""
    try:
        root = ET.fromstring(xml_path.read_bytes())
    except Exception as e:
        return {}

    rg = root.find(".//researchGroup")
    if rg is None or not rg.text:
        return {}

    raw_diag = rg.text.strip().upper()
    if "NORMAL" in raw_diag or "CN" in raw_diag or "CONTROL" in raw_diag:
        diag = "CN"
        label = 0
    elif "MCI" in raw_diag:
        diag = "MCI"
        label = 1
    elif "AD" in raw_diag or "ALZHEIMER" in raw_diag:
        diag = "AD"
        label = 2
    else:
        return {}

    subj = root.find(".//subjectIdentifier")
    img = root.find(".//imageUID")
    series = root.find(".//seriesIdentifier")
    date = root.find(".//studyDate")

    s_id = subj.text.strip() if subj is not None and subj.text else None
    i_id = img.text.strip() if img is not None and img.text else None
    ser_id = series.text.strip() if series is not None and series.text else None
    dt = date.text.strip()[:10] if date is not None and date.text else ""

    # Fallbacks from filename
    if not i_id:
        m = re.search(r"I(\d+)", xml_path.name)
        if m:
            i_id = m.group(1)
    if not s_id:
        m = re.search(r"(\d{3}_S_\d{4})", xml_path.name)
        if m:
            s_id = m.group(1)
    if not ser_id:
        m = re.search(r"S(\d+)", xml_path.name)
        if m:
            ser_id = m.group(1)

    # Demographics
    age_elem = root.find(".//subjectAge")
    age = float(age_elem.text.strip()) if age_elem is not None and age_elem.text else np.nan

    sex_elem = root.find(".//subjectSex")
    sex = 1.0 if (sex_elem is not None and sex_elem.text and sex_elem.text.strip().upper().startswith("M")) else 0.0

    weight_elem = root.find(".//weightKg")
    weight = float(weight_elem.text.strip()) if weight_elem is not None and weight_elem.text else np.nan

    # Genetics
    a1_elem = root.find(".//apoeA1")
    apoe_a1 = float(a1_elem.text.strip()) if a1_elem is not None and a1_elem.text and a1_elem.text.strip().isdigit() else np.nan

    a2_elem = root.find(".//apoeA2")
    apoe_a2 = float(a2_elem.text.strip()) if a2_elem is not None and a2_elem.text and a2_elem.text.strip().isdigit() else np.nan

    # Cognitive and functional assessments
    mmse = np.nan
    cdr = np.nan
    faq = np.nan
    npi = np.nan
    gds = np.nan

    for score in root.findall(".//assessmentScore"):
        attr = score.get("attribute", "").upper()
        val = score.text.strip() if score.text else ""
        try:
            val_f = float(val)
            if "MMSCORE" in attr:
                mmse = val_f
            elif "CDGLOBAL" in attr:
                cdr = val_f
            elif "FAQTOTAL" in attr:
                faq = val_f
            elif "NPISCORE" in attr:
                npi = val_f
            elif "GDTOTAL" in attr or "GDSCALE" in attr:
                gds = val_f
        except (ValueError, TypeError):
            continue

    return {
        "xml_path": str(xml_path),
        "subject_id": s_id,
        "image_id": i_id,
        "series_id": ser_id,
        "date_acquired": dt,
        "diagnosis": diag,
        "target_label": label,
        "clin_age": age,
        "clin_sex": sex,
        "clin_weight": weight,
        "clin_apoe_a1": apoe_a1,
        "clin_apoe_a2": apoe_a2,
        "clin_mmse": mmse,
        "clin_cdr": cdr,
        "clin_faq": faq,
        "clin_npi": npi,
        "clin_gds": gds,
    }


def priority_score(nifti_path: str) -> int:
    score = 0
    if "Scaled" in nifti_path:
        score += 10
    if "N3" in nifti_path:
        score += 5
    if "B1" in nifti_path:
        score += 3
    if "GradWarp" in nifti_path:
        score += 2
    return score


def main():
    print(f"Ingesting XMLs from: {[str(d) for d in XML_DIRS]}")
    xml_files = []
    for xd in XML_DIRS:
        xml_files.extend(list(xd.glob("**/*.xml")))
    print(f"Found {len(xml_files)} XML files.")

    xml_dict = {}
    for x in xml_files:
        meta = parse_clinical_xml(x)
        if meta and meta.get("image_id"):
            xml_dict[meta["image_id"]] = meta

    print(f"Successfully parsed metadata for {len(xml_dict)} unique Image IDs.")

    print(f"Scanning NIfTIs from: {[str(d) for d in MRI_DIRS]}")
    nii_files = []
    for md in MRI_DIRS:
        nii_files.extend(list(md.glob("**/*.nii*")))
    print(f"Found {len(nii_files)} NIfTI scans.")

    matched_scans = []
    for n in nii_files:
        # Quality check: discard any non-MRI masks
        if "Mask" in str(n) or "mask" in n.name:
            continue

        m = re.search(r"I(\d+)", n.name)
        if m and m.group(1) in xml_dict:
            meta = xml_dict[m.group(1)]
            s_match = re.search(r"(\d{3}_S_\d{4})", str(n))
            subj = s_match.group(1) if s_match else meta["subject_id"]
            d_match = re.search(r"(\d{4}-\d{2}-\d{2})", str(n))
            date = d_match.group(1) if d_match else meta["date_acquired"]

            row = {
                "nifti_path": str(n),
                "image_id": m.group(1),
                "subject_id": subj,
                "date_acquired": date,
                **{k: v for k, v in meta.items() if k not in ["date_acquired", "subject_id", "image_id"]},
            }
            matched_scans.append(row)

    print(f"Matched NIfTI scans: {len(matched_scans)}")

    # Deterministic priority deduplication per physical session (subject_id, date_acquired)
    session_groups = defaultdict(list)
    for s in matched_scans:
        session_groups[(s["subject_id"], s["date_acquired"])].append(s)

    dedup_scans = []
    for key, group in session_groups.items():
        best = sorted(group, key=lambda x: priority_score(x["nifti_path"]), reverse=True)[0]
        dedup_scans.append(best)

    print(f"Deduplicated scans (1 per physical session): {len(dedup_scans)}")

    df = pd.DataFrame(dedup_scans)
    unique_subjects = df["subject_id"].unique()
    print(f"Total Unique Subjects: {len(unique_subjects)}")

    # Subject-level stratification (Train: 70%, Val: 15%, Test: 15%)
    # Group subjects by their single consensus diagnosis
    subj_diag = df.groupby("subject_id")["diagnosis"].agg(lambda s: s.mode()[0]).to_dict()

    np.random.seed(42)
    diag_to_subjs = defaultdict(list)
    for subj, diag in subj_diag.items():
        diag_to_subjs[diag].append(subj)

    train_subjs = set()
    val_subjs = set()
    test_subjs = set()

    for diag, subjs in sorted(diag_to_subjs.items()):
        subjs_shuffled = np.random.permutation(sorted(subjs))
        n = len(subjs_shuffled)
        n_train = int(np.round(n * 0.70))
        n_val = int(np.round(n * 0.15))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        tr = subjs_shuffled[:n_train]
        va = subjs_shuffled[n_train:n_train + n_val]
        te = subjs_shuffled[n_train + n_val:]

        train_subjs.update(tr)
        val_subjs.update(va)
        test_subjs.update(te)

    # Assign split column
    def assign_split(s_id):
        if s_id in train_subjs:
            return "train"
        elif s_id in val_subjs:
            return "validation"
        else:
            return "test"

    df["split"] = df["subject_id"].apply(assign_split)

    # Invariant assertions
    assert len(train_subjs.intersection(val_subjs)) == 0, "Leakage between train and val!"
    assert len(train_subjs.intersection(test_subjs)) == 0, "Leakage between train and test!"
    assert len(val_subjs.intersection(test_subjs)) == 0, "Leakage between val and test!"

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"\nSaved clean manifest to: {OUTPUT_MANIFEST}")

    # Summary table
    print("\n=== SPLIT SUMMARY ===")
    for sp in ["train", "validation", "test"]:
        sub_df = df[df["split"] == sp]
        sub_subjs = sub_df["subject_id"].nunique()
        diag_counts = sub_df["diagnosis"].value_counts().to_dict()
        print(f"{sp.capitalize():<12}: {len(sub_df):>3} scans | {sub_subjs:>3} subjects | Diagnoses: {diag_counts}")


if __name__ == "__main__":
    main()
