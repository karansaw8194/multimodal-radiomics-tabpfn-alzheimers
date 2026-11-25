from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier

from adni_pipeline.single_pair_pipeline import run_live_single_pair_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v2_artifact_package_integrity() -> None:
    pkg_dir = PROJECT_ROOT / "artifacts" / "v2_gwo_tabpfn"
    assert pkg_dir.is_dir(), f"Artifact directory missing: {pkg_dir}"

    required_files = [
        "model.joblib",
        "imputer.joblib",
        "scaler.joblib",
        "selected_features.json",
        "feature_names.json",
        "manifest.csv",
        "class_mapping.json",
        "pipeline_config.json",
        "metadata.json",
    ]

    for fname in required_files:
        fpath = pkg_dir / fname
        assert fpath.is_file(), f"Missing required artifact: {fname}"

    # Verify model is TabPFNClassifier, NOT LogisticRegression
    model = joblib.load(pkg_dir / "model.joblib")
    assert isinstance(model, TabPFNClassifier), f"Model is not TabPFNClassifier: {type(model)}"
    assert not isinstance(model, LogisticRegression), "Model must not be LogisticRegression!"

    # Verify class mapping
    with open(pkg_dir / "class_mapping.json") as f:
        cmap = json.load(f)
    assert cmap == {"CN": 0, "MCI": 1, "AD": 2}

    # Verify selected features
    with open(pkg_dir / "selected_features.json") as f:
        sel_feats = json.load(f)
    assert len(sel_feats) > 0
    assert all(isinstance(x, str) for x in sel_feats)


def test_zero_subject_leakage_and_no_masks() -> None:
    manifest_path = PROJECT_ROOT / "reports" / "data_audit" / "v2_clean_manifest.csv"
    assert manifest_path.is_file()

    df = pd.read_csv(manifest_path)

    # 1. Verify no masks
    mask_rows = df["processing_version"].str.contains("Mask", case=False, na=False)
    assert not mask_rows.any(), "Clean manifest must contain ZERO mask rows!"

    # 2. Verify zero subject leakage between splits
    train_subjs = set(df.loc[df["split"] == "train", "subject_id"])
    val_subjs = set(df.loc[df["split"] == "validation", "subject_id"])
    test_subjs = set(df.loc[df["split"] == "test", "subject_id"])

    assert not (train_subjs & val_subjs), "Train and Validation share subjects!"
    assert not (train_subjs & test_subjs), "Train and Test share subjects!"
    assert not (val_subjs & test_subjs), "Validation and Test share subjects!"

    # 3. Exactly 1 scan per physical scanning session
    sessions = df.groupby(["subject_id", "date_acquired"]).size()
    assert (sessions == 1).all(), "Every session must have exactly 1 deduplicated scan!"


def test_prohibit_archived_and_lr_hho_artifacts(tmp_path: Path) -> None:
    nifti_path = PROJECT_ROOT / "demo_inputs" / "mri" / "new_patient_scan.nii"
    xml_path = PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml"
    out_dir = tmp_path / "out"

    # Attempting to load from an archive path must raise ValueError
    with pytest.raises(ValueError, match="strictly prohibited"):
        run_live_single_pair_inference(
            nifti_path=nifti_path,
            xml_path=xml_path,
            output_dir=out_dir,
            artifact_pkg_dir=PROJECT_ROOT / "artifacts" / "archive" / "experiments_logistic_regression",
        )

    # Attempting to load from an HHO path must raise ValueError
    with pytest.raises(ValueError, match="strictly prohibited"):
        run_live_single_pair_inference(
            nifti_path=nifti_path,
            xml_path=xml_path,
            output_dir=out_dir,
            artifact_pkg_dir=PROJECT_ROOT / "artifacts" / "archive" / "hho_selection_v1",
        )


def test_live_inference_execution(tmp_path: Path) -> None:
    nifti_path = PROJECT_ROOT / "demo_inputs" / "mri" / "new_patient_scan.nii"
    xml_path = PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml"
    out_dir = tmp_path / "live_test"

    result = run_live_single_pair_inference(
        nifti_path=nifti_path,
        xml_path=xml_path,
        output_dir=out_dir,
        artifact_pkg_dir=PROJECT_ROOT / "artifacts" / "v2_gwo_tabpfn",
        modality="multimodal",
    )

    assert result.predicted_class in ["CN", "MCI", "AD"]
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4
    assert Path(result.report_html).is_file()
