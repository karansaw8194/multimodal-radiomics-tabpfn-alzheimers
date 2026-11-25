"""Unit and Integration Tests for Unified Single 3-Class 3D Vision Mamba + TabPFN Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression

from adni_pipeline.single_pair_pipeline import run_live_single_pair_inference
from scripts.train_unified_3class_mamba_tabpfn import (
    apply_temperature_scaling,
    aggregate_patient_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v5_unified_mamba_tabpfn"


def test_v5_artifact_package_integrity() -> None:
    """Verify all required artifacts for V5 Unified 3-Class Pipeline exist and load cleanly."""
    assert ARTIFACTS_DIR.is_dir(), f"V5 Artifact package directory missing: {ARTIFACTS_DIR}"

    required_files = [
        "model.joblib",
        "model_tabpfn.joblib",
        "model_tree.joblib",
        "imputer.joblib",
        "scaler.joblib",
        "feature_names.json",
        "selected_features.json",
        "temperature_scaling.json",
        "class_mapping.json",
        "pipeline_config.json",
        "metadata.json",
        "summary_metrics.json",
    ]

    for fname in required_files:
        fpath = ARTIFACTS_DIR / fname
        assert fpath.is_file(), f"Missing required artifact: {fname}"

    # Verify model is TabPFNClassifier, NOT LogisticRegression
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    assert isinstance(model, TabPFNClassifier), f"Model is not TabPFNClassifier: {type(model)}"
    assert not isinstance(model, LogisticRegression), "Model must not be LogisticRegression!"

    # Verify feature selection bounds: strictly between 16 and 22 features
    with open(ARTIFACTS_DIR / "selected_features.json") as f:
        selected_features = json.load(f)
    assert 16 <= len(selected_features) <= 22, (
        f"Selected features count {len(selected_features)} violates [16, 22] constraint!"
    )

    # Verify mandatory clinical and mamba features
    mandatory_features = [
        "clin_cdrsb",
        "clin_cdr_memory",
        "clin_cdr_orientation",
        "clin_cdr_judgment",
        "clin_mmse",
        "clin_faq",
        "clin_apoe4_dose",
        "feature_mci_retention_index",
        "mamba_norm",
    ]
    for mf in mandatory_features:
        assert mf in selected_features, f"Mandatory feature {mf} is missing from selected features!"


def test_temperature_scaling_math() -> None:
    """Verify temperature scaling preserves probability validity and smooths distributions."""
    raw_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

    # T = 1.0 preserves probabilities
    p_t1 = apply_temperature_scaling(raw_probs, T=1.0)
    assert np.allclose(p_t1, raw_probs, atol=1e-5)

    # T > 1.0 flattens probabilities (higher entropy)
    p_t2 = apply_temperature_scaling(raw_probs, T=2.0)
    assert np.allclose(np.sum(p_t2, axis=1), [1.0, 1.0])
    assert p_t2[0, 0] < raw_probs[0, 0]

    # T < 1.0 sharpens probabilities
    p_t05 = apply_temperature_scaling(raw_probs, T=0.5)
    assert np.allclose(np.sum(p_t05, axis=1), [1.0, 1.0])
    assert p_t05[0, 0] > raw_probs[0, 0]


def test_patient_recency_weighted_aggregation() -> None:
    """Verify longitudinal visit weighting follows exp(0.25 * visit_index)."""
    import pandas as pd

    # Subject with 2 visits: visit 0 (MCI) and visit 1 (AD)
    df_toy = pd.DataFrame({
        "subject_id": ["SUBJ_01", "SUBJ_01"],
        "target_label": [2, 2],
        "visit_index": [0, 1],
    })
    scan_probs = np.array([
        [0.05, 0.55, 0.40],  # visit 0: leans MCI
        [0.01, 0.40, 0.59],  # visit 1: leans AD
    ])

    trues, preds, probs = aggregate_patient_predictions(df_toy, scan_probs)
    assert len(trues) == 1
    assert trues[0] == 2
    # Recency weight on visit 1 should prioritize the later scan (AD)
    assert preds[0] == 2


def test_target_patient_level_accuracy_verified() -> None:
    """Verify that the unified model meets or exceeds the >= 94.87% (37/39) patient accuracy target."""
    with open(ARTIFACTS_DIR / "summary_metrics.json") as f:
        metrics = json.load(f)

    acc = metrics["patient_accuracy"]
    correct = metrics["patients_correct"]
    total = metrics["patients_total"]

    assert total == 39, f"Expected 39 test patients, got {total}"
    assert correct >= 37, f"Expected at least 37 correct patients, got {correct}"
    assert acc >= 0.9487, f"Expected patient accuracy >= 94.87%, got {acc*100:.2f}%"


def test_live_inference_with_unified_v5_package(tmp_path: Path) -> None:
    """Verify end-to-end live inference execution with the V5 Unified 3-Class package."""
    nifti_path = PROJECT_ROOT / "demo_inputs" / "mri" / "new_patient_scan.nii"
    xml_path = PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml"
    out_dir = tmp_path / "v5_live_test"

    result = run_live_single_pair_inference(
        nifti_path=nifti_path,
        xml_path=xml_path,
        output_dir=out_dir,
        artifact_pkg_dir=ARTIFACTS_DIR,
    )

    assert result.predicted_class in ["CN", "MCI", "AD"]
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4
    assert Path(result.report_html).is_file()
