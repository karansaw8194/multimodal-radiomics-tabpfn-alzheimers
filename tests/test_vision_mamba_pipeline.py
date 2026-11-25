"""Unit and Integration Tests for 3D Vision Mamba (Med-Mamba) + TabPFN Pipeline."""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pytest
import torch

from adni_pipeline.clinical_parser_enhanced import parse_enhanced_clinical_xml
from adni_pipeline.vision_mamba_3d import (
    MedMamba3D,
    extract_mamba_features,
    get_frozen_mamba_model,
    preprocess_volume_for_mamba,
)
from adni_pipeline.single_pair_pipeline import run_live_single_pair_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vision_mamba_architecture_and_forward() -> None:
    """Verify MedMamba3D model shape, parameters, and deterministic frozen inference."""
    model = MedMamba3D(embedding_dim=24, seed=42)
    model.eval()

    # Verify no gradients
    for param in model.parameters():
        assert not param.requires_grad, "Vision Mamba parameters must be frozen!"

    # Forward pass on synthetic 3D volume
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert out1.shape == (1, 24)
    # Check determinism
    assert torch.allclose(out1, out2, atol=1e-5), "Forward pass must be strictly deterministic!"


def test_preprocess_volume_for_mamba() -> None:
    """Verify non-zero normalization and resampling to (96, 96, 96)."""
    raw_vol = np.zeros((120, 140, 100), dtype=np.float32)
    raw_vol[20:100, 30:110, 20:80] = np.random.uniform(10.0, 50.0, (80, 80, 60))

    tensor = preprocess_volume_for_mamba(raw_vol, target_shape=(96, 96, 96))
    assert tensor.shape == (1, 1, 96, 96, 96)
    assert isinstance(tensor, torch.Tensor)
    assert not torch.isnan(tensor).any()


def test_enhanced_clinical_parser_adni_xml() -> None:
    """Verify parsing of granular CDRSB, sub-scores, APOE4, and interaction terms."""
    xml_path = PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml"
    if xml_path.is_file():
        res = parse_enhanced_clinical_xml(xml_path)
        assert "clin_cdr" in res
        assert "clin_cdrsb" in res
        assert "clin_cdr_memory" in res
        assert "feature_cdr_severity" in res
        assert "feature_cognitive_deficit" in res
        assert res["clin_cdrsb"] >= 0.0


def test_v4_artifact_package_integrity() -> None:
    """Verify all artifacts for V4 Two-Stage Mamba + TabPFN exist and load cleanly."""
    pkg_dir = PROJECT_ROOT / "artifacts" / "v4_mamba_tabpfn"
    assert pkg_dir.is_dir(), f"V4 Artifact package directory missing: {pkg_dir}"

    required_files = [
        "model_stage1.joblib",
        "model_stage2.joblib",
        "stage2_platt_scaler.joblib",
        "imputer.joblib",
        "scaler.joblib",
        "feature_names.json",
        "stage1_selected_features.json",
        "stage2_selected_features.json",
        "stage2_optimal_threshold.json",
        "class_mapping.json",
        "pipeline_config.json",
        "metadata.json",
    ]

    for fname in required_files:
        fpath = pkg_dir / fname
        assert fpath.is_file(), f"Missing required V4 artifact: {fname}"

    with open(pkg_dir / "stage1_selected_features.json") as f:
        s1_feats = json.load(f)
    with open(pkg_dir / "stage2_selected_features.json") as f:
        s2_feats = json.load(f)

    # Constraint check: 10 to 18 features
    assert 10 <= len(s1_feats) <= 18, f"Stage 1 features count {len(s1_feats)} violates [10, 18] constraint!"
    assert 10 <= len(s2_feats) <= 18, f"Stage 2 features count {len(s2_feats)} violates [10, 18] constraint!"


def test_live_inference_with_v4_mamba_package(tmp_path: Path) -> None:
    """Verify end-to-end live inference execution with the V4 Mamba package."""
    nifti_path = PROJECT_ROOT / "demo_inputs" / "mri" / "new_patient_scan.nii"
    xml_path = PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml"
    out_dir = tmp_path / "v4_live_test"

    result = run_live_single_pair_inference(
        nifti_path=nifti_path,
        xml_path=xml_path,
        output_dir=out_dir,
        artifact_pkg_dir=PROJECT_ROOT / "artifacts" / "v4_mamba_tabpfn",
    )

    assert result.predicted_class in ["CN", "MCI", "AD"]
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4
    assert Path(result.report_html).is_file()
