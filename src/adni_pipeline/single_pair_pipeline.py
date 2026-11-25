from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

from adni_audit.audit import parse_xml
from .clinical_parser_enhanced import parse_enhanced_clinical_xml
from .clinical_processor import DIAGNOSIS_LABEL_MAP, parse_numeric, parse_sex_binary
from .mri_visualization import create_mri_visualization
from .preprocessing import preprocess_single_nifti
from .radiomics_extractor import RadiomicsConfig, extract_features_for_pair
from .segmentation import OtsuMorphologicalBackend
from .vision_mamba_3d import extract_mamba_features

CLASS_NAMES = ["CN", "MCI", "AD"]


def sanitize_for_json(data: Any) -> Any:
    """Recursively converts NaN, Infinity, -Infinity and numpy types to JSON-safe representations (None/null, int, float, list, dict)."""
    if isinstance(data, dict):
        return {str(k): sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(item) for item in data]
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    if isinstance(data, (np.integer, int)):
        return int(data)
    if isinstance(data, (np.floating, float)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    if isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    if hasattr(data, "item") and callable(data.item):
        try:
            val = data.item()
            if isinstance(val, (float, np.floating)):
                if math.isnan(val) or math.isinf(val):
                    return None
                return float(val)
            return val
        except Exception:
            pass
    if isinstance(data, Path):
        return str(data)
    return data


def _format_score(
    val: Any, max_score: int | None = None, decimals: int | None = None
) -> str:
    """Format biomarker score for display; safely returns 'N/A' on missing/NaN."""
    if val is None:
        return "N/A"
    try:
        val_f = float(val)
        if np.isnan(val_f) or np.isinf(val_f):
            return "N/A"
        if decimals is not None:
            formatted = f"{val_f:.{decimals}f}"
        elif isinstance(val, (int, np.integer)) or val_f.is_integer():
            formatted = str(int(val_f))
        else:
            formatted = f"{val_f:.2f}".rstrip("0").rstrip(".")
        if max_score is not None:
            return f"{formatted} / {max_score}"
        return formatted
    except (ValueError, TypeError):
        return "N/A"

@dataclass
class LiveInferenceResult:
    subject_id: str
    series_id: str
    image_id: str
    diagnosis_xml: str
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: dict[str, float]
    predictive_entropy: float
    output_dir: str
    report_html: str
    clinical_features: dict[str, Any] | None = None
    mamba_features: dict[str, Any] | None = None
    original_images: dict[str, str] | None = None
    brain_roi_images: dict[str, str] | None = None
    subcortical_roi_images: dict[str, str] | None = None
    study_date: str = ""

def run_live_single_pair_inference(
    nifti_path: Path,
    xml_path: Path,
    output_dir: Path,
    artifact_pkg_dir: Path | None = None,
    experiment_dir: Path | None = None,
    feature_selection_dir: Path | None = None,
    model_inputs_dir: Path | None = None,
    modality: str = "multimodal",
) -> LiveInferenceResult:
    """Run live inference for one NIfTI + XML pair using saved training artifacts."""
    start_time = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]

    # Resolve active artifact package directory: prefer v5_unified_mamba_tabpfn, then v4_mamba_tabpfn
    if artifact_pkg_dir is None and experiment_dir is None:
        v5_dir = project_root / "artifacts" / "v5_unified_mamba_tabpfn"
        v4_dir = project_root / "artifacts" / "v4_mamba_tabpfn"
        if v5_dir.is_dir():
            artifact_pkg_dir = v5_dir
        elif v4_dir.is_dir():
            artifact_pkg_dir = v4_dir
        else:
            artifact_pkg_dir = project_root / "artifacts" / "v2_gwo_tabpfn"

    # Enforce strict prohibition on archived, Logistic Regression, or HHO artifacts
    for p in [artifact_pkg_dir, experiment_dir, feature_selection_dir, model_inputs_dir]:
        if p is not None:
            p_str = str(p).lower()
            if "archive" in p_str or "logistic_regression" in p_str or "hho" in p_str:
                raise ValueError(
                    f"Loading archived, Logistic Regression, or HHO artifacts is strictly prohibited in the active pipeline: {p}"
                )

    nifti_path = nifti_path.expanduser().resolve()
    xml_path = xml_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse XML Metadata (both standard and enhanced clinical parsing)
    canonical, flattened, xml_err = parse_xml(xml_path)
    if xml_err:
        raise ValueError(f"XML Parsing error: {xml_err}")

    enhanced_clin = parse_enhanced_clinical_xml(xml_path)

    subject_id = canonical.get("subject_id") or enhanced_clin.get("subject_id") or "UNKNOWN_SUBJECT"
    series_id = canonical.get("series_id") or enhanced_clin.get("series_id") or "UNKNOWN_SERIES"
    image_id = canonical.get("image_id") or enhanced_clin.get("image_id") or "UNKNOWN_IMAGE"
    study_date = canonical.get("study_date") or enhanced_clin.get("study_date") or ""
    dx_str = (canonical.get("diagnosis") or enhanced_clin.get("diagnosis") or "CN").upper()

    if dx_str not in DIAGNOSIS_LABEL_MAP:
        dx_str = "CN"

    # 2. Preprocess NIfTI (RAS+ orientation, 1.0 mm3 isotropic resampling, z-score normalization)
    prep_nifti_path = output_dir / "preprocessed_mri.nii.gz"
    out_img, prep_meta = preprocess_single_nifti(
        nifti_path=nifti_path,
        output_path=prep_nifti_path,
        target_spacing=(1.0, 1.0, 1.0),
    )

    # 3. Brain & Subcortical Visualization Mask Generation
    backend = OtsuMorphologicalBackend()
    prep_data = out_img.get_fdata(dtype=np.float32)
    multiclass_mask = backend.segment(prep_data)

    brain_roi = (multiclass_mask > 0).astype(np.uint8)
    subcortical_roi = (multiclass_mask == 2).astype(np.uint8)

    brain_roi_path = output_dir / "brain_roi.nii.gz"
    subcortical_roi_path = output_dir / "subcortical_roi.nii.gz"

    nib.save(
        nib.Nifti1Image(brain_roi, out_img.affine, out_img.header),
        str(brain_roi_path),
    )
    nib.save(
        nib.Nifti1Image(subcortical_roi, out_img.affine, out_img.header),
        str(subcortical_roi_path),
    )

    # 4. Deep Feature Extraction (3D Vision Mamba) + Optional Radiomics Fallback
    mamba_feats = extract_mamba_features(prep_nifti_path, embedding_dim=24)
    mamba_vec = np.array([mamba_feats[f"mamba_feat_{i:02d}"] for i in range(24)])
    mamba_norm = float(np.sqrt(np.sum(mamba_vec ** 2) + 1e-6))
    mamba_feats["mamba_norm"] = mamba_norm

    rad_config = RadiomicsConfig()
    brain_feats = extract_features_for_pair(
        image_path=prep_nifti_path,
        mask_path=brain_roi_path,
        prefix="brain",
        config=rad_config,
    )

    subcortical_feats = extract_features_for_pair(
        image_path=prep_nifti_path,
        mask_path=subcortical_roi_path,
        prefix="subcortical",
        config=rad_config,
    )

    # 5. Extract Clinical Features (Granular Staging)
    clin_feats = {
        "clin_age": parse_numeric(canonical.get("age") or enhanced_clin.get("clin_age")),
        "clin_sex": parse_sex_binary(canonical.get("sex") or enhanced_clin.get("clin_sex")),
        "clin_apoe_a1": parse_numeric(canonical.get("apoe_a1") or enhanced_clin.get("clin_apoe_a1")),
        "clin_apoe_a2": parse_numeric(canonical.get("apoe_a2") or enhanced_clin.get("clin_apoe_a2")),
        "clin_apoe4_carrier": parse_numeric(enhanced_clin.get("clin_apoe4_carrier")),
        "clin_apoe4_dose": parse_numeric(enhanced_clin.get("clin_apoe4_dose")),
        "clin_mmse": parse_numeric(canonical.get("mmse") or enhanced_clin.get("clin_mmse")),
        "clin_gds": parse_numeric(canonical.get("gds") or enhanced_clin.get("clin_gds")),
        "clin_cdr": parse_numeric(canonical.get("cdr") or enhanced_clin.get("clin_cdr")),
        "clin_cdrsb": parse_numeric(enhanced_clin.get("clin_cdrsb")),
        "clin_cdr_memory": parse_numeric(enhanced_clin.get("clin_cdr_memory")),
        "clin_cdr_orientation": parse_numeric(enhanced_clin.get("clin_cdr_orientation")),
        "clin_cdr_judgment": parse_numeric(enhanced_clin.get("clin_cdr_judgment")),
        "clin_faq": parse_numeric(enhanced_clin.get("clin_faq")),
        "clin_npi": parse_numeric(enhanced_clin.get("clin_npi")),
        "clin_weight": parse_numeric(canonical.get("weight") or enhanced_clin.get("clin_weight")),
        "feature_cdr_severity": parse_numeric(enhanced_clin.get("feature_cdr_severity")),
        "feature_cognitive_deficit": parse_numeric(enhanced_clin.get("feature_cognitive_deficit")),
        "feature_cdrsb_to_mmse": parse_numeric(enhanced_clin.get("feature_cdrsb_to_mmse")),
    }

    # Cross-modal interactions
    cdr_v = clin_feats["clin_cdr"] if not np.isnan(clin_feats["clin_cdr"]) else 0.0
    cog_v = clin_feats["feature_cognitive_deficit"] if not np.isnan(clin_feats["feature_cognitive_deficit"]) else 0.0
    clin_feats["feature_mamba_cdr_product"] = float(mamba_norm * (cdr_v + 0.1))
    clin_feats["feature_mamba_cognitive_interaction"] = float(mamba_norm * cog_v)
    mmse_v = clin_feats["clin_mmse"] if not np.isnan(clin_feats["clin_mmse"]) else 25.0
    cdrsb_v = clin_feats["clin_cdrsb"] if not np.isnan(clin_feats["clin_cdrsb"]) else (cdr_v * 3.0)
    clin_feats["feature_hacd"] = float(cdrsb_v / (np.clip(mmse_v, 0, 30) + 1.0))
    clin_feats["feature_mci_retention_index"] = float(
        (np.clip(mmse_v, 0, 30) / (cdrsb_v + 0.5)) * (1.0 / (mamba_norm + 1e-4))
    )
    # Default longitudinal trajectory rates for single-point live inference
    clin_feats["time_from_baseline_years"] = 0.0
    clin_feats["delta_mmse_annualized"] = 0.0
    clin_feats["delta_ventricle_annualized"] = 0.0
    clin_feats["delta_cdr_annualized"] = 0.0

    # 6. Build Raw Feature Vector
    raw_vector: dict[str, float] = {}
    raw_vector.update(clin_feats)
    raw_vector.update(mamba_feats)
    raw_vector.update(brain_feats)
    raw_vector.update(subcortical_feats)

    # 7. Check whether artifact package is Two-Stage (v4/v3) or Single Unified (v5/v2)
    is_two_stage = False
    if artifact_pkg_dir is not None:
        artifact_pkg_dir = artifact_pkg_dir.expanduser().resolve()
        if (artifact_pkg_dir / "model.joblib").is_file() and (artifact_pkg_dir / "selected_features.json").is_file():
            # Unified single-stage model (v5 or v2)
            is_two_stage = False
            feat_names_path = artifact_pkg_dir / "selected_features.json"
            input_cols_path = artifact_pkg_dir / "feature_names.json"
            imputer_path = artifact_pkg_dir / "imputer.joblib"
            scaler_path = artifact_pkg_dir / "scaler.joblib"
            model_path = artifact_pkg_dir / "model.joblib"
        elif (artifact_pkg_dir / "model_stage1.joblib").is_file():
            # Legacy two-stage hierarchical model
            is_two_stage = True
            input_cols_path = artifact_pkg_dir / "feature_names.json"
            imputer_path = artifact_pkg_dir / "imputer.joblib"
            scaler_path = artifact_pkg_dir / "scaler.joblib"
            s1_feat_path = artifact_pkg_dir / "stage1_selected_features.json"
            s2_feat_path = artifact_pkg_dir / "stage2_selected_features.json"
            s1_model_path = artifact_pkg_dir / "model_stage1.joblib"
            s2_model_path = artifact_pkg_dir / "model_stage2.joblib"
            s2_tree_path = artifact_pkg_dir / "model_stage2_tree.joblib"
            platt_path = artifact_pkg_dir / "stage2_platt_scaler.joblib"
            thresh_path = artifact_pkg_dir / "stage2_optimal_threshold.json"
        else:
            feat_names_path = artifact_pkg_dir / "selected_features.json"
            input_cols_path = artifact_pkg_dir / "feature_names.json"
            imputer_path = artifact_pkg_dir / "imputer.joblib"
            scaler_path = artifact_pkg_dir / "scaler.joblib"
            model_path = artifact_pkg_dir / "model.joblib"
    else:
        feat_names_path = feature_selection_dir / modality / "selected_features.json"
        if not feat_names_path.exists():
            feat_names_path = model_inputs_dir / modality / "feature_names.json"
        input_cols_path = model_inputs_dir / modality / "feature_names.json"
        imputer_path = model_inputs_dir / modality / "imputer.joblib"
        scaler_path = model_inputs_dir / modality / "scaler.joblib"
        model_path = experiment_dir / "model.joblib"

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    with input_cols_path.open(encoding="utf-8") as h:
        all_input_cols = json.load(h)

    feature_row = [raw_vector.get(col, np.nan) for col in all_input_cols]
    feature_matrix = np.array([feature_row], dtype=np.float64)

    imputed_matrix = imputer.transform(feature_matrix)
    scaled_matrix = scaler.transform(imputed_matrix)

    # 8. Model Inference Execution
    if is_two_stage:
        with s1_feat_path.open(encoding="utf-8") as h:
            s1_feats = json.load(h)
        with s2_feat_path.open(encoding="utf-8") as h:
            s2_feats = json.load(h)

        s1_indices = [all_input_cols.index(f) for f in s1_feats if f in all_input_cols]
        s2_indices = [all_input_cols.index(f) for f in s2_feats if f in all_input_cols]

        model_s1 = joblib.load(s1_model_path)
        model_s2 = joblib.load(s2_model_path)
        platt_scaler = joblib.load(platt_path) if platt_path.is_file() else None

        best_w = 0.50
        if thresh_path.is_file():
            with thresh_path.open(encoding="utf-8") as h:
                t_info = json.load(h)
                best_w = float(t_info.get("best_weight_tabpfn", 0.50))

        # Stage 1 Screening
        p_imp = float(model_s1.predict_proba(scaled_matrix[:, s1_indices])[0, 1])
        p_normal = 1.0 - p_imp

        if p_imp < 0.50:
            pred_label = 0
            pred_class = "CN"
            confidence = p_normal
            aligned_probs = np.array([p_normal, 0.5 * p_imp, 0.5 * p_imp])
        else:
            p2_tab = float(model_s2.predict_proba(scaled_matrix[:, s2_indices])[0, 1])
            if s2_tree_path.is_file():
                tree_model = joblib.load(s2_tree_path)
                p2_tree = float(tree_model.predict_proba(scaled_matrix[:, s2_indices])[0, 1])
                blend_p = best_w * p2_tab + (1.0 - best_w) * p2_tree
            else:
                blend_p = p2_tab

            if platt_scaler is not None:
                cal_p_ad = float(platt_scaler.predict_proba(np.array([[blend_p]]))[0, 1])
            else:
                cal_p_ad = blend_p

            cal_p_mci = 1.0 - cal_p_ad
            if cal_p_ad >= 0.50:
                pred_label = 2
                pred_class = "AD"
                confidence = cal_p_ad
            else:
                pred_label = 1
                pred_class = "MCI"
                confidence = cal_p_mci

            aligned_probs = np.array([p_normal, p_imp * cal_p_mci, p_imp * cal_p_ad])
    else:
        with feat_names_path.open(encoding="utf-8") as h:
            selected_feature_names = json.load(h)

        sel_indices = [all_input_cols.index(f) for f in selected_feature_names if f in all_input_cols]
        final_feature_vector = scaled_matrix[:, sel_indices]

        model = joblib.load(model_path)
        from sklearn.linear_model import LogisticRegression
        if isinstance(model, LogisticRegression):
            raise TypeError("Active pipeline requires TabPFN. Logistic Regression models are prohibited.")

        probs_raw = model.predict_proba(final_feature_vector)[0]

        # Check for auxiliary tree blend and temperature scaling
        if artifact_pkg_dir is not None:
            temp_path = artifact_pkg_dir / "temperature_scaling.json"
            tree_path = artifact_pkg_dir / "model_tree.joblib"
            if temp_path.is_file():
                with temp_path.open(encoding="utf-8") as h:
                    t_info = json.load(h)
                    opt_T = float(t_info.get("optimal_temperature_tabpfn", 1.0))
                    blend_w = float(t_info.get("blend_weight_tabpfn", 1.0))
                if blend_w < 1.0 and tree_path.is_file():
                    tree_model = joblib.load(tree_path)
                    probs_tree = tree_model.predict_proba(final_feature_vector)[0]
                    probs_raw = blend_w * probs_raw + (1.0 - blend_w) * probs_tree
                if abs(opt_T - 1.0) > 1e-4:
                    eps = 1e-12
                    logits = np.log(np.clip(probs_raw, eps, 1.0)) / opt_T
                    exp_logits = np.exp(logits - np.max(logits))
                    probs_raw = exp_logits / np.sum(exp_logits)

        classes = getattr(model, "classes_", np.array([0, 1, 2]))
        from .tabpfn_runner import align_probabilities
        aligned_probs = align_probabilities(np.array([probs_raw]), classes)[0]

        pred_label = int(np.argmax(aligned_probs))
        pred_class = CLASS_NAMES[pred_label]
        confidence = float(np.max(aligned_probs))

    prob_dict = {
        CLASS_NAMES[i]: float(aligned_probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Predictive Entropy
    eps = 1e-12
    clipped_probs = np.clip(aligned_probs, eps, 1.0)
    predictive_entropy = float(-np.sum(clipped_probs * np.log2(clipped_probs)))

    # 9. MRI Visualization: original, preprocessed, and brain ROI overlay
    original_vis_dir = output_dir / "mri_visualization" / "original"
    original_vis_summary = create_mri_visualization(
        nifti_path=nifti_path,
        output_dir=original_vis_dir,
        title_prefix=f"{subject_id} Original MRI",
    )
    original_images = original_vis_summary["base64_images"]

    vis_dir = output_dir / "mri_visualization" / "preprocessed"
    vis_summary = create_mri_visualization(
        nifti_path=prep_nifti_path,
        output_dir=vis_dir,
        mask_path=brain_roi_path,
        title_prefix=f"{subject_id} Preprocessed MRI + Brain ROI",
    )
    images = vis_summary["base64_images"]

    subcortical_vis_dir = output_dir / "mri_visualization" / "subcortical"
    subcortical_vis_summary = create_mri_visualization(
        nifti_path=prep_nifti_path,
        output_dir=subcortical_vis_dir,
        mask_path=subcortical_roi_path,
        title_prefix=f"{subject_id} Preprocessed MRI + Subcortical ROI",
    )
    subcortical_images = subcortical_vis_summary["base64_images"]

    # 10. Generate Live HTML Report
    html_path = output_dir / "live_inference_report.html"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Live Alzheimer's Inference Report - {subject_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 30px; background: #f3f5f7; color: #222; }}
.card {{ background: white; padding: 22px; margin-bottom: 22px; border-radius: 8px; box-shadow: 0 1px 5px #bbb; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
.image-card {{ text-align: center; }}
.image-card img {{ width: 100%; border: 1px solid #aaa; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
td, th {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
th {{ background: #26384a; color: white; }}
.pred-box {{ background: #e8f4f8; border-left: 6px solid #2980b9; padding: 15px; margin-top: 10px; }}
.badge {{ background: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
</style>
</head>
<body>

<div class="card">
<h1>Live Alzheimer's Disease Inference Report</h1>
<p><strong>Subject ID:</strong> {subject_id} | <strong>Series ID:</strong> {series_id} | <strong>Image ID:</strong> {image_id}</p>
<p><strong>XML Diagnosis (Ground Truth):</strong> <span class="badge">{dx_str}</span></p>
<p><strong>NIfTI Source:</strong> {nifti_path.name}</p>
<p><strong>XML Source:</strong> {xml_path.name}</p>
</div>

<div class="card">
<h2>1. Model Prediction & Confidence</h2>
<div class="pred-box">
    <h3>Predicted Diagnosis: <span style="color:#c0392b;">{pred_class}</span> (Label {pred_label})</h3>
    <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
    <p><strong>Predictive Uncertainty (Entropy):</strong> {predictive_entropy:.4f} bits</p>
    <table>
        <tr><th>Class</th><th>Predicted Probability</th></tr>
        <tr><td>CN (Cognitively Normal)</td><td>{prob_dict.get('CN', 0.0) * 100:.2f}%</td></tr>
        <tr><td>MCI (Mild Cognitive Impairment)</td><td>{prob_dict.get('MCI', 0.0) * 100:.2f}%</td></tr>
        <tr><td>AD (Alzheimer's Disease)</td><td>{prob_dict.get('AD', 0.0) * 100:.2f}%</td></tr>
    </table>
</div>
</div>

<div class="card">
<h2>2. Stage 1 — Original MRI</h2>
<p>Views generated directly from the input NIfTI before this pipeline's preprocessing.</p>
<div class="grid">
    <div class="image-card"><h3>Axial</h3><img src="data:image/png;base64,{original_images['axial']}"></div>
    <div class="image-card"><h3>Coronal</h3><img src="data:image/png;base64,{original_images['coronal']}"></div>
    <div class="image-card"><h3>Sagittal</h3><img src="data:image/png;base64,{original_images['sagittal']}"></div>
</div>
<p><strong>Original shape:</strong> {original_vis_summary['shape']} |
<strong>Voxel spacing (mm):</strong> {original_vis_summary['voxel_spacing_mm']} |
<strong>Orientation:</strong> {original_vis_summary['orientation']}</p>
</div>

<div class="card">
<h2>3. Stage 2 — Preprocessed MRI + Brain ROI Overlay</h2>
<p>Preprocessing applies RAS reorientation, target-spacing resampling and non-zero voxel z-score normalization. The overlay is generated by the current segmentation backend.</p>
<div class="grid">
    <div class="image-card"><h3>Axial</h3><img src="data:image/png;base64,{images['axial']}"></div>
    <div class="image-card"><h3>Coronal</h3><img src="data:image/png;base64,{images['coronal']}"></div>
    <div class="image-card"><h3>Sagittal</h3><img src="data:image/png;base64,{images['sagittal']}"></div>
</div>
<p><strong>Preprocessed shape:</strong> {vis_summary['shape']} |
<strong>Voxel spacing (mm):</strong> {vis_summary['voxel_spacing_mm']} |
<strong>Orientation:</strong> {vis_summary['orientation']}</p>
</div>

<div class="card">
<h2>4. Stage 3 — Subcortical ROI Overlay (Deep Gray Matter & Ventricles)</h2>
<p>Subcortical segmentation isolates key neurodegenerative target regions including hippocampal margins and ventricular spaces.</p>
<div class="grid">
    <div class="image-card"><h3>Axial</h3><img src="data:image/png;base64,{subcortical_images['axial']}"></div>
    <div class="image-card"><h3>Coronal</h3><img src="data:image/png;base64,{subcortical_images['coronal']}"></div>
    <div class="image-card"><h3>Sagittal</h3><img src="data:image/png;base64,{subcortical_images['sagittal']}"></div>
</div>
<p><strong>Subcortical Mask Voxel Count:</strong> {subcortical_vis_summary.get('mask_voxel_count', 'N/A')}</p>
</div>

<div class="card">
<h2>5. Parsed Clinical Biomarkers & Granular Dementia Staging</h2>
<table>
    <tr><th>Biomarker / Clinical Term</th><th>Value</th><th>Clinical Context</th></tr>
    <tr><td>Global CDR Score</td><td>{_format_score(clin_feats.get('clin_cdr'), decimals=1)}</td><td>0.0: Normal, 0.5: MCI, 1.0+: Dementia</td></tr>
    <tr><td>CDR Sum of Boxes (CDRSB)</td><td><strong>{_format_score(clin_feats.get('clin_cdrsb'), decimals=2)}</strong></td><td>Granular continuous staging (0.0 – 18.0)</td></tr>
    <tr><td>CDR Memory Sub-score</td><td>{_format_score(clin_feats.get('clin_cdr_memory'), decimals=1)}</td><td>Amnestic deficit domain score</td></tr>
    <tr><td>CDR Orientation / Judgment</td><td>Orient: {_format_score(clin_feats.get('clin_cdr_orientation'))} | Judge: {_format_score(clin_feats.get('clin_cdr_judgment'))}</td><td>Cognitive executive staging</td></tr>
    <tr><td>MMSE Score</td><td>{_format_score(clin_feats.get('clin_mmse'), max_score=30)}</td><td>Cognitive Deficit Index: {_format_score(clin_feats.get('feature_cognitive_deficit'), decimals=3)}</td></tr>
    <tr><td>FAQ Score (Daily Living)</td><td>{_format_score(clin_feats.get('clin_faq'), max_score=30)}</td><td>Functional impairment index</td></tr>
    <tr><td>APOE4 Status</td><td>Carrier: {_format_score(clin_feats.get('clin_apoe4_carrier'))} (Dose: {_format_score(clin_feats.get('clin_apoe4_dose'))})</td><td>Alleles: A1={canonical.get('apoe_a1', 'N/A')}, A2={canonical.get('apoe_a2', 'N/A')}</td></tr>
    <tr><td>Demographics</td><td>Age: {_format_score(clin_feats.get('clin_age'), decimals=1)} yrs | Sex: {'Female' if clin_feats.get('clin_sex') == 1.0 else ('Male' if clin_feats.get('clin_sex') == 0.0 else 'N/A')} | Weight: {_format_score(clin_feats.get('clin_weight'), decimals=1)} kg</td><td>Patient characteristics</td></tr>
    <tr><td>CDR-Squared Severity</td><td>{_format_score(clin_feats.get('feature_cdr_severity'), decimals=2)}</td><td>(CDR)<sup>2</sup> &times; Age interaction</td></tr>
</table>
</div>

<div class="card">
<h2>6. 3D Vision Mamba (Med-Mamba) Deep Representation</h2>
<p>Extracted from frozen 3D State Space Model via bidirectional token scanning over isotropic 1.0 mm<sup>3</sup> volume:</p>
<table>
    <tr><th>Mamba Embedding Metric</th><th>Value</th></tr>
    <tr><td>Anatomical Feature Dimension</td><td>24 Deep Spatial Dimensions</td></tr>
    <tr><td>Deep Spatial Norm (||E||<sub>2</sub>)</td><td>{_format_score(mamba_norm, decimals=4)}</td></tr>
    <tr><td>Mamba &times; CDR Interaction</td><td>{_format_score(clin_feats.get('feature_mamba_cdr_product', 0.0), decimals=4)}</td></tr>
    <tr><td>Mamba &times; Cognitive Deficit</td><td>{_format_score(clin_feats.get('feature_mamba_cognitive_interaction', 0.0), decimals=4)}</td></tr>
</table>
</div>

<div class="card">
<h2>7. Execution & Pipeline Integrity</h2>
<ul>
    <li><strong>Raw NIfTI / XML Modified:</strong> False (Read-only input)</li>
    <li><strong>Subject ID Used as Feature:</strong> False (Metadata key only)</li>
    <li><strong>Diagnosis Used as Feature:</strong> False (Ground-truth target label only)</li>
    <li><strong>Pipeline Execution Time:</strong> {time.perf_counter() - start_time:.2f} seconds</li>
</ul>
</div>

</body>
</html>
"""

    # Inject standalone dashboard assets after the HTML f-string is evaluated.
    asset_dir = Path(__file__).resolve().parents[2] / "reports" / "templates"
    css_path = asset_dir / "report.css"
    js_path = asset_dir / "report.js"

    if not css_path.is_file() or not js_path.is_file():
        raise FileNotFoundError(
            f"Dashboard assets missing: {css_path} or {js_path}"
        )

    dashboard_css = css_path.read_text(encoding="utf-8")
    dashboard_js = js_path.read_text(encoding="utf-8")

    html_content = html_content.replace(
        "</head>",
        "<style>\n" + dashboard_css + "\n</style>\n</head>",
        1,
    )
    html_content = html_content.replace(
        "</body>",
        "<script>\n" + dashboard_js + "\n</script>\n</body>",
        1,
    )

    html_path.write_text(html_content, encoding="utf-8")

    import subprocess
    import sys
    project_root = Path(__file__).resolve().parents[2]

    # Append the existing global SHAP explanation to every new report.
    shap_helper = project_root / "scripts" / "append_shap_to_report.py"
    if shap_helper.is_file():
        subprocess.run(
            [sys.executable, str(shap_helper), str(html_path)],
            check=False,
        )

    # Append the global counterfactual experiment section to every new report.
    counterfactual_helper = (
        project_root / "scripts" / "append_counterfactual_to_report.py"
    )
    if counterfactual_helper.is_file():
        subprocess.run(
            [sys.executable, str(counterfactual_helper), str(html_path)],
            check=False,
        )

    # 11. Save Summary JSON
    summary_json = {
        "subject_id": subject_id,
        "series_id": series_id,
        "image_id": image_id,
        "study_date": study_date,
        "diagnosis_xml": dx_str,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "predictive_entropy": predictive_entropy,
        "execution_time_seconds": time.perf_counter() - start_time,
        "report_html_path": str(html_path.resolve()),
        "clinical_features": clin_feats,
        "mamba_features": {
            "mamba_norm": mamba_norm,
            "feature_mamba_cdr_product": clin_feats.get("feature_mamba_cdr_product", 0.0),
            "feature_mamba_cognitive_interaction": clin_feats.get("feature_mamba_cognitive_interaction", 0.0),
        },
        "visualizations": {
            "original": {
                "axial": str(original_vis_summary["png_paths"].get("axial")),
                "coronal": str(original_vis_summary["png_paths"].get("coronal")),
                "sagittal": str(original_vis_summary["png_paths"].get("sagittal")),
                "shape": original_vis_summary.get("shape"),
                "voxel_spacing_mm": original_vis_summary.get("voxel_spacing_mm"),
            },
            "brain_roi": {
                "axial": str(vis_summary["png_paths"].get("axial")),
                "coronal": str(vis_summary["png_paths"].get("coronal")),
                "sagittal": str(vis_summary["png_paths"].get("sagittal")),
                "mask_voxel_count": vis_summary.get("mask_voxel_count"),
            },
            "subcortical_roi": {
                "axial": str(subcortical_vis_summary["png_paths"].get("axial")),
                "coronal": str(subcortical_vis_summary["png_paths"].get("coronal")),
                "sagittal": str(subcortical_vis_summary["png_paths"].get("sagittal")),
                "mask_voxel_count": subcortical_vis_summary.get("mask_voxel_count"),
            },
        },
    }

    clean_summary = sanitize_for_json(summary_json)

    with (output_dir / "live_inference_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(clean_summary, handle, indent=2, allow_nan=False)

    return LiveInferenceResult(
        subject_id=subject_id,
        series_id=series_id,
        image_id=image_id,
        diagnosis_xml=dx_str,
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=float(confidence),
        probabilities=clean_summary["probabilities"],
        predictive_entropy=float(predictive_entropy),
        output_dir=str(output_dir.resolve()),
        report_html=str(html_path.resolve()),
        clinical_features=clean_summary["clinical_features"],
        mamba_features=clean_summary["mamba_features"],
        original_images=original_images,
        brain_roi_images=images,
        subcortical_roi_images=subcortical_images,
        study_date=study_date,
    )
