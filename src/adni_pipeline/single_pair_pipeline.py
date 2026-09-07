from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

from adni_audit.audit import parse_xml
from .clinical_processor import DIAGNOSIS_LABEL_MAP, parse_numeric, parse_sex_binary
from .mri_visualization import create_mri_visualization
from .preprocessing import preprocess_single_nifti
from .radiomics_extractor import RadiomicsConfig, extract_features_for_pair
from .segmentation import OtsuMorphologicalBackend

CLASS_NAMES = ["CN", "MCI", "AD"]

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

def run_live_single_pair_inference(
    nifti_path: Path,
    xml_path: Path,
    experiment_dir: Path,
    feature_selection_dir: Path,
    model_inputs_dir: Path,
    output_dir: Path,
    modality: str = "multimodal",
) -> LiveInferenceResult:
    """Run live inference for one NIfTI + XML pair using saved training artifacts."""
    start_time = time.perf_counter()

    nifti_path = nifti_path.expanduser().resolve()
    xml_path = xml_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse XML Metadata
    canonical, flattened, xml_err = parse_xml(xml_path)
    if xml_err:
        raise ValueError(f"XML Parsing error: {xml_err}")

    subject_id = canonical.get("subject_id", "UNKNOWN_SUBJECT")
    series_id = canonical.get("series_id", "UNKNOWN_SERIES")
    image_id = canonical.get("image_id", "UNKNOWN_IMAGE")
    dx_str = canonical.get("diagnosis", "CN").upper()

    if dx_str not in DIAGNOSIS_LABEL_MAP:
        dx_str = "CN"

    # 2. Preprocess NIfTI
    prep_nifti_path = output_dir / "preprocessed_mri.nii.gz"
    out_img, prep_meta = preprocess_single_nifti(
        nifti_path=nifti_path,
        output_path=prep_nifti_path,
        target_spacing=(1.0, 1.0, 1.0),
    )

    # 3. Segmentation & ROI Mask Generation
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

    # 4. Extract Radiomics Features
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

    # 5. Extract Clinical Features
    clin_feats = {
        "clin_age": parse_numeric(canonical.get("age")),
        "clin_sex": parse_sex_binary(canonical.get("sex")),
        "clin_apoe_a1": parse_numeric(canonical.get("apoe_a1")),
        "clin_apoe_a2": parse_numeric(canonical.get("apoe_a2")),
        "clin_mmse": parse_numeric(canonical.get("mmse")),
        "clin_gds": parse_numeric(canonical.get("gds")),
        "clin_cdr": parse_numeric(canonical.get("cdr")),
        "clin_weight": parse_numeric(canonical.get("weight")),
    }

    # 6. Build Raw Feature Vector
    raw_vector: dict[str, float] = {}
    if modality in ["clinical", "multimodal"]:
        raw_vector.update(clin_feats)
    if modality in ["radiomics", "multimodal"]:
        raw_vector.update(brain_feats)
        raw_vector.update(subcortical_feats)

    # Load Feature Names list
    feat_names_path = (
        feature_selection_dir / modality / "selected_features.json"
    )
    if not feat_names_path.exists():
        feat_names_path = model_inputs_dir / modality / "feature_names.json"

    with feat_names_path.open(encoding="utf-8") as h:
        selected_feature_names = json.load(h)

    # 7. Apply Saved Imputer & Scaler
    imputer_path = model_inputs_dir / modality / "imputer.joblib"
    scaler_path = model_inputs_dir / modality / "scaler.joblib"

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    # Align features with training column order
    input_cols_path = model_inputs_dir / modality / "feature_names.json"
    with input_cols_path.open(encoding="utf-8") as h:
        all_input_cols = json.load(h)

    feature_row = [raw_vector.get(col, np.nan) for col in all_input_cols]
    feature_matrix = np.array([feature_row], dtype=np.float64)

    # Transform
    imputed_matrix = imputer.transform(feature_matrix)
    scaled_matrix = scaler.transform(imputed_matrix)

    # Filter to Selected Features
    all_input_cols_arr = np.array(all_input_cols)
    selected_mask = np.isin(all_input_cols_arr, selected_feature_names)
    final_feature_vector = scaled_matrix[:, selected_mask]

    # 8. Model Prediction
    model_path = experiment_dir / "model.joblib"
    model = joblib.load(model_path)

    probs_raw = model.predict_proba(final_feature_vector)[0]
    pred_label = int(np.argmax(probs_raw))
    pred_class = CLASS_NAMES[pred_label]
    confidence = float(np.max(probs_raw))

    prob_dict = {
        CLASS_NAMES[i]: float(probs_raw[i])
        for i in range(len(CLASS_NAMES))
    }

    # Predictive Entropy
    eps = 1e-12
    clipped_probs = np.clip(probs_raw, eps, 1.0)
    predictive_entropy = float(-np.sum(clipped_probs * np.log2(clipped_probs)))

    # 9. MRI Visualization
    vis_dir = output_dir / "mri_visualization"
    vis_summary = create_mri_visualization(
        nifti_path=prep_nifti_path,
        output_dir=vis_dir,
        mask_path=brain_roi_path,
        title_prefix=f"{subject_id} Live Inference",
    )

    # 10. Generate Live HTML Report
    html_path = output_dir / "live_inference_report.html"
    images = vis_summary["base64_images"]

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
<h2>2. Preprocessed 3D MRI Slices (Isotropic 1mm RAS)</h2>
<div class="grid">
    <div class="image-card"><h3>Axial</h3><img src="data:image/png;base64,{images['axial']}"></div>
    <div class="image-card"><h3>Coronal</h3><img src="data:image/png;base64,{images['coronal']}"></div>
    <div class="image-card"><h3>Sagittal</h3><img src="data:image/png;base64,{images['sagittal']}"></div>
</div>
</div>

<div class="card">
<h2>3. Parsed Clinical XML Parameters</h2>
<table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Age</td><td>{canonical.get('age', 'N/A')}</td></tr>
    <tr><td>Sex</td><td>{canonical.get('sex', 'N/A')}</td></tr>
    <tr><td>APOE Alleles</td><td>A1: {canonical.get('apoe_a1', 'N/A')} | A2: {canonical.get('apoe_a2', 'N/A')}</td></tr>
    <tr><td>MMSE Score</td><td>{canonical.get('mmse', 'N/A')}</td></tr>
    <tr><td>GDS Score</td><td>{canonical.get('gds', 'N/A')}</td></tr>
    <tr><td>CDR Score</td><td>{canonical.get('cdr', 'N/A')}</td></tr>
    <tr><td>Weight (kg)</td><td>{canonical.get('weight', 'N/A')}</td></tr>
</table>
</div>

<div class="card">
<h2>4. Execution & Pipeline Integrity</h2>
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

    html_path.write_text(html_content, encoding="utf-8")

    # 11. Save Summary JSON
    summary_json = {
        "subject_id": subject_id,
        "series_id": series_id,
        "image_id": image_id,
        "diagnosis_xml": dx_str,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "predictive_entropy": predictive_entropy,
        "execution_time_seconds": time.perf_counter() - start_time,
        "report_html_path": str(html_path.resolve()),
    }

    with (output_dir / "live_inference_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary_json, handle, indent=2)

    return LiveInferenceResult(
        subject_id=subject_id,
        series_id=series_id,
        image_id=image_id,
        diagnosis_xml=dx_str,
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=confidence,
        probabilities=prob_dict,
        predictive_entropy=predictive_entropy,
        output_dir=str(output_dir.resolve()),
        report_html=str(html_path.resolve()),
    )
