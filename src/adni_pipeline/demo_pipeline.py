from __future__ import annotations

import csv
import json
import random
import time

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Required file is empty: {path}")

    return dataframe

def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load a JSON object if the file exists; otherwise return an empty dict."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")

    return data

def choose_demo_record(
    manifest_path: Path,
    subject_id: str | None,
    acquisition_group_id: str | None,
    random_seed: int,
) -> dict[str, Any]:
    dataframe = load_csv(manifest_path)

    required_columns = {
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "nifti_path",
        "xml_path",
        "processing_version",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Selected manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    dataframe = dataframe[
        dataframe["split"].astype(str) == "test"
    ].copy()

    if subject_id is not None:
        dataframe = dataframe[
            dataframe["subject_id"].astype(str)
            == str(subject_id)
        ]

    if acquisition_group_id is not None:
        dataframe = dataframe[
            dataframe["acquisition_group_id"].astype(str)
            == str(acquisition_group_id)
        ]

    if dataframe.empty:
        raise ValueError(
            "No matching test record was found in the selected manifest."
        )

    generator = random.Random(random_seed)
    selected_index = generator.choice(
        list(dataframe.index)
    )

    row = dataframe.loc[selected_index].to_dict()

    return {
        key: (
            None
            if pd.isna(value)
            else value
        )
        for key, value in row.items()
    }

def find_catalog_row(
    catalog_path: Path,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    if not catalog_path.exists():
        return None

    dataframe = pd.read_csv(catalog_path)

    if dataframe.empty:
        return None

    required = {
        "subject_id",
        "acquisition_group_id",
    }

    if not required.issubset(dataframe.columns):
        return None

    matches = dataframe[
        (
            dataframe["subject_id"].astype(str)
            == str(record["subject_id"])
        )
        & (
            dataframe["acquisition_group_id"].astype(str)
            == str(record["acquisition_group_id"])
        )
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()

def find_feature_row(
    feature_path: Path,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    return find_catalog_row(
        catalog_path=feature_path,
        record=record,
    )

def find_prediction_row(
    prediction_path: Path,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    if not prediction_path.exists():
        return None

    dataframe = pd.read_csv(prediction_path)

    required = {
        "subject_id",
        "acquisition_group_id",
        "prediction",
    }

    if not required.issubset(dataframe.columns):
        return None

    matches = dataframe[
        (
            dataframe["subject_id"].astype(str)
            == str(record["subject_id"])
        )
        & (
            dataframe["acquisition_group_id"].astype(str)
            == str(record["acquisition_group_id"])
        )
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()

def make_stage(
    name: str,
    status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": name,
        "status": status,
        "details": details,
    }

def run_demo(
    project_root: Path,
    mode: str,
    subject_id: str | None,
    acquisition_group_id: str | None,
    random_seed: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    if mode not in {
        "clinical",
        "radiomics",
        "multimodal",
    }:
        raise ValueError(
            "mode must be clinical, radiomics, or multimodal"
        )

    selected_manifest = (
        project_root
        / "reports"
        / "data_audit"
        / "selected_processing"
        / "selected_manifest.csv"
    )

    record = choose_demo_record(
        manifest_path=selected_manifest,
        subject_id=subject_id,
        acquisition_group_id=acquisition_group_id,
        random_seed=random_seed,
    )

    stages: list[dict[str, Any]] = []

    nifti_path = Path(record["nifti_path"])
    xml_path = Path(record["xml_path"])

    stages.append(
        make_stage(
            "Validated acquisition selected",
            "COMPLETE",
            {
                "subject_id": record["subject_id"],
                "acquisition_group_id": record[
                    "acquisition_group_id"
                ],
                "split": record["split"],
                "diagnosis_xml": record[
                    "diagnosis_xml"
                ],
                "processing_version": record[
                    "processing_version"
                ],
            },
        )
    )

    stages.append(
        make_stage(
            "Raw MRI and XML availability",
            "COMPLETE"
            if nifti_path.exists() and xml_path.exists()
            else "FAILED",
            {
                "nifti_path": str(nifti_path),
                "xml_path": str(xml_path),
                "nifti_exists": nifti_path.exists(),
                "xml_exists": xml_path.exists(),
            },
        )
    )

    preprocessing_catalog = (
        project_root
        / "data"
        / "derived"
        / "preprocessed"
        / "preprocessing_catalog.csv"
    )

    preprocessing_row = find_catalog_row(
        preprocessing_catalog,
        record,
    )

    stages.append(
        make_stage(
            "MRI preprocessing",
            "COMPLETE"
            if preprocessing_row is not None
            else "NOT_AVAILABLE",
            {
                "catalog": str(preprocessing_catalog),
                "preprocessed_path": (
                    preprocessing_row.get(
                        "preprocessed_nifti_path"
                    )
                    if preprocessing_row
                    else None
                ),
                "orientation": (
                    preprocessing_row.get(
                        "preprocessed_orientation"
                    )
                    if preprocessing_row
                    else None
                ),
                "preprocessed_shape": (
                    preprocessing_row.get(
                        "preprocessed_shape"
                    )
                    if preprocessing_row
                    else None
                ),
            },
        )
    )

    segmentation_catalog = (
        project_root
        / "data"
        / "derived"
        / "segmentations"
        / "segmentation_catalog.csv"
    )

    segmentation_row = find_catalog_row(
        segmentation_catalog,
        record,
    )

    stages.append(
        make_stage(
            "Segmentation and ROI masks",
            "COMPLETE"
            if segmentation_row is not None
            else "NOT_AVAILABLE",
            {
                "catalog": str(segmentation_catalog),
                "multiclass_mask_path": (
                    segmentation_row.get(
                        "multiclass_mask_path"
                    )
                    if segmentation_row
                    else None
                ),
                "brain_roi_mask_path": (
                    segmentation_row.get(
                        "brain_roi_mask_path"
                    )
                    if segmentation_row
                    else None
                ),
                "subcortical_roi_mask_path": (
                    segmentation_row.get(
                        "subcortical_roi_mask_path"
                    )
                    if segmentation_row
                    else None
                ),
            },
        )
    )

    radiomics_path = (
        project_root
        / "data"
        / "derived"
        / "radiomics"
        / "radiomics_features.csv"
    )

    radiomics_row = find_feature_row(
        radiomics_path,
        record,
    )

    stages.append(
        make_stage(
            "MRI radiomics features",
            "COMPLETE"
            if radiomics_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(radiomics_path),
                "feature_count": (
                    len(radiomics_row)
                    if radiomics_row
                    else 0
                ),
                "engine": "SimpleITK_Native",
            },
        )
    )

    clinical_path = (
        project_root
        / "data"
        / "derived"
        / "clinical"
        / "clinical_features.csv"
    )

    clinical_row = find_feature_row(
        clinical_path,
        record,
    )

    stages.append(
        make_stage(
            "Clinical XML features",
            "COMPLETE"
            if clinical_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(clinical_path),
                "clinical_fields": [
                    key
                    for key in (
                        clinical_row or {}
                    )
                    if key.startswith("clin_")
                ],
            },
        )
    )

    multimodal_path = (
        project_root
        / "data"
        / "derived"
        / "multimodal_features.csv"
    )

    multimodal_row = find_feature_row(
        multimodal_path,
        record,
    )

    stages.append(
        make_stage(
            "MRI and clinical multimodal alignment",
            "COMPLETE"
            if multimodal_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(multimodal_path),
                "mode": mode,
                "aligned": multimodal_row is not None,
            },
        )
    )

    selection_name = "baseline"

    selected_feature_path = (
        project_root
        / "data"
        / "derived"
        / "selected_features"
        / selection_name
        / mode
        / "test.csv"
    )

    stages.append(
        make_stage(
            "Train-only feature selection",
            "COMPLETE"
            if selected_feature_path.exists()
            else "NOT_AVAILABLE",
            {
                "method": selection_name,
                "mode": mode,
                "selected_test_table": str(
                    selected_feature_path
                ),
            },
        )
    )

    experiment_name = f"{mode}_baseline"

    prediction_path = (
        project_root
        / "experiments"
        / "tabpfn"
        / experiment_name
        / "test_predictions.csv"
    )

    prediction_row = find_prediction_row(
        prediction_path,
        record,
    )

    stages.append(
        make_stage(
            "TabPFN inference",
            "COMPLETE"
            if prediction_row is not None
            else "NOT_AVAILABLE",
            {
                "experiment": experiment_name,
                "prediction_path": str(
                    prediction_path
                ),
                "prediction": (
                    prediction_row.get(
                        "prediction"
                    )
                    if prediction_row
                    else None
                ),
                "probabilities": {
                    key: prediction_row.get(key)
                    for key in (
                        "probability_CN",
                        "probability_MCI",
                        "probability_AD",
                    )
                }
                if prediction_row
                else {},
            },
        )
    )

    explanation_root = (
        project_root
        / "experiments"
        / "tabpfn"
        / "clinical_baseline"
    )

    shap_summary_path = (
        explanation_root
        / "shap"
        / "shap_summary.json"
    )

    counterfactual_summary_path = (
        explanation_root
        / "counterfactuals"
        / "counterfactual_summary.json"
    )

    calibration_summary_path = (
        project_root
        / "reports"
        / "calibration_uncertainty"
        / "tabpfn_clinical_baseline_calibration_summary.json"
    )

    shap_summary = load_json_if_exists(
        shap_summary_path
    )

    counterfactual_summary = load_json_if_exists(
        counterfactual_summary_path
    )

    calibration_summary = load_json_if_exists(
        calibration_summary_path
    )

    stages.append(
        make_stage(
            "SHAP explanation",
            "COMPLETE"
            if shap_summary
            else "NOT_AVAILABLE",
            {
                "summary_path": str(
                    shap_summary_path
                ),
                "feature_count": shap_summary.get(
                    "feature_count"
                ),
                "shap_shape": shap_summary.get(
                    "shap_values_shape"
                ),
            },
        )
    )

    stages.append(
        make_stage(
            "Counterfactual explanation",
            "COMPLETE"
            if counterfactual_summary
            else "NOT_AVAILABLE",
            {
                "summary_path": str(
                    counterfactual_summary_path
                ),
                "successful_count": (
                    counterfactual_summary.get(
                        "successful_counterfactual_count"
                    )
                ),
            },
        )
    )

    stages.append(
        make_stage(
            "Uncertainty and calibration",
            "COMPLETE"
            if calibration_summary
            else "NOT_AVAILABLE",
            {
                "summary_path": str(
                    calibration_summary_path
                ),
                "ece": calibration_summary.get(
                    "expected_calibration_error_ece"
                ),
                "brier_score": calibration_summary.get(
                    "multiclass_brier_score"
                ),
            },
        )
    )

    completed_count = sum(
        stage["status"] == "COMPLETE"
        for stage in stages
    )

    demo_result = {
        "demo_mode": mode,
        "demo_type": "artifact_replay_inference",
        "random_seed": random_seed,
        "record": record,
        "stages": stages,
        "completed_stage_count": completed_count,
        "total_stage_count": len(stages),
        "runtime_seconds": time.perf_counter()
        - start_time,
        "raw_data_modified": False,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "important_note": (
            "This is a presentation replay using generated artifacts "
            "and saved predictions. It is not a new research evaluation."
        ),
    }

    return demo_result

def write_demo_outputs(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_dir / "demo_result.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    rows = []

    for stage in result["stages"]:
        rows.append(
            {
                "stage": stage["stage"],
                "status": stage["status"],
                "details": json.dumps(
                    stage["details"],
                    sort_keys=True,
                ),
            }
        )

    csv_path = output_dir / "demo_stage_status.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "status",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    html_path = output_dir / "demo_report.html"

    record = result["record"]

    stage_rows = "\n".join(
        (
            "<tr>"
            f"<td>{stage['stage']}</td>"
            f"<td>{stage['status']}</td>"
            f"<td><pre>{json.dumps(stage['details'], indent=2)}</pre></td>"
            "</tr>"
        )
        for stage in result["stages"]
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Alzheimer Pipeline Demonstration</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    color: #222;
    background: #f4f6f8;
}}
.card {{
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 5px #bbb;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 8px;
    vertical-align: top;
}}
th {{
    background: #26384a;
    color: white;
}}
.complete {{
    color: green;
    font-weight: bold;
}}
.not_available {{
    color: darkorange;
    font-weight: bold;
}}
pre {{
    white-space: pre-wrap;
    word-break: break-word;
}}
</style>
</head>
<body>
<div class="card">
<h1>Alzheimer's Disease Detection Pipeline Demonstration</h1>
<p><strong>Demo type:</strong> Artifact replay inference</p>
<p><strong>Mode:</strong> {result["demo_mode"]}</p>
<p><strong>Subject:</strong> {record.get("subject_id")}</p>
<p><strong>Acquisition group:</strong> {record.get("acquisition_group_id")}</p>
<p><strong>Split:</strong> {record.get("split")}</p>
<p><strong>XML diagnosis:</strong> {record.get("diagnosis_xml")}</p>
<p><strong>Processing version:</strong> {record.get("processing_version")}</p>
</div>

<div class="card">
<h2>Pipeline Stage Results</h2>
<table>
<tr>
<th>Stage</th>
<th>Status</th>
<th>Details</th>
</tr>
{stage_rows}
</table>
</div>

<div class="card">
<h2>Integrity Rules</h2>
<ul>
<li>Raw data modified: {result["raw_data_modified"]}</li>
<li>Subject ID used as feature: {result["subject_id_used_as_feature"]}</li>
<li>Diagnosis used as feature: {result["diagnosis_used_as_feature"]}</li>
<li>Target used as feature: {result["target_used_as_feature"]}</li>
</ul>
</div>
</body>
</html>
"""

    html_path.write_text(
        html,
        encoding="utf-8",
    )

    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "html": str(html_path.resolve()),
    }
