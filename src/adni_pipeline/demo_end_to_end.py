from __future__ import annotations

import base64
import csv
import html
import json
import random

from pathlib import Path
from typing import Any

import pandas as pd

from .mri_visualization import create_mri_visualization

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def select_record(
    manifest_path: Path,
    subject_id: str | None,
    acquisition_group_id: str | None,
    random_seed: int,
) -> dict[str, Any]:
    dataframe = load_csv(manifest_path)

    if dataframe.empty:
        raise ValueError(
            f"Selected manifest is empty or unavailable: {manifest_path}"
        )

    required = {
        "subject_id",
        "acquisition_group_id",
        "split",
        "diagnosis_xml",
        "nifti_path",
        "xml_path",
        "processing_version",
    }

    missing = required - set(dataframe.columns)

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
            "No matching test acquisition was found."
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

def find_matching_row(
    path: Path,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    dataframe = load_csv(path)

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

def stage(
    name: str,
    status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "details": details,
    }

def build_stage_table(
    stages: list[dict[str, Any]],
) -> str:
    rows = []

    for item in stages:
        status_class = item["status"].lower()

        rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td class='{status_class}'>"
            f"{html.escape(item['status'])}</td>"
            "<td><pre>"
            f"{html.escape(json.dumps(item['details'], indent=2))}"
            "</pre></td>"
            "</tr>"
        )

    return "\n".join(rows)

def create_html_report(
    result: dict[str, Any],
    visualization: dict[str, Any],
    output_path: Path,
) -> None:
    record = result["record"]
    stages = result["stages"]
    images = visualization["base64_images"]

    image_cards = []

    for plane in (
        "axial",
        "coronal",
        "sagittal",
    ):
        image_cards.append(
            f"""
            <div class="image-card">
                <h3>{plane.title()}</h3>
                <img src="data:image/png;base64,{images[plane]}">
            </div>
            """
        )

    image_html = "\n".join(image_cards)
    stage_html = build_stage_table(stages)

    prediction = result.get("prediction", {})

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>End-to-End Alzheimer's MRI Demonstration</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    background: #f3f5f7;
    color: #222;
}}
.card {{
    background: white;
    padding: 22px;
    margin-bottom: 22px;
    border-radius: 8px;
    box-shadow: 0 1px 5px #bbb;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}}
.image-card {{
    text-align: center;
}}
.image-card img {{
    width: 100%;
    border: 1px solid #aaa;
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
.failed {{
    color: red;
    font-weight: bold;
}}
pre {{
    white-space: pre-wrap;
    word-break: break-word;
}}
.metric {{
    display: inline-block;
    padding: 12px;
    margin: 5px;
    background: #e8f0f8;
    border-radius: 5px;
}}
</style>
</head>
<body>

<div class="card">
<h1>End-to-End Alzheimer's MRI Demonstration</h1>
<p><strong>Demo type:</strong> Artifact-based single-acquisition trace</p>
<p><strong>Mode:</strong> {html.escape(result["mode"])}</p>
<p><strong>Subject:</strong> {html.escape(str(record.get("subject_id")))}</p>
<p><strong>Acquisition group:</strong>
{html.escape(str(record.get("acquisition_group_id")))}</p>
<p><strong>Split:</strong> {html.escape(str(record.get("split")))}</p>
<p><strong>XML diagnosis:</strong>
{html.escape(str(record.get("diagnosis_xml")))}</p>
<p><strong>Processing version:</strong>
{html.escape(str(record.get("processing_version")))}</p>
</div>

<div class="card">
<h2>MRI Visualization</h2>
<p><strong>NIfTI:</strong>
{html.escape(str(record.get("nifti_path")))}</p>
<p><strong>Shape:</strong> {visualization["shape"]}</p>
<p><strong>Spacing:</strong>
{visualization["voxel_spacing_mm"]} mm</p>
<p><strong>Orientation:</strong>
{html.escape(str(visualization["orientation"]))}</p>
<div class="grid">
{image_html}
</div>
</div>

<div class="card">
<h2>Prediction</h2>
<pre>{html.escape(json.dumps(prediction, indent=2))}</pre>
</div>

<div class="card">
<h2>Pipeline Stages</h2>
<table>
<tr>
<th>Stage</th>
<th>Status</th>
<th>Details</th>
</tr>
{stage_html}
</table>
</div>

<div class="card">
<h2>Integrity Rules</h2>
<ul>
<li>Raw data modified: False</li>
<li>Subject ID used as feature: False</li>
<li>Diagnosis used as feature: False</li>
<li>Target used as feature: False</li>
<li>Demo represents a single acquisition, not dataset-level performance</li>
</ul>
</div>

</body>
</html>
"""

    output_path.write_text(
        html_content,
        encoding="utf-8",
    )

def run_end_to_end_demo(
    project_root: Path,
    output_dir: Path,
    mode: str,
    subject_id: str | None,
    acquisition_group_id: str | None,
    random_seed: int,
) -> dict[str, Any]:
    if mode not in {
        "clinical",
        "radiomics",
        "multimodal",
    }:
        raise ValueError(
            "mode must be clinical, radiomics, or multimodal"
        )

    manifest_path = (
        project_root
        / "reports"
        / "data_audit"
        / "selected_processing"
        / "selected_manifest.csv"
    )

    record = select_record(
        manifest_path=manifest_path,
        subject_id=subject_id,
        acquisition_group_id=acquisition_group_id,
        random_seed=random_seed,
    )

    run_dir = output_dir.resolve()
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inputs_dir = run_dir / "inputs"
    images_dir = run_dir / "images"
    stages_dir = run_dir / "stages"

    inputs_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    stages_dir.mkdir(exist_ok=True)

    nifti_path = Path(record["nifti_path"])
    xml_path = Path(record["xml_path"])

    if not nifti_path.exists():
        raise FileNotFoundError(
            f"Selected NIfTI does not exist: {nifti_path}"
        )

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Selected XML does not exist: {xml_path}"
        )

    with (
        inputs_dir / "selected_input.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            record,
            handle,
            indent=2,
            sort_keys=True,
        )

    visualization = create_mri_visualization(
        nifti_path=nifti_path,
        output_dir=images_dir,
        title_prefix=(
            f"{record['subject_id']} MRI"
        ),
    )

    preprocessing_catalog = (
        project_root
        / "data"
        / "derived"
        / "preprocessed"
        / "preprocessing_catalog.csv"
    )

    segmentation_catalog = (
        project_root
        / "data"
        / "derived"
        / "segmentations"
        / "segmentation_catalog.csv"
    )

    radiomics_path = (
        project_root
        / "data"
        / "derived"
        / "radiomics"
        / "radiomics_features.csv"
    )

    clinical_path = (
        project_root
        / "data"
        / "derived"
        / "clinical"
        / "clinical_features.csv"
    )

    multimodal_path = (
        project_root
        / "data"
        / "derived"
        / "multimodal_features.csv"
    )

    preprocessing_row = find_matching_row(
        preprocessing_catalog,
        record,
    )

    segmentation_row = find_matching_row(
        segmentation_catalog,
        record,
    )

    radiomics_row = find_matching_row(
        radiomics_path,
        record,
    )

    clinical_row = find_matching_row(
        clinical_path,
        record,
    )

    multimodal_row = find_matching_row(
        multimodal_path,
        record,
    )

    stages = [
        stage(
            "Validated acquisition selected",
            "COMPLETE",
            {
                "manifest": str(manifest_path),
                "subject_id": record[
                    "subject_id"
                ],
                "acquisition_group_id": record[
                    "acquisition_group_id"
                ],
            },
        ),
        stage(
            "MRI and XML input validation",
            "COMPLETE",
            {
                "nifti_path": str(nifti_path),
                "xml_path": str(xml_path),
                "nifti_exists": nifti_path.exists(),
                "xml_exists": xml_path.exists(),
            },
        ),
        stage(
            "MRI visualization",
            "COMPLETE",
            {
                "image_directory": str(images_dir),
                "shape": visualization["shape"],
                "orientation": visualization[
                    "orientation"
                ],
                "spacing_mm": visualization[
                    "voxel_spacing_mm"
                ],
            },
        ),
        stage(
            "MRI preprocessing artifact",
            "COMPLETE"
            if preprocessing_row is not None
            else "NOT_AVAILABLE",
            {
                "catalog": str(
                    preprocessing_catalog
                ),
                "record_found": preprocessing_row
                is not None,
            },
        ),
        stage(
            "Segmentation and ROI artifacts",
            "COMPLETE"
            if segmentation_row is not None
            else "NOT_AVAILABLE",
            {
                "catalog": str(
                    segmentation_catalog
                ),
                "record_found": segmentation_row
                is not None,
            },
        ),
        stage(
            "Radiomics features",
            "COMPLETE"
            if radiomics_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(radiomics_path),
                "record_found": radiomics_row
                is not None,
            },
        ),
        stage(
            "Clinical XML features",
            "COMPLETE"
            if clinical_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(clinical_path),
                "record_found": clinical_row
                is not None,
            },
        ),
        stage(
            "Multimodal alignment",
            "COMPLETE"
            if multimodal_row is not None
            else "NOT_AVAILABLE",
            {
                "feature_table": str(
                    multimodal_path
                ),
                "record_found": multimodal_row
                is not None,
            },
        ),
    ]

    prediction = {}

    prediction_path = (
        project_root
        / "experiments"
        / "tabpfn"
        / f"{mode}_baseline"
        / "test_predictions.csv"
    )

    prediction_row = find_matching_row(
        prediction_path,
        record,
    )

    if prediction_row is not None:
        prediction = prediction_row

        stages.append(
            stage(
                "TabPFN inference",
                "COMPLETE",
                {
                    "prediction_file": str(
                        prediction_path
                    ),
                    "prediction": prediction_row.get(
                        "prediction"
                    ),
                    "probabilities": {
                        key: prediction_row.get(key)
                        for key in (
                            "probability_CN",
                            "probability_MCI",
                            "probability_AD",
                        )
                    },
                },
            )
        )
    else:
        stages.append(
            stage(
                "TabPFN inference",
                "NOT_AVAILABLE",
                {
                    "prediction_file": str(
                        prediction_path
                    ),
                    "reason": (
                        "No matching prediction row was found "
                        "for this acquisition."
                    ),
                },
            )
        )

    result = {
        "mode": mode,
        "record": record,
        "prediction": prediction,
        "stages": stages,
        "raw_data_modified": False,
        "subject_id_used_as_feature": False,
        "diagnosis_used_as_feature": False,
        "target_used_as_feature": False,
        "note": (
            "This report visualizes one validated acquisition and "
            "previously generated artifacts. It is not a new evaluation."
        ),
    }

    with (
        stages_dir / "stage_status.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            stages,
            handle,
            indent=2,
            sort_keys=True,
        )

    with (
        run_dir / "demo_result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    create_html_report(
        result=result,
        visualization=visualization,
        output_path=run_dir / "demo_report.html",
    )

    return result
