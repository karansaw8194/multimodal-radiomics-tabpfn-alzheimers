from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import tornado.httpserver
import tornado.ioloop
import tornado.web

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.single_pair_pipeline import (
    CLASS_NAMES,
    LiveInferenceResult,
    run_live_single_pair_inference,
    sanitize_for_json,
)

logger = logging.getLogger("adni_web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXECUTOR = ThreadPoolExecutor(max_workers=2)

DEMO_CATALOG = [
    {
        "id": "adni_141_s_0696",
        "title": "ADNI 141_S_0696 (Confirmed AD)",
        "description": "Patient with established Alzheimer's Disease (CDR 1.0, MMSE 20, APOE4 carrier).",
        "expected_class": "AD",
        "nifti_path": PROJECT_ROOT / "demo_inputs" / "mri" / "ADNI_141_S_0696_MR_MPR__GradWarp__B1_Correction__Mask_ICS_20071119104742622_S18373_I82730.nii",
        "xml_path": PROJECT_ROOT / "demo_inputs" / "clinical" / "ADNI_141_S_0696_MPR__GradWarp__B1_Correction__Mask_S18373_I82730.xml",
    },
    {
        "id": "new_cn_scan",
        "title": "Patient 941_S_1197 (Cognitively Normal Control)",
        "description": "Healthy elderly control (CDR 0.0, MMSE 29, high cognitive reserve).",
        "expected_class": "CN",
        "nifti_path": PROJECT_ROOT / "demo_inputs" / "mri" / "new_CN_scan.nii",
        "xml_path": PROJECT_ROOT / "demo_inputs" / "clinical" / "new_CN_metadata.xml",
    },
    {
        "id": "new_patient_scan",
        "title": "Clinical Patient Evaluation (Scan 01)",
        "description": "Diagnostic evaluation case with complete multimodality biomarkers.",
        "expected_class": "Evaluation",
        "nifti_path": PROJECT_ROOT / "demo_inputs" / "mri" / "new_patient_scan.nii",
        "xml_path": PROJECT_ROOT / "demo_inputs" / "clinical" / "new_patient_metadata.xml",
    },
    {
        "id": "new_scan_02",
        "title": "Follow-Up Patient Evaluation (Scan 02)",
        "description": "Longitudinal follow-up scan paired with ADNI XML clinical staging.",
        "expected_class": "Evaluation",
        "nifti_path": PROJECT_ROOT / "demo_inputs" / "mri" / "new_scan_02.nii",
        "xml_path": PROJECT_ROOT / "demo_inputs" / "clinical" / "new_scan_02_metadata.xml",
    },
]


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self) -> None:
        self.set_status(204)
        self.finish()

    def write_json(self, data: Any, status: int = 200) -> None:
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        clean_data = sanitize_for_json(data)
        self.write(json.dumps(clean_data, indent=2, allow_nan=False))


class IndexHandler(BaseHandler):
    def get(self) -> None:
        html_file = Path(__file__).resolve().parent / "web" / "index.html"
        if not html_file.is_file():
            self.write("<h1>Web interface not found. Please verify deployment.</h1>")
            return
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(html_file.read_text(encoding="utf-8"))


class DemosHandler(BaseHandler):
    def get(self) -> None:
        out = []
        for d in DEMO_CATALOG:
            nii_ok = d["nifti_path"].is_file()
            xml_ok = d["xml_path"].is_file()
            size_mb = round(d["nifti_path"].stat().st_size / (1024 * 1024), 2) if nii_ok else 0
            out.append({
                "id": d["id"],
                "title": d["title"],
                "description": d["description"],
                "expected_class": d["expected_class"],
                "available": nii_ok and xml_ok,
                "nifti_filename": d["nifti_path"].name,
                "xml_filename": d["xml_path"].name,
                "nifti_size_mb": size_mb,
            })
        self.write_json({"status": "success", "demos": out})


def _execute_inference_task(
    nifti_path: Path,
    xml_path: Path,
    output_dir: Path,
    modality: str,
) -> dict[str, Any]:
    result: LiveInferenceResult = run_live_single_pair_inference(
        nifti_path=nifti_path,
        xml_path=xml_path,
        output_dir=output_dir,
        modality=modality,
    )

    summary_file = output_dir / "live_inference_summary.json"
    summary_data = {}
    if summary_file.is_file():
        try:
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load summary json: {e}")

    # Build relative URLs for browser downloads
    rel_out = output_dir.relative_to(PROJECT_ROOT)
    base_url = f"/{rel_out.as_posix()}"

    res_dict = {
        "status": "success",
        "subject_id": result.subject_id,
        "series_id": result.series_id,
        "image_id": result.image_id,
        "study_date": result.study_date or summary_data.get("study_date", ""),
        "diagnosis_xml": result.diagnosis_xml,
        "predicted_class": result.predicted_class,
        "predicted_label": result.predicted_label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "predictive_entropy": result.predictive_entropy,
        "execution_time_seconds": summary_data.get("execution_time_seconds", 0.0),
        "clinical_features": result.clinical_features or summary_data.get("clinical_features", {}),
        "mamba_features": result.mamba_features or summary_data.get("mamba_features", {}),
        "visualizations": {
            "original_images": result.original_images or {},
            "brain_roi_images": result.brain_roi_images or {},
            "subcortical_roi_images": result.subcortical_roi_images or {},
            "metadata": summary_data.get("visualizations", {}),
        },
        "downloads": {
            "report_html_url": f"{base_url}/live_inference_report.html",
            "summary_json_url": f"{base_url}/live_inference_summary.json",
            "preprocessed_mri_url": f"{base_url}/preprocessed_mri.nii.gz",
            "brain_roi_url": f"{base_url}/brain_roi.nii.gz",
            "subcortical_roi_url": f"{base_url}/subcortical_roi.nii.gz",
        },
        "output_dir": str(output_dir),
    }
    return res_dict


class AnalyzeHandler(BaseHandler):
    async def post(self) -> None:
        session_id = uuid.uuid4().hex[:12]
        output_dir = PROJECT_ROOT / "reports" / "web_sessions" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        modality = self.get_argument("modality", "multimodal")
        demo_id = self.get_argument("demo_id", "").strip()

        target_nifti: Path | None = None
        target_xml: Path | None = None

        if demo_id:
            demo_match = next((d for d in DEMO_CATALOG if d["id"] == demo_id), None)
            if not demo_match:
                self.write_json(
                    {"status": "error", "message": f"Demo identifier not recognized: {demo_id}"},
                    status=400,
                )
                return
            target_nifti = demo_match["nifti_path"]
            target_xml = demo_match["xml_path"]
            if not target_nifti.is_file() or not target_xml.is_file():
                self.write_json(
                    {"status": "error", "message": "Demo files missing on disk."},
                    status=404,
                )
                return
        else:
            # Handle uploaded files
            mri_files = self.request.files.get("mri_file")
            xml_files = self.request.files.get("xml_file")

            if not mri_files or len(mri_files) == 0:
                self.write_json(
                    {"status": "error", "message": "No MRI file uploaded (.nii or .nii.gz)."},
                    status=400,
                )
                return

            if not xml_files or len(xml_files) == 0:
                self.write_json(
                    {"status": "error", "message": "No Clinical XML file uploaded (.xml)."},
                    status=400,
                )
                return

            mri_item = mri_files[0]
            xml_item = xml_files[0]

            mri_filename = Path(mri_item["filename"]).name
            xml_filename = Path(xml_item["filename"]).name

            if not (mri_filename.endswith(".nii") or mri_filename.endswith(".nii.gz")):
                self.write_json(
                    {
                        "status": "error",
                        "message": f"Invalid MRI file format: {mri_filename}. Expected .nii or .nii.gz",
                    },
                    status=400,
                )
                return

            if not xml_filename.endswith(".xml"):
                self.write_json(
                    {
                        "status": "error",
                        "message": f"Invalid Clinical Metadata format: {xml_filename}. Expected .xml",
                    },
                    status=400,
                )
                return

            upload_dir = output_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            target_nifti = upload_dir / mri_filename
            target_xml = upload_dir / xml_filename

            target_nifti.write_bytes(mri_item["body"])
            target_xml.write_bytes(xml_item["body"])

        logger.info(f"Starting inference for session {session_id} on {target_nifti.name} & {target_xml.name}")

        try:
            loop = tornado.ioloop.IOLoop.current()
            result_data = await loop.run_in_executor(
                EXECUTOR,
                _execute_inference_task,
                target_nifti,
                target_xml,
                output_dir,
                modality,
            )
            self.write_json(result_data)
        except Exception as exc:
            logger.exception(f"Inference failed for session {session_id}: {exc}")
            self.write_json(
                {
                    "status": "error",
                    "message": f"Pipeline analysis error: {str(exc)}",
                    "session_id": session_id,
                },
                status=500,
            )


class HealthHandler(BaseHandler):
    def get(self) -> None:
        self.write_json({"status": "healthy", "service": "adni_multimodal_radiomics_platform"})


def make_app() -> tornado.web.Application:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    static_dir = Path(__file__).resolve().parent / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[Any] = [
        (r"/", IndexHandler),
        (r"/api/demos", DemosHandler),
        (r"/api/analyze", AnalyzeHandler),
        (r"/api/health", HealthHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": str(static_dir)}),
        (r"/reports/(.*)", tornado.web.StaticFileHandler, {"path": str(reports_dir)}),
    ]

    return tornado.web.Application(
        handlers,
        debug=False,
        autoreload=False,
    )


def run_server(port: int = 8000, host: str = "0.0.0.0") -> None:
    app = make_app()
    # 250 MB max buffer size to handle high-resolution 3D NIfTI files
    server = tornado.httpserver.HTTPServer(app, max_buffer_size=250 * 1024 * 1024)
    server.listen(port, address=host)
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("=" * 70)
    print("  ADNI MULTIMODAL RADIOMICS & TABPFN CLINICAL PLATFORM")
    print("=" * 70)
    print(f"  Web Interface available at: http://{display_host}:{port}")
    print(f"  API Health:                http://{display_host}:{port}/api/health")
    print(f"  Demo Catalog:              http://{display_host}:{port}/api/demos")
    print("=" * 70)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    run_server()
