from __future__ import annotations

import json
import os
from pathlib import Path
import re
import pytest
import tornado.testing

os.environ.setdefault("ASYNC_TEST_TIMEOUT", "60")

from adni_pipeline.web_server import make_app, DEMO_CATALOG

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestWebApp(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def get_async_test_timeout(self):
        return 60.0

    def test_health_endpoint(self):
        response = self.fetch("/api/health")
        assert response.code == 200
        data = json.loads(response.body.decode("utf-8"))
        assert data.get("status") == "healthy"

    def test_demos_endpoint(self):
        response = self.fetch("/api/demos")
        assert response.code == 200
        data = json.loads(response.body.decode("utf-8"))
        assert data.get("status") == "success"
        assert len(data.get("demos", [])) == len(DEMO_CATALOG)
        for demo in data["demos"]:
            assert "id" in demo
            assert "title" in demo
            assert demo["available"] is True

    def test_index_page(self):
        response = self.fetch("/")
        assert response.code == 200
        body = response.body.decode("utf-8")
        assert "NeuroScan AI" in body
        assert "Multi-Planar Neuroimaging & ROI Slices" in body
        assert "Patient Demographics & Granular Dementia Staging" in body
        assert "Upload Structural MRI" in body

    def test_analyze_demo_preset(self):
        # Test analyze endpoint with preset demo
        body = "demo_id=new_cn_scan&modality=multimodal"
        response = self.fetch(
            "/api/analyze",
            method="POST",
            body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            request_timeout=60.0,
            connect_timeout=60.0,
        )
        assert response.code == 200
        data = json.loads(response.body.decode("utf-8"))
        assert data.get("status") == "success"
        assert data.get("predicted_class") in ["CN", "MCI", "AD"]
        assert "probabilities" in data
        assert "confidence" in data
        assert "clinical_features" in data
        assert "visualizations" in data
        vis = data["visualizations"]
        assert "brain_roi_images" in vis
        assert "subcortical_roi_images" in vis
        assert "original_images" in vis
        assert "axial" in vis["brain_roi_images"]
        assert "coronal" in vis["brain_roi_images"]
        assert "sagittal" in vis["brain_roi_images"]

    def test_analyze_demo_preset_with_missing_faq(self):
        # adni_141_s_0696 has missing FAQ/NPI clinical fields that previously produced unquoted NaN
        def _reject_nan(token: str):
            raise ValueError(f"Unacceptable non-standard JSON token: {token}")

        body = "demo_id=adni_141_s_0696&modality=multimodal"
        response = self.fetch(
            "/api/analyze",
            method="POST",
            body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            request_timeout=60.0,
            connect_timeout=60.0,
        )
        assert response.code == 200
        raw_body = response.body.decode("utf-8")
        assert not re.search(r':\s*NaN\b', raw_body)
        assert not re.search(r':\s*Infinity\b', raw_body)

        data = json.loads(raw_body, parse_constant=_reject_nan)
        assert data.get("status") == "success"
        assert data.get("predicted_class") in ["CN", "MCI", "AD"]
        assert data["clinical_features"]["clin_faq"] is None
        assert data["clinical_features"]["clin_npi"] is None
