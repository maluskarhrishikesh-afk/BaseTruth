from __future__ import annotations

import io

import cv2
import numpy as np
from starlette.testclient import TestClient


def _tiny_jpeg() -> bytes:
    """Return a tiny valid JPEG for the upload endpoint."""
    img = np.full((32, 32, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_face_scan_endpoint_returns_canonical_contract(monkeypatch) -> None:
    from basetruth.api import create_app
    from basetruth.face_scan import service

    monkeypatch.setattr(
        service,
        "run_face_scan_static",
        lambda _file_bytes, filename, environment=None: {
            "filename": filename,
            "scan_type": "face_scan",
            "mode": "static",
            "schema_version": "1.0.0",
            "verdict": "GENUINE",
            "risk_score_0_100": 12.5,
            "confidence_0_100": 88.0,
            "confidence_reason": "Clear image.",
            "overall_explanation": "Low spoof-risk signals.",
            "honest_review": "Looks genuine.",
            "evidence": ["One face detected."],
            "trace": {"decision_trace_id": "fs_test", "processing_time_ms": 10, "rules_version": "face-scan-rules-1.0.0", "model_version": "heuristics-only", "timestamp_utc": "2026-05-01T00:00:00Z"},
            "environment": {"platform": "upload", "virtual_camera_suspected": False},
            "checks": {"face_detection": {"face_count": 1}, "active_liveness": {"status": "not_run"}},
            "artifacts": {"best_frame_available": False, "challenge_snapshots_available": False},
        },
    )

    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post("/api/v1/face-scan", files={"file": ("selfie.jpg", io.BytesIO(_tiny_jpeg()), "image/jpeg")})

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["scan_type"] == "face_scan"
    assert payload["schema_version"] == "1.0.0"
    assert payload["risk_score_0_100"] == 12.5
    assert payload["confidence_0_100"] == 88.0


def test_face_scan_endpoint_maps_no_face_to_400(monkeypatch) -> None:
    from basetruth.api import create_app
    from basetruth.face_scan import service

    monkeypatch.setattr(
        service,
        "run_face_scan_static",
        lambda _file_bytes, _filename, environment=None: {"error": "No face found in the image.", "error_type": "no_face"},
    )

    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post("/api/v1/face-scan", files={"file": ("selfie.jpg", io.BytesIO(_tiny_jpeg()), "image/jpeg")})

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "No face found in the image."