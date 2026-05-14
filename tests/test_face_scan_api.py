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


# ── Video endpoints ───────────────────────────────────────────────────────────

def test_get_video_endpoint_returns_403_when_flag_off(monkeypatch) -> None:
    """GET /api/v1/face-scan/sessions/{session_id}/video must return 403 when flag is off."""
    monkeypatch.delenv("FACE_SCAN_RECORD_VIDEO", raising=False)
    from basetruth.api import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.get("/api/v1/face-scan/sessions/test-sess-123/video")
    assert resp.status_code == 403, f"Expected 403 when flag is off, got {resp.status_code}"


def test_get_video_endpoint_returns_404_when_no_video_key(monkeypatch) -> None:
    """GET /api/v1/face-scan/sessions/{session_id}/video returns 404 when no video exists."""
    import os
    monkeypatch.setenv("FACE_SCAN_RECORD_VIDEO", "true")
    from basetruth.api import create_app
    from unittest.mock import patch, MagicMock

    # DB row exists but video_key is None → 404
    fake_row = MagicMock()
    fake_row.video_key = None

    with patch("basetruth.db.get_face_scan_live_result", return_value=fake_row):
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.get("/api/v1/face-scan/sessions/test-sess-no-vid/video")

    assert resp.status_code == 404, f"Expected 404 when video_key is None, got {resp.status_code}"


def test_get_video_endpoint_returns_200_with_presigned_url(monkeypatch) -> None:
    """GET /api/v1/face-scan/sessions/{session_id}/video returns a presigned URL on success."""
    monkeypatch.setenv("FACE_SCAN_RECORD_VIDEO", "true")
    from basetruth.api import create_app
    from unittest.mock import patch, MagicMock

    fake_row = MagicMock()
    fake_row.video_key = "face-scan-video/test-sess.mp4"
    fake_row.verdict = "GENUINE"

    with patch("basetruth.db.get_face_scan_live_result", return_value=fake_row), \
         patch("basetruth.store.get_face_scan_video_presigned_url",
               return_value="https://minio.example.com/face-scan-video/test-sess.mp4"):
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.get("/api/v1/face-scan/sessions/test-sess/video")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "video_url" in data
    assert "minio.example.com" in data["video_url"]
    assert data["expires_in_seconds"] == 3600


def test_delete_video_endpoint_returns_404_when_no_video(monkeypatch) -> None:
    """DELETE /api/v1/face-scan/sessions/{session_id}/video returns 404 when no video exists."""
    monkeypatch.setenv("FACE_SCAN_RECORD_VIDEO", "true")
    from basetruth.api import create_app
    from unittest.mock import patch

    with patch("basetruth.db.get_face_scan_live_result", return_value=None):
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.delete("/api/v1/face-scan/sessions/missing-sess/video")

    assert resp.status_code == 404


def test_delete_video_endpoint_returns_200_and_calls_delete(monkeypatch) -> None:
    """DELETE /api/v1/face-scan/sessions/{session_id}/video deletes from MinIO and nulls DB key."""
    monkeypatch.setenv("FACE_SCAN_RECORD_VIDEO", "true")
    from basetruth.api import create_app
    from unittest.mock import patch, MagicMock, call

    fake_row = MagicMock()
    fake_row.video_key = "face-scan-video/del-sess.mp4"

    with patch("basetruth.db.get_face_scan_live_result", return_value=fake_row), \
         patch("basetruth.store.delete_face_scan_video", return_value=True) as mock_del, \
         patch("basetruth.db.update_face_scan_live_video_key", return_value=True) as mock_update:
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.delete("/api/v1/face-scan/sessions/del-sess/video")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json()["deleted"] is True
    mock_del.assert_called_once_with("face-scan-video/del-sess.mp4")
    mock_update.assert_called_once_with("del-sess", None)


def test_list_face_scan_live_results_returns_rows() -> None:
    """list_face_scan_live_results returns a list of ORM-like objects without touching the DB."""
    from unittest.mock import MagicMock, patch
    from basetruth.db import list_face_scan_live_results

    # Build two fake rows that look like detached FaceScanLiveResult instances.
    row1 = MagicMock(session_id="sess-aaa", verdict="GENUINE", video_key="face-scan-video/sess-aaa.mp4")
    row2 = MagicMock(session_id="sess-bbb", verdict="SUSPICIOUS", video_key=None)

    with patch("basetruth.db.db_session") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        (
            mock_session.query.return_value
            .order_by.return_value
            .limit.return_value
            .all.return_value
        ) = [row1, row2]

        rows = list_face_scan_live_results(limit=10)

    assert len(rows) == 2
    assert rows[0].session_id == "sess-aaa"
    assert rows[1].video_key is None


def test_list_face_scan_live_results_returns_empty_on_error() -> None:
    """list_face_scan_live_results returns [] gracefully when the DB raises."""
    from unittest.mock import patch
    from basetruth.db import list_face_scan_live_results

    with patch("basetruth.db.db_session", side_effect=RuntimeError("DB down")):
        rows = list_face_scan_live_results(limit=5)

    assert rows == []
