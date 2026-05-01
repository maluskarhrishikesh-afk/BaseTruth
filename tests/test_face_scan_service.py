from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np


def _tiny_face_image(fill_value: int = 160) -> bytes:
    """Return a valid JPEG for static Face Scan tests."""
    img = np.full((180, 180, 3), fill_value, dtype=np.uint8)
    cv2.rectangle(img, (45, 45), (135, 135), (fill_value - 20, fill_value - 20, fill_value - 20), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _fake_face() -> SimpleNamespace:
    """Build a minimal face object compatible with the Face Scan service."""
    return SimpleNamespace(
        bbox=np.array([40.0, 40.0, 140.0, 140.0], dtype=np.float32),
        kps=np.array(
            [
                [68.0, 76.0],
                [112.0, 76.0],
                [90.0, 95.0],
                [72.0, 120.0],
                [108.0, 120.0],
            ],
            dtype=np.float32,
        ),
        det_score=0.97,
    )


def test_run_face_scan_static_returns_canonical_contract(monkeypatch) -> None:
    from basetruth.face_scan import service

    monkeypatch.setattr(service, "_detect_faces", lambda _img: [_fake_face()])

    result = service.run_face_scan_static(_tiny_face_image(), "selfie.jpg")

    assert result["filename"] == "selfie.jpg"
    assert result["scan_type"] == "face_scan"
    assert result["mode"] == "static"
    assert result["schema_version"] == service.FACE_SCAN_SCHEMA_VERSION
    assert result["verdict"] in {"GENUINE", "SUSPICIOUS", "INCONCLUSIVE", "DEEPFAKE"}
    assert isinstance(result["risk_score_0_100"], float)
    assert isinstance(result["confidence_0_100"], float)
    assert isinstance(result["evidence"], list)
    assert result["checks"]["active_liveness"]["status"] == "not_run"
    assert result["checks"]["face_detection"]["face_count"] == 1


def test_run_face_scan_static_is_deterministic(monkeypatch) -> None:
    from basetruth.face_scan import service

    monkeypatch.setattr(service, "_detect_faces", lambda _img: [_fake_face()])
    image = _tiny_face_image()

    first = service.run_face_scan_static(image, "selfie.jpg")
    second = service.run_face_scan_static(image, "selfie.jpg")

    assert first["verdict"] == second["verdict"]
    assert first["risk_score_0_100"] == second["risk_score_0_100"]
    assert first["confidence_0_100"] == second["confidence_0_100"]
    assert first["checks"]["photo_authenticity"] == second["checks"]["photo_authenticity"]


def test_run_face_scan_static_returns_inconclusive_for_low_quality(monkeypatch) -> None:
    from basetruth.face_scan import service

    monkeypatch.setattr(service, "_detect_faces", lambda _img: [_fake_face()])

    result = service.run_face_scan_static(_tiny_face_image(fill_value=0), "dark.jpg")

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["confidence_0_100"] < 35.0
    assert "confidence" in result["confidence_reason"].lower() or "lighting" in result["confidence_reason"].lower()