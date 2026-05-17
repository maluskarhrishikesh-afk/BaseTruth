from __future__ import annotations

from basetruth.ui.pages.video_kyc import (
    _normalize_vkyc_result_payload,
    _normalize_external_api_url,
    _prepare_vkyc_face_scan_result_for_display,
)


def test_normalize_external_api_url_rewrites_localhost() -> None:
    assert _normalize_external_api_url("http://localhost:8000") == "http://127.0.0.1:8000"


def test_normalize_external_api_url_rewrites_ipv6_loopback() -> None:
    assert _normalize_external_api_url("http://[::1]:8000") == "http://127.0.0.1:8000"


def test_normalize_external_api_url_preserves_real_host() -> None:
    assert _normalize_external_api_url("https://demo.basetruth.ai") == "https://demo.basetruth.ai"


def test_normalize_vkyc_result_payload_handles_none() -> None:
    assert _normalize_vkyc_result_payload(None) == {}


def test_prepare_vkyc_face_scan_result_for_display_backfills_filename() -> None:
    payload = {
        "face_scan_session_id": "demo123",
        "video_key": "face-scan-video/demo123.mp4",
    }

    prepared = _prepare_vkyc_face_scan_result_for_display(payload)

    assert prepared["filename"] == "face_scan_live_demo123.jpg"
    assert prepared["video_key"] == "face-scan-video/demo123.mp4"
    assert "filename" not in payload


def test_prepare_vkyc_face_scan_result_for_display_preserves_existing_filename() -> None:
    payload = {
        "face_scan_session_id": "demo123",
        "filename": "face_scan_live_existing.jpg",
    }

    prepared = _prepare_vkyc_face_scan_result_for_display(payload)

    assert prepared["filename"] == "face_scan_live_existing.jpg"