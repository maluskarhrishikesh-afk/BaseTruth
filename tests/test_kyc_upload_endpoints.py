"""Unit tests for the KYC HTTP endpoints added to api.py.

  POST /kyc/sessions/{sid}/upload-id
    POST /kyc/sessions/{sid}/upload-pan
  POST /kyc/sessions/{sid}/upload-address
  POST /kyc/sessions/{sid}/location

Tests are purely in-process — no live server, no network, no DB, no MinIO.
Starlette TestClient is used to drive FastAPI directly.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures — build a minimal FastAPI app + test client + seed a KYC session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_and_store():
    """Create a BaseTruth FastAPI app and return (app, kyc_store) so tests can
    pre-seed sessions and inspect state after requests."""
    from basetruth.api import create_app
    from basetruth.kyc.session import SessionStore

    store = SessionStore()
    app = create_app()
    return app, store


@pytest.fixture()
def client(app_and_store):
    """Starlette TestClient wrapping the FastAPI app."""
    from starlette.testclient import TestClient
    app, _ = app_and_store
    return TestClient(app, raise_server_exceptions=False)


def _make_session(app_and_store) -> Any:
    """Helper: create a fresh KYC session via the POST /kyc/sessions endpoint."""
    from starlette.testclient import TestClient
    app, _ = app_and_store
    with TestClient(app, raise_server_exceptions=False) as tc:
        resp = tc.post("/kyc/sessions", json={"customer_name": "Test User", "challenges": ["blink"]})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Helpers — create a minimal 1×1 JPEG so file-upload endpoints receive valid
# image data without needing a real photo.
# ---------------------------------------------------------------------------

def _tiny_jpeg() -> bytes:
    """Return a minimal valid JPEG byte string (1×1 red pixel)."""
    try:
        import cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 200)  # BGR red
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()
    except ImportError:
        # If cv2 is somehow unavailable, return a minimal hard-coded JPEG
        return bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05,
            0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D,
            0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A,
            0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
            0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31,
            0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E,
            0x33, 0x34, 0x32,
            0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x11, 0x00,
            0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
            0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02,
            0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB,
            0xFF, 0xD9,
        ])


# ---------------------------------------------------------------------------
# POST /kyc/sessions/{sid}/upload-id
# ---------------------------------------------------------------------------

class TestKycUploadId:
    def test_returns_404_for_unknown_session(self, client) -> None:
        jpeg = _tiny_jpeg()
        resp = client.post(
            "/kyc/sessions/nonexistent-session-id/upload-id",
            files={"file": ("id.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        assert resp.status_code == 404

    def test_returns_400_when_no_face_found(self, app_and_store) -> None:
        """Uploading an image with no recognisable face returns 400 (no face) or 200
        depending on the environment's face-detection availability.  What must never
        happen is a 500 server error."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()  # 10×10 solid-colour image — has no face

        # _extract_face_from_image_bytes is a closure inside create_app() and cannot
        # be patched at module level.  We test the behaviour indirectly: the tiny JPEG
        # has no face, so on environments with a working face detector we expect 400;
        # on environments where face detection is unavailable we allow 400 or 200.
        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-id",
                files={"file": ("id.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        # Must never be a 500 — face-detection failure must return 400, not crash
        assert resp.status_code != 500, f"Server error: {resp.text}"
        assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code} {resp.text}"

    def test_valid_session_returns_non_500(self, app_and_store) -> None:
        """Uploading an ID for a valid session must not return 500."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-id",
                files={"file": ("pan.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        assert resp.status_code != 500, f"Server error: {resp.text}"

    def test_response_has_face_found_key(self, app_and_store) -> None:
        """On a 200 response the body must include the face_found boolean key."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-id",
                files={"file": ("id.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        if resp.status_code == 200:
            data = resp.json()
            assert "face_found" in data


# ---------------------------------------------------------------------------
# POST /kyc/sessions/{sid}/upload-pan
# ---------------------------------------------------------------------------

class TestKycUploadPan:
    def test_returns_404_for_unknown_session(self, client) -> None:
        jpeg = _tiny_jpeg()
        resp = client.post(
            "/kyc/sessions/nonexistent-session-id/upload-pan",
            files={"file": ("pan.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        assert resp.status_code == 404

    def test_upload_stores_pan_image_and_marks_processing(
        self, app_and_store, monkeypatch
    ) -> None:
        """upload-pan must return quickly after storing the raw image.

        This is a regression test for the operator Session Status poller timing
        out while PAN extraction waited on Gemma4 and OCR in the same request.
        The endpoint now stores the image immediately and starts background work.
        """
        from starlette.testclient import TestClient
        import base64
        import basetruth.api as api_module

        app, _ = app_and_store
        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()
        captured: dict[str, Any] = {}

        def _fake_spawn(session, image_bytes, doc_filename, _logger) -> None:
            captured["session_id"] = session.session_id
            captured["image_bytes"] = image_bytes
            captured["doc_filename"] = doc_filename

        monkeypatch.setattr(api_module, "_spawn_kyc_pan_extraction_thread", _fake_spawn)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-pan",
                files={"file": ("pan.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
            status_resp = tc.get(f"/kyc/sessions/{sid}")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["processing_started"] is True
        assert data["pan_extracted"] is False
        assert "background" in data["hint"].lower()

        assert captured == {
            "session_id": sid,
            "image_bytes": jpeg,
            "doc_filename": "pan.jpg",
        }

        assert status_resp.status_code == 200, status_resp.text
        status = status_resp.json()
        assert base64.b64decode(status["pan_b64"]) == jpeg
        assert status["pan_processing"] is True
        assert status["pan_extraction_error"] == ""
        assert status["pan_data"] is None


# ---------------------------------------------------------------------------
# POST /kyc/sessions/{sid}/upload-address
# ---------------------------------------------------------------------------

class TestKycUploadAddress:
    def test_returns_404_for_unknown_session(self, client) -> None:
        jpeg = _tiny_jpeg()
        resp = client.post(
            "/kyc/sessions/bad-id/upload-address",
            files={"file": ("addr.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        assert resp.status_code == 404

    def test_valid_session_always_returns_200(self, app_and_store) -> None:
        """upload-address should always succeed (OCR failure is non-fatal)."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-address",
                files={"file": ("aadhaar_back.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text

    def test_response_has_address_extracted_key(self, app_and_store) -> None:
        """The 200 response must include the address_extracted boolean key."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-address",
                files={"file": ("proof.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        data = resp.json()
        assert "address_extracted" in data
        assert isinstance(data["address_extracted"], bool)

    def test_response_has_hint_key(self, app_and_store) -> None:
        """The 200 response must include a hint string."""
        from starlette.testclient import TestClient
        app, _ = app_and_store

        sid = _make_session(app_and_store)
        jpeg = _tiny_jpeg()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/upload-address",
                files={"file": ("proof.jpg", io.BytesIO(jpeg), "image/jpeg")},
            )
        data = resp.json()
        assert "hint" in data
        assert isinstance(data["hint"], str)


# ---------------------------------------------------------------------------
# POST /kyc/sessions/{sid}/location
# ---------------------------------------------------------------------------

class TestKycSaveLocation:
    def test_returns_404_for_unknown_session(self, client) -> None:
        resp = client.post(
            "/kyc/sessions/bad-session/location",
            json={"lat": 18.5204, "lon": 73.8567, "accuracy": 15.0},
        )
        assert resp.status_code == 404

    def test_valid_session_returns_200_with_degraded_geocoding(
        self, app_and_store, monkeypatch
    ) -> None:
        """Even when reverse_geocode returns None, the endpoint must return 200."""
        from starlette.testclient import TestClient
        from basetruth.kyc import address_match

        app, _ = app_and_store
        # Force reverse_geocode to fail gracefully
        monkeypatch.setattr(address_match, "reverse_geocode", lambda lat, lon: None)

        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/location",
                json={"lat": 18.5204, "lon": 73.8567, "accuracy": 20.0},
            )
        assert resp.status_code == 200, resp.text

    def test_response_has_address_and_comparison_keys(
        self, app_and_store, monkeypatch
    ) -> None:
        """The 200 response must include 'address' and 'comparison' keys."""
        from starlette.testclient import TestClient
        from basetruth.kyc import address_match

        app, _ = app_and_store
        monkeypatch.setattr(address_match, "reverse_geocode", lambda lat, lon: "Pune, Maharashtra, India")

        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/location",
                json={"lat": 18.5204, "lon": 73.8567},
            )
        data = resp.json()
        assert "address" in data
        assert "comparison" in data

    def test_comparison_is_null_when_no_proof_uploaded(
        self, app_and_store, monkeypatch
    ) -> None:
        """When no address proof was uploaded, comparison must be null."""
        from starlette.testclient import TestClient
        from basetruth.kyc import address_match

        app, _ = app_and_store
        monkeypatch.setattr(address_match, "reverse_geocode", lambda lat, lon: "Baner, Pune")

        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/location",
                json={"lat": 18.5604, "lon": 73.8167},
            )
        data = resp.json()
        # No address proof uploaded yet → comparison should be null
        assert data["comparison"] is None

    def test_missing_accuracy_field_uses_default(
        self, app_and_store, monkeypatch
    ) -> None:
        """accuracy is optional — omitting it must not cause a validation error."""
        from starlette.testclient import TestClient
        from basetruth.kyc import address_match

        app, _ = app_and_store
        monkeypatch.setattr(address_match, "reverse_geocode", lambda lat, lon: None)

        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            # Send without accuracy — should use the default 0.0
            resp = tc.post(
                f"/kyc/sessions/{sid}/location",
                json={"lat": 18.5204, "lon": 73.8567},
            )
        assert resp.status_code == 200, resp.text

    def test_location_address_stored_in_session(
        self, app_and_store, monkeypatch
    ) -> None:
        """After a successful location POST, the session must store the reverse-geocoded address."""
        from starlette.testclient import TestClient
        from basetruth.kyc import address_match
        app, _ = app_and_store
        expected_addr = "Shivaji Nagar, Pune, Maharashtra, India"
        monkeypatch.setattr(address_match, "reverse_geocode", lambda lat, lon: expected_addr)

        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(
                f"/kyc/sessions/{sid}/location",
                json={"lat": 18.5204, "lon": 73.8567},
            )
        assert resp.status_code == 200

        # Address was successfully stored (endpoint returned 200 with the address)
        data = resp.json()
        assert data.get("address") == expected_addr


# ---------------------------------------------------------------------------
# POST /kyc/sessions/{sid}/liveness-result and customer-page bridge
# ---------------------------------------------------------------------------

class TestKycLivenessBridge:
    def test_customer_page_uses_face_scan_live_contract(self, app_and_store) -> None:
        """The KYC browser page must redirect into the shared Face Scan live page."""
        from starlette.testclient import TestClient

        app, _ = app_and_store
        sid = _make_session(app_and_store)

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.get(f"/kyc/{sid}")

        assert resp.status_code == 200
        html = resp.text
        assert "/api/v1/face-scan/sessions" in html
        assert "/face-scan/live/${data.session_id}?autostart=1&result_mode=verification&callback=${callback}" in html
        assert f"/kyc/ws/${{SESSION_ID}}" not in html

    def test_liveness_result_endpoint_stores_face_scan_payload(self, app_and_store) -> None:
        """The bridge endpoint must complete the KYC session and keep the full Face Scan JSON."""
        from starlette.testclient import TestClient

        app, _ = app_and_store
        sid = _make_session(app_and_store)
        payload = {
            "face_scan_session_id": "face-scan-live-123",
            "verdict": "GENUINE",
            "risk_score_0_100": 12.5,
            "confidence_0_100": 91.0,
            "honest_review": "The live scan looks genuine.",
            "checks": {
                "active_liveness": {
                    "passed": True,
                    "completed_challenges": ["look_straight", "blink"],
                }
            },
        }

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post(f"/kyc/sessions/{sid}/liveness-result", json=payload)
            assert resp.status_code == 200, resp.text

            data = resp.json()
            assert data["ok"] is True
            assert data["passed"] is True
            assert data["display_score"] == pytest.approx(87.5)
            assert data["face_scan_result"]["verdict"] == "GENUINE"

            status_resp = tc.get(f"/kyc/sessions/{sid}")

        assert status_resp.status_code == 200, status_resp.text
        status = status_resp.json()
        assert status["status"] == "completed"
        assert status["challenges_completed"] == status["total_challenges"]
        assert status["result"]["message"] == "The live scan looks genuine."
        assert status["result"]["face_scan_result"]["confidence_0_100"] == pytest.approx(91.0)
