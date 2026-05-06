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


def test_signal_agreement_raises_confidence_vs_disagreement(monkeypatch) -> None:
    """Confidence is higher when sub-signals agree than when they contradict each other.

    Two scans on the same image, same quality — but one has both signals pointing high
    (both presentation and synthetic risk high) and the other has them pointing in
    opposite directions. The agreeing case should produce higher confidence.
    """
    from basetruth.face_scan import service

    monkeypatch.setattr(service, "_detect_faces", lambda _img: [_fake_face()])
    image = _tiny_face_image()

    # Both signals agree: presentation and synthetic both high (clear fraud signal)
    monkeypatch.setattr(service, "_edge_halo_score", lambda _: 85.0)
    monkeypatch.setattr(service, "_compression_residual_score", lambda _: 80.0)
    monkeypatch.setattr(service, "_landmark_asymmetry_score", lambda _face, _h: 78.0)
    result_agree = service.run_face_scan_static(image, "test_agree.jpg")

    # Signals disagree: presentation high, synthetic low (conflicting evidence)
    monkeypatch.setattr(service, "_edge_halo_score", lambda _: 85.0)
    monkeypatch.setattr(service, "_compression_residual_score", lambda _: 80.0)
    monkeypatch.setattr(service, "_landmark_asymmetry_score", lambda _face, _h: 5.0)
    result_disagree = service.run_face_scan_static(image, "test_disagree.jpg")

    assert result_agree["confidence_0_100"] > result_disagree["confidence_0_100"], (
        f"Agreeing signals should yield higher confidence: "
        f"agree={result_agree['confidence_0_100']}, disagree={result_disagree['confidence_0_100']}"
    )


# ── Narrative (Gemma4 honest_review enrichment) ───────────────────────────────

def test_generate_face_scan_narrative_uses_llm_when_available(monkeypatch) -> None:
    """When the LLM returns a non-empty response, honest_review is replaced and
    narrative_source carries the model name."""
    from basetruth.face_scan import narrative
    import basetruth.integrations.ollama as _ollama_mod

    def _fake_route(system_prompt, user_prompt, image_bytes_list, feature=None, **_kwargs):
        return ("This face scan looks genuine. All checks passed. No action required.", "gemma4_ollama", "gemma4:e2b", "")

    monkeypatch.setattr(_ollama_mod, "_route_vlm_chat", _fake_route)

    result = {
        "mode": "static",
        "verdict": "GENUINE",
        "risk_score_0_100": 12.0,
        "confidence_0_100": 91.0,
        "honest_review": "Original rule-based review.",
        "evidence": ["One face detected.", "No replay signals."],
        "checks": {},
    }
    review, source = narrative.generate_face_scan_narrative(result)

    assert review == "This face scan looks genuine. All checks passed. No action required."
    assert source == "gemma4 (gemma4:e2b)"


def test_generate_face_scan_narrative_falls_back_when_llm_fails(monkeypatch) -> None:
    """When the LLM raises an exception, honest_review falls back to the rule-based text."""
    from basetruth.face_scan import narrative
    import basetruth.integrations.ollama as _ollama_mod

    def _boom(*_args, **_kwargs):
        raise ConnectionError("Ollama is offline")

    monkeypatch.setattr(_ollama_mod, "_route_vlm_chat", _boom)

    result = {
        "mode": "live",
        "verdict": "SUSPICIOUS",
        "risk_score_0_100": 45.0,
        "confidence_0_100": 72.0,
        "honest_review": "Rule-based fallback text.",
        "evidence": ["Moderate replay signals."],
        "checks": {},
    }
    review, source = narrative.generate_face_scan_narrative(result)

    assert review == "Rule-based fallback text."
    assert source == "rule_based"


def test_run_face_scan_static_carries_narrative_source(monkeypatch) -> None:
    """run_face_scan_static must include narrative_source in the returned dict."""
    from basetruth.face_scan import service, narrative

    monkeypatch.setattr(service, "_detect_faces", lambda _img: [_fake_face()])
    # Patch generate_face_scan_narrative on the narrative module (service imports it as _narrative_mod)
    monkeypatch.setattr(
        narrative, "generate_face_scan_narrative",
        lambda _result: ("Fake LLM review.", "gemma4 (test-model)"),
    )

    result = service.run_face_scan_static(_tiny_face_image(), "test.jpg")

    assert "narrative_source" in result, "narrative_source must be present in static scan result"
    assert result["honest_review"] == "Fake LLM review."
    assert result["narrative_source"] == "gemma4 (test-model)"