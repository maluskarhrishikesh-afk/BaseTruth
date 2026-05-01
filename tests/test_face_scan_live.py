from __future__ import annotations

from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

_CV2_IMENCODE = getattr(__import__("cv2"), "imencode")


def _frame_hashes() -> List[str]:
    return [
        "0f0f0f0f0f0f0f0f",
        "f0f0f0f0f0f0f0f0",
        "00ff00ff00ff00ff",
        "ff00ff00ff00ff00",
        "3333cccc3333cccc",
        "cccc3333cccc3333",
    ]


def _metric_frame(index: int, yaw: float, pitch: float, nose_rel_x: float, frame_hash: str, brightness: float = 128.0) -> Dict[str, Any]:
    return {
        "frame_index": index,
        "yaw": yaw,
        "pitch": pitch,
        "nose_rel_x": nose_rel_x,
        "ear": 0.28,
        "face_center_dist": 0.04,
        "laplacian_var": 160.0,
        "brightness_mean": brightness,
        "edge_density": 0.21,
        "bbox_area_ratio": 0.19,
        "frame_hash": frame_hash,
        "detector_confidence": 0.98,
        "face_box": [12, 16, 116, 152],
    }


def _completed_session(history: List[Dict[str, Any]]):
    from basetruth.face_scan import live

    session = live.FaceScanLiveSession(session_id="session-live-1", challenges=["look_straight", "blink", "nod"])
    session.current_challenge_idx = len(session.challenges)
    session.challenge_results = [
        {"index": 0, "challenge": "look_straight", "passed": True},
        {"index": 1, "challenge": "blink", "passed": True},
        {"index": 2, "challenge": "nod", "passed": True},
    ]
    session.all_frame_history = history
    session.frames_received = len(history)
    session.best_live_frame_bytes = b"jpeg-bytes"
    session.last_face_box = history[-1]["face_box"]
    session.environment["observed_fps"] = 5.0
    session.environment["camera_resolution"] = [640, 480]
    return session


def test_build_live_face_scan_result_marks_stable_session_genuine() -> None:
    from basetruth.face_scan import live

    hashes = _frame_hashes()
    history = [
        _metric_frame(idx, yaw=0.02 * idx, pitch=0.01 * idx, nose_rel_x=0.03 * idx, frame_hash=hashes[idx % len(hashes)], brightness=126.0 + idx)
        for idx in range(12)
    ]
    session = _completed_session(history)

    result = live.build_live_face_scan_result(session)

    assert result["mode"] == "live"
    assert result["scan_type"] == "face_scan"
    assert result["verdict"] == "GENUINE"
    assert result["checks"]["replay_heuristics"]["score_0_100"] < 35.0
    assert result["checks"]["temporal_consistency"]["score_0_100"] < 35.0
    assert result["checks"]["active_liveness"]["passed"] is True


def test_build_live_face_scan_result_flags_replay_like_pattern() -> None:
    from basetruth.face_scan import live

    history = [
        _metric_frame(idx, yaw=0.03 * idx, pitch=0.0, nose_rel_x=0.02 * idx, frame_hash="aaaaaaaaaaaaaaaa", brightness=70.0 if idx % 2 == 0 else 180.0)
        for idx in range(12)
    ]
    session = _completed_session(history)

    result = live.build_live_face_scan_result(session)

    assert result["checks"]["replay_heuristics"]["score_0_100"] > 80.0
    assert result["verdict"] == "DEEPFAKE"
    assert result["risk_score_0_100"] >= 50.0


def test_build_live_face_scan_result_does_not_flag_normal_challenge_motion_as_suspicious() -> None:
    from basetruth.face_scan import live

    hashes = _frame_hashes()
    history = []
    for idx, (yaw, pitch, nose_rel_x) in enumerate(
        [
            (0.00, 0.02, 0.50),
            (0.01, 0.03, 0.50),
            (0.00, 0.02, 0.49),
            (0.00, 0.16, 0.50),
            (0.00, 0.28, 0.50),
            (0.00, 0.13, 0.50),
            (-0.03, 0.02, 0.48),
            (-0.09, 0.02, 0.44),
            (-0.17, 0.02, 0.40),
            (0.03, 0.02, 0.52),
            (0.10, 0.02, 0.57),
            (0.18, 0.02, 0.62),
        ]
    ):
        history.append(
            _metric_frame(
                idx,
                yaw=yaw,
                pitch=pitch,
                nose_rel_x=nose_rel_x,
                frame_hash=hashes[idx % len(hashes)],
                brightness=124.0,
            )
        )

    session = _completed_session(history)
    session.challenge_results = [
        {"index": 0, "challenge": "look_straight", "passed": True},
        {"index": 1, "challenge": "nod", "passed": True},
        {"index": 2, "challenge": "turn_left", "passed": True},
        {"index": 3, "challenge": "turn_right", "passed": True},
    ]
    session.challenges = ["look_straight", "nod", "turn_left", "turn_right"]
    session.current_challenge_idx = len(session.challenges)
    session.challenge_frame_history = {
        "ch_0": history[0:3],
        "ch_1": history[3:6],
        "ch_2": history[6:9],
        "ch_3": history[9:12],
    }

    result = live.build_live_face_scan_result(session)

    assert result["checks"]["temporal_consistency"]["score_0_100"] < 35.0
    assert result["verdict"] == "GENUINE"


def test_face_scan_live_session_status_contract_is_not_kyc() -> None:
    from basetruth.api import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    create_resp = client.post("/api/v1/face-scan/sessions", json={"challenges": ["blink", "nod"]})

    assert create_resp.status_code == 200, create_resp.text
    created = create_resp.json()
    assert created["session_url"].startswith("/face-scan/live/")
    assert created["challenges"] == ["look_straight", "blink", "nod"]

    status_resp = client.get(f"/api/v1/face-scan/sessions/{created['session_id']}")
    assert status_resp.status_code == 200, status_resp.text
    status = status_resp.json()
    assert status["current_challenge"] == "look_straight"
    assert "aadhaar_qr" not in status
    assert "address_dtls" not in status
    assert "pan_data" not in status


def test_face_scan_live_websocket_returns_live_result(monkeypatch) -> None:
    from basetruth.api import create_app
    from basetruth.face_scan import live

    def _fake_process(session, _frame: str) -> Dict[str, Any]:
        payload = {
            "filename": f"face_scan_live_{session.session_id}.jpg",
            "scan_type": "face_scan",
            "mode": "live",
            "schema_version": "1.0.0",
            "verdict": "GENUINE",
            "risk_score_0_100": 14.0,
            "confidence_0_100": 91.0,
            "confidence_reason": "Stable frames.",
            "overall_explanation": "Low replay and temporal-risk signals.",
            "honest_review": "The live session looks genuine.",
            "evidence": ["Completed live challenges: look_straight, blink, nod."],
            "trace": {"decision_trace_id": "fs_live_test", "processing_time_ms": 18, "rules_version": "face-scan-rules-1.0.0", "model_version": "heuristics-only", "timestamp_utc": "2026-05-01T00:00:00Z"},
            "environment": {"platform": "web", "observed_fps": 5.0, "virtual_camera_suspected": False},
            "checks": {
                "face_detection": {"status": "pass", "face_count": 1},
                "temporal_consistency": {"status": "pass", "score_0_100": 12.0},
                "replay_heuristics": {"status": "pass", "score_0_100": 9.0},
                "active_liveness": {"status": "pass", "passed": True, "completed_challenges": ["look_straight", "blink", "nod"], "challenge_count": 3, "best_frame_available": True},
            },
            "artifacts": {"best_frame_available": True, "challenge_snapshots_available": True},
        }
        session.status = "completed"
        session.result = payload
        return {"type": "result", **payload}

    monkeypatch.setattr(live, "process_live_frame_message", _fake_process)

    client = TestClient(create_app(), raise_server_exceptions=False)
    create_resp = client.post("/api/v1/face-scan/sessions", json={"challenges": ["blink", "nod"]})
    sid = create_resp.json()["session_id"]

    with client.websocket_connect(f"/api/v1/face-scan/ws/{sid}") as ws:
        ws.send_json({"type": "meta", "camera_width": 640, "camera_height": 480, "observed_fps": 5.0, "platform": "pytest"})
        ws.send_json({"type": "frame", "data": "ZmFrZQ=="})
        result = ws.receive_json()

    assert result["type"] == "result"
    assert result["mode"] == "live"
    assert result["verdict"] == "GENUINE"

    status_resp = client.get(f"/api/v1/face-scan/sessions/{sid}")
    status = status_resp.json()
    assert status["result"]["mode"] == "live"
    assert status["result"]["checks"]["replay_heuristics"]["score_0_100"] == 9.0


def test_face_scan_live_websocket_disconnect_keeps_session_resumable() -> None:
    from basetruth.api import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    create_resp = client.post("/api/v1/face-scan/sessions", json={"challenges": ["blink", "nod"]})
    sid = create_resp.json()["session_id"]

    with client.websocket_connect(f"/api/v1/face-scan/ws/{sid}") as ws:
        ws.send_json({"type": "meta", "camera_width": 640, "camera_height": 480, "observed_fps": 5.0, "platform": "pytest"})

    status_resp = client.get(f"/api/v1/face-scan/sessions/{sid}")
    status = status_resp.json()
    assert status["status"] == "waiting"
    assert status["current_challenge"] == "look_straight"


def test_face_scan_live_websocket_processing_error_keeps_session_resumable(monkeypatch) -> None:
    from basetruth.api import create_app
    from basetruth.face_scan import live

    def _boom(_session, _frame: str) -> Dict[str, Any]:
        raise RuntimeError("unexpected frame failure")

    monkeypatch.setattr(live, "process_live_frame_message", _boom)

    client = TestClient(create_app(), raise_server_exceptions=False)
    create_resp = client.post("/api/v1/face-scan/sessions", json={"challenges": ["blink", "nod"]})
    sid = create_resp.json()["session_id"]

    with client.websocket_connect(f"/api/v1/face-scan/ws/{sid}") as ws:
        ws.send_json({"type": "meta", "camera_width": 640, "camera_height": 480, "observed_fps": 5.0, "platform": "pytest"})
        ws.send_json({"type": "frame", "data": "ZmFrZQ=="})
        with pytest.raises(Exception):
            ws.receive_json()

    status_resp = client.get(f"/api/v1/face-scan/sessions/{sid}")
    status = status_resp.json()
    assert status["status"] == "waiting"
    assert status["current_challenge"] == "look_straight"


def test_process_live_frame_message_recovers_from_detector_error(monkeypatch) -> None:
    import base64
    import numpy as np

    from basetruth.face_scan import live

    img = np.full((48, 48, 3), 180, dtype=np.uint8)
    ok, encoded = _CV2_IMENCODE(".jpg", img)
    assert ok

    session = live.FaceScanLiveSession(session_id="session-live-2", challenges=["look_straight", "blink"])
    payload = base64.b64encode(encoded.tobytes()).decode()

    monkeypatch.setattr(live, "_detect_faces", lambda _img: (_ for _ in ()).throw(RuntimeError("detector crashed")))

    result = live.process_live_frame_message(session, payload)

    assert result["type"] == "status"
    assert result["face_detected"] is False
    assert "temporary issue" in result["feedback"].lower()
    assert session.status == "waiting"
    assert session.frames_without_face == 1