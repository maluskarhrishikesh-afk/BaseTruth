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
        # New liveness signal fields — defaults represent a healthy, organic live session
        "left_eye_x_norm":  0.28 + 0.003 * (index % 3),   # slight natural jitter
        "left_eye_y_norm":  0.42 + 0.002 * (index % 5),
        "right_eye_x_norm": 0.72 + 0.003 * (index % 4),
        "right_eye_y_norm": 0.41 + 0.002 * (index % 3),
        "interocular_px_norm": 0.44,
        "server_recv_mono": 10.0 + index * 0.125 + (index % 3) * 0.018,  # organic jitter
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

    monkeypatch.setattr(live, "_detect_faces", lambda _img, **_kw: (_ for _ in ()).throw(RuntimeError("detector crashed")))

    result = live.process_live_frame_message(session, payload)

    assert result["type"] == "status"
    assert result["face_detected"] is False
    assert "temporary issue" in result["feedback"].lower()
    assert session.status == "waiting"
    assert session.frames_without_face == 1


# ── Signal 1: Saccade / Eye Micro-Jitter ──────────────────────────────────────

def test_saccade_analysis_passes_organic_eye_jitter() -> None:
    """Frames with natural, low-amplitude eye variance should produce a low risk score."""
    from basetruth.face_scan import live

    # Simulate 10 frames with small, varied eye micro-movements (organic jitter)
    history = []
    import math
    for i in range(10):
        f = _metric_frame(i, yaw=0.01 * i, pitch=0.0, nose_rel_x=0.50, frame_hash=_frame_hashes()[i % len(_frame_hashes())])
        # Organic jitter: sinusoidal at amplitude 0.010 — well above the stillness
        # detection threshold so the function reliably returns low risk
        f["left_eye_x_norm"]  = 0.28 + 0.010 * math.sin(i * 1.3)
        f["left_eye_y_norm"]  = 0.42 + 0.009 * math.sin(i * 0.9)
        f["right_eye_x_norm"] = 0.72 + 0.010 * math.sin(i * 1.1)
        f["right_eye_y_norm"] = 0.41 + 0.009 * math.sin(i * 0.7)
        history.append(f)

    result = live._compute_saccade_analysis(history)

    assert result["score_0_100"] < 50.0, "Organic jitter should not be flagged as suspicious"
    assert result["mean_eye_jitter"] > 0.0


def test_saccade_analysis_flags_frozen_eyes() -> None:
    """Eye positions that never change across frames indicate a static photo replay."""
    from basetruth.face_scan import live

    history = []
    for i in range(10):
        f = _metric_frame(i, yaw=0.01 * i, pitch=0.0, nose_rel_x=0.50, frame_hash=_frame_hashes()[i % len(_frame_hashes())])
        # Perfectly frozen eye positions — no micro-jitter at all (photo / static render)
        f["left_eye_x_norm"]  = 0.280
        f["left_eye_y_norm"]  = 0.420
        f["right_eye_x_norm"] = 0.720
        f["right_eye_y_norm"] = 0.410
        history.append(f)

    result = live._compute_saccade_analysis(history)

    assert result["score_0_100"] >= 60.0, "Frozen eyes must be flagged with high risk"
    assert result["eye_stillness_risk"] >= 60.0


def test_saccade_analysis_returns_neutral_for_insufficient_frames() -> None:
    """Fewer than 6 frames with eye data should return a non-zero neutral score."""
    from basetruth.face_scan import live

    history = [_metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[0]) for i in range(3)]
    # Remove eye fields so the function treats them as missing
    for f in history:
        f.pop("left_eye_x_norm", None)
        f.pop("right_eye_x_norm", None)

    result = live._compute_saccade_analysis(history)

    assert result["score_0_100"] == 20.0  # neutral sentinel
    assert result["mean_eye_jitter"] == 0.0


# ── Signal 2: FFT Screen-Frequency Analysis ───────────────────────────────────

def test_screen_frequency_analysis_low_risk_for_organic_peaks() -> None:
    """Mean FFT peak concentration below 0.20 (organic texture) should produce zero risk."""
    from basetruth.face_scan import live

    # 0.12 represents a real face with diffuse mid-frequency energy (well below 0.20 threshold)
    history = [
        {**_metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[i % len(_frame_hashes())]), "fft_grid_peak_ratio": 0.12}
        for i in range(8)
    ]
    result = live._compute_screen_frequency_analysis(history)

    assert result["score_0_100"] == 0.0
    assert abs(result["mean_fft_grid_peak"] - 0.12) < 0.001


def test_screen_frequency_analysis_high_risk_for_screen_peaks() -> None:
    """Mean FFT peak concentration above 0.40 (Moiré screen grid) should produce near-maximum risk."""
    from basetruth.face_scan import live

    # 0.55 represents a filmed screen with concentrated spectral peaks (above 0.40 high threshold)
    history = [
        {**_metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[i % len(_frame_hashes())]), "fft_grid_peak_ratio": 0.55}
        for i in range(8)
    ]
    result = live._compute_screen_frequency_analysis(history)

    assert result["score_0_100"] >= 90.0


def test_screen_frequency_analysis_neutral_when_no_fft_data() -> None:
    """Sessions without any fft_grid_peak_ratio key return zero risk (neutral)."""
    from basetruth.face_scan import live

    history = [_metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[0]) for i in range(6)]
    result = live._compute_screen_frequency_analysis(history)

    assert result["score_0_100"] == 0.0
    assert result["mean_fft_grid_peak"] == 0.0


# ── Signal 3: Frame Timing Jitter ────────────────────────────────────────────

def test_frame_timing_jitter_flags_uniform_delivery() -> None:
    """Perfectly uniform inter-frame intervals (injection tool) must produce high risk."""
    from basetruth.face_scan import live

    # Simulate 10 frames arriving at exactly 125 ms intervals — no variance
    history = []
    for i in range(10):
        f = _metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[i % len(_frame_hashes())])
        f["server_recv_mono"] = 0.0 + i * 0.125  # perfectly uniform
        history.append(f)

    result = live._compute_frame_timing_jitter(history)

    assert result["score_0_100"] >= 60.0, "Perfectly uniform delivery must be flagged"
    assert result["interval_cv"] < 0.05


def test_frame_timing_jitter_passes_organic_variance() -> None:
    """Frames with natural timing variance (real browser) should produce low risk."""
    from basetruth.face_scan import live

    import math
    history = []
    t = 0.0
    for i in range(12):
        f = _metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[i % len(_frame_hashes())])
        # Organic jitter: vary intervals between ~80 ms and ~200 ms
        interval = 0.125 + 0.04 * math.sin(i * 2.1) + 0.025 * (i % 3)
        t += interval
        f["server_recv_mono"] = t
        history.append(f)

    result = live._compute_frame_timing_jitter(history)

    assert result["score_0_100"] < 50.0, "Organically varied timing must not be flagged"
    assert result["interval_cv"] >= 0.10


def test_frame_timing_jitter_neutral_for_insufficient_frames() -> None:
    """Fewer than 6 frames with timestamps should return a neutral zero score."""
    from basetruth.face_scan import live

    history = [_metric_frame(i, 0.0, 0.0, 0.50, _frame_hashes()[0]) for i in range(3)]
    for f in history:
        f.pop("server_recv_mono", None)

    result = live._compute_frame_timing_jitter(history)

    assert result["score_0_100"] == 0.0
    assert result["interval_cv"] == 0.0


# ── Signal 4: 3D Depth Consistency ───────────────────────────────────────────

def _turn_session_with_iod(iod_constant: bool) -> "live.FaceScanLiveSession":
    """Build a session that completed a turn_left challenge.

    When iod_constant=True, the interocular distance never decreases during the
    turn — simulating a flat 2D photo. When False, it decreases as expected from
    a 3D face (more yaw → lower iod).
    """
    from basetruth.face_scan import live

    session = live.FaceScanLiveSession(session_id="session-depth-1", challenges=["look_straight", "turn_left"])
    session.current_challenge_idx = 2
    session.challenge_results = [
        {"index": 0, "challenge": "look_straight", "passed": True},
        {"index": 1, "challenge": "turn_left",     "passed": True},
    ]

    turn_frames = []
    for i in range(8):
        # Yaw goes from 0.0 to -0.20 (turning left)
        yaw = -(i * 0.025)
        # 3D face: iod decreases as yaw grows; flat photo: iod stays at 0.44
        iod = 0.44 if iod_constant else 0.44 - abs(yaw) * 0.6
        turn_frames.append({
            "frame_index": i, "yaw": yaw, "pitch": 0.0, "nose_rel_x": 0.50 + yaw * 0.5,
            "interocular_px_norm": iod, "frame_hash": "abc", "brightness_mean": 128.0,
            "laplacian_var": 150.0, "bbox_area_ratio": 0.18, "detector_confidence": 0.97,
            "face_box": [10, 10, 110, 150], "edge_density": 0.2, "ear": 0.28,
        })

    session.challenge_frame_history = {"ch_0": [], "ch_1": turn_frames}
    session.all_frame_history = turn_frames
    session.frames_received = len(turn_frames)
    session.best_live_frame_bytes = b"jpeg"
    session.last_face_box = [10, 10, 110, 150]
    return session


def test_depth_consistency_passes_for_real_3d_face() -> None:
    """IOD that decreases during turns (3D face) should produce a near-zero risk score."""
    from basetruth.face_scan import live

    session = _turn_session_with_iod(iod_constant=False)
    result = live._compute_depth_consistency(session)

    assert result["score_0_100"] < 35.0, "3D face geometry must not be flagged"
    assert result["iod_yaw_correlation"] < 0.0, "Correlation must be negative for 3D face"


def test_depth_consistency_flags_flat_photo() -> None:
    """Constant IOD during head turns (flat photo) should produce a high risk score."""
    from basetruth.face_scan import live

    session = _turn_session_with_iod(iod_constant=True)
    result = live._compute_depth_consistency(session)

    assert result["score_0_100"] >= 35.0, "Flat photo geometry must be flagged"


def test_depth_consistency_skips_without_turn_challenges() -> None:
    """Sessions without turn challenges return a neutral zero score."""
    from basetruth.face_scan import live

    session = live.FaceScanLiveSession(session_id="session-depth-2", challenges=["look_straight", "blink"])
    session.current_challenge_idx = 2
    session.challenge_results = [
        {"index": 0, "challenge": "look_straight", "passed": True},
        {"index": 1, "challenge": "blink",          "passed": True},
    ]
    session.challenge_frame_history = {"ch_0": [], "ch_1": []}
    session.all_frame_history = []

    result = live._compute_depth_consistency(session)

    assert result["score_0_100"] == 0.0
    assert result["iod_yaw_correlation"] == 0.0


# ── Signal 5: Extended Virtual Camera Fingerprinting ─────────────────────────

def test_virtual_camera_extended_labels_are_flagged() -> None:
    """Device labels matching any extended token must set virtual_camera_suspected=True."""
    from basetruth.face_scan import live

    new_tokens = [
        "DroidCam Source",
        "EpocCam Virtual Camera",
        "iVCam Webcam",
        "XSplit VCam",
        "mmhmm Camera",
        "Iriun Webcam",
        "Camo (Reincubate)",
        "NDI Virtual Input",
        "Wirecast Virtual Camera",
        "Logitech Capture",
    ]
    for label in new_tokens:
        session = live.FaceScanLiveSession(session_id="virt-test", challenges=["blink"])
        live.handle_live_meta(session, {
            "camera_width": 640, "camera_height": 480, "observed_fps": 8.0,
            "user_agent": "Mozilla/5.0", "platform": "Win32", "device_label": label,
        })
        assert session.environment["virtual_camera_suspected"] is True, \
            f"Expected virtual_camera_suspected=True for device label: {label!r}"


def test_real_camera_label_is_not_flagged() -> None:
    """A generic camera label must NOT set virtual_camera_suspected."""
    from basetruth.face_scan import live

    session = live.FaceScanLiveSession(session_id="real-cam-test", challenges=["blink"])
    live.handle_live_meta(session, {
        "camera_width": 1280, "camera_height": 720, "observed_fps": 30.0,
        "user_agent": "Mozilla/5.0", "platform": "Win32",
        "device_label": "FaceTime HD Camera (Built-in)",
    })
    assert session.environment["virtual_camera_suspected"] is False


# ── Integration: result payload includes all new check keys ──────────────────

def test_build_live_result_includes_all_new_check_keys() -> None:
    """The result payload must include all four new signal check blocks."""
    from basetruth.face_scan import live

    hashes = _frame_hashes()
    history = [
        _metric_frame(idx, yaw=0.02 * idx, pitch=0.01 * idx, nose_rel_x=0.50, frame_hash=hashes[idx % len(hashes)])
        for idx in range(12)
    ]
    session = _completed_session(history)
    result = live.build_live_face_scan_result(session)

    checks = result["checks"]
    assert "saccade_analysis"  in checks, "saccade_analysis check block must be present"
    assert "screen_frequency"  in checks, "screen_frequency check block must be present"
    assert "frame_timing"      in checks, "frame_timing check block must be present"
    assert "depth_consistency" in checks, "depth_consistency check block must be present"

    for key in ("saccade_analysis", "screen_frequency", "frame_timing", "depth_consistency"):
        block = checks[key]
        assert "status"      in block, f"{key} must have a status field"
        assert "score_0_100" in block, f"{key} must have a score_0_100 field"
        assert block["status"] in ("pass", "review"), f"{key} status must be pass or review"


# ── Perceptual hash / difference hash ────────────────────────────────────────

def test_phash_and_dhash_produce_different_fingerprints_for_different_crops() -> None:
    """pHash and dHash of two structurally different images must differ significantly."""
    import numpy as np
    from basetruth.face_scan import live

    # pHash is frequency-based and brightness-invariant, so two uniform-colour images
    # (differing only in brightness) produce the same hash. Use structurally different
    # images (checkerboard vs gradient) to trigger genuine hash differences.
    checker = np.indices((32, 32)).sum(axis=0).astype(np.uint8) % 2 * 255  # checkerboard
    gradient = np.tile(np.linspace(0, 255, 32), (32, 1)).astype(np.uint8)   # left-to-right gradient

    ph_checker  = live._perceptual_hash(checker)
    ph_gradient = live._perceptual_hash(gradient)
    dh_checker  = live._difference_hash(checker)
    dh_gradient = live._difference_hash(gradient)

    # Structurally different images must produce different hashes for at least one type
    assert ph_checker != ph_gradient or dh_checker != dh_gradient, (
        "Structurally different images must produce different pHash or dHash"
    )
    assert isinstance(ph_checker, str) and len(ph_checker) > 0
    assert isinstance(dh_checker, str) and len(dh_checker) > 0


def test_is_repeat_frame_pair_uses_phash_when_available() -> None:
    """A pair of frames that share identical pHash should be flagged as a repeat."""
    from basetruth.face_scan import live

    # Build two frames with the same pHash (same image) but different aHash strings
    # to confirm the new logic picks up the pHash signal even when aHash differs.
    same_phash = "aaaa1111bbbbcccc"
    frame_a = _metric_frame(0, 0.0, 0.0, 0.50, frame_hash="0f0f0f0f0f0f0f0f")
    frame_b = _metric_frame(1, 0.0, 0.0, 0.50, frame_hash="f0f0f0f0f0f0f0f0")
    frame_a["frame_hash_phash"] = same_phash
    frame_b["frame_hash_phash"] = same_phash  # identical pHash → should be a repeat

    assert live._is_repeat_frame_pair(frame_a, frame_b) is True, (
        "Frames with identical pHash should be classified as a repeat pair"
    )


def test_is_repeat_frame_pair_legacy_fallback_uses_ahash_only() -> None:
    """When pHash/dHash fields are absent, falls back to aHash comparison."""
    from basetruth.face_scan import live

    # Both frames share the same aHash (no pHash fields present)
    same_ahash = "aaaaaaaaaaaaaaaa"
    frame_a = _metric_frame(0, 0.0, 0.0, 0.50, frame_hash=same_ahash)
    frame_b = _metric_frame(1, 0.0, 0.0, 0.50, frame_hash=same_ahash)
    # Confirm no new hash fields are present
    assert "frame_hash_phash" not in frame_a
    assert "frame_hash_phash" not in frame_b

    assert live._is_repeat_frame_pair(frame_a, frame_b) is True, (
        "Legacy frames with only aHash should still be caught as repeats"
    )


def test_is_repeat_frame_pair_different_frames_not_flagged() -> None:
    """Frames with clearly different hashes across all types should not be flagged."""
    from basetruth.face_scan import live

    frame_a = _metric_frame(0, 0.0, 0.0, 0.50, frame_hash="0f0f0f0f0f0f0f0f")
    frame_b = _metric_frame(1, 0.0, 0.0, 0.50, frame_hash="f0f0f0f0f0f0f0f0")
    frame_a["frame_hash_phash"] = "0000111122223333"
    frame_b["frame_hash_phash"] = "ffffeeeeddddcccc"
    frame_a["frame_hash_dhash"] = "aaaa0000ffff0000"
    frame_b["frame_hash_dhash"] = "0000ffffaaaa1111"

    assert live._is_repeat_frame_pair(frame_a, frame_b) is False, (
        "Clearly different frames should not be classified as a repeat pair"
    )


def test_build_live_face_scan_result_carries_narrative_source(monkeypatch) -> None:
    """build_live_face_scan_result must include narrative_source in its returned dict."""
    from basetruth.face_scan import live, narrative

    monkeypatch.setattr(
        narrative, "generate_face_scan_narrative",
        lambda _result: ("Fake LLM live review.", "gemma4 (test-model)"),
    )

    hashes = _frame_hashes()
    history = [
        _metric_frame(idx, yaw=0.02 * idx, pitch=0.01 * idx, nose_rel_x=0.50, frame_hash=hashes[idx % len(hashes)])
        for idx in range(12)
    ]
    session = _completed_session(history)
    result = live.build_live_face_scan_result(session)

    assert "narrative_source" in result, "narrative_source must be present in live scan result"
    assert result["honest_review"] == "Fake LLM live review."
    assert result["narrative_source"] == "gemma4 (test-model)"