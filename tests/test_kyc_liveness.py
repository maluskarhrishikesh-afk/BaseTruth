from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from basetruth.kyc.liveness import analyze_challenge, extract_features, run_face_match


def _make_face(
    *,
    kps=None,
    bbox=None,
    det_score=None,
    ear=None,
    normed_embedding=None,
):
    return SimpleNamespace(
        kps=np.array(kps, dtype=float) if kps is not None else None,
        bbox=np.array(bbox, dtype=float) if bbox is not None else np.array([0, 0, 1, 1], dtype=float),
        det_score=det_score,
        ear=ear,
        normed_embedding=normed_embedding,
    )


def test_extract_features_requires_face_keypoints() -> None:
    face = _make_face(kps=None)

    with pytest.raises(ValueError, match="Face keypoints not available"):
        extract_features(face)


def test_extract_features_normalizes_pose_and_defaults_optional_values() -> None:
    face = _make_face(
        kps=[[10, 20], [30, 20], [24, 32], [12, 40], [28, 40]],
        bbox=[0, 0, 40, 80],
        det_score=None,
        ear=None,
    )

    features = extract_features(face)

    assert features["nose_rel_x"] == pytest.approx(0.6)
    assert features["nose_rel_y"] == pytest.approx(0.4)
    assert features["yaw"] == pytest.approx(0.2)
    assert features["pitch"] == pytest.approx(0.6)
    assert features["det_score"] == pytest.approx(1.0)
    assert features["ear"] == pytest.approx(0.30)


def test_analyze_challenge_turn_left_passes_when_nose_moves_far_enough() -> None:
    # Mirrored frame: subject turns THEIR left → nose swings image-LEFT → yaw NEGATIVE
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.42, "yaw": -0.18, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.35, "yaw": -0.38, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


def test_analyze_challenge_turn_left_passes_for_modest_low_fps_turn() -> None:
    history = [
        {"nose_rel_x": 0.50, "yaw": 0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.46, "yaw": -0.09, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.41, "yaw": -0.16, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


def test_analyze_challenge_turn_left_does_not_pass_for_opposite_direction() -> None:
    history = [
        {"nose_rel_x": 0.50, "yaw": 0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.56, "yaw": 0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.61, "yaw": 0.19, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result["passed"] is False


def test_analyze_challenge_turn_left_passes_on_clear_relative_shift_from_starting_pose() -> None:
    history = [
        {"nose_rel_x": 0.54, "yaw": -0.02, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.52, "yaw": -0.04, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.49, "yaw": -0.07, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.45, "yaw": -0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.40, "yaw": -0.14, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


def test_analyze_challenge_turn_right_requests_more_movement_when_not_far_enough() -> None:
    # Mirrored frame: subject tries to turn THEIR right (yaw positive) but not far enough (< 0.20)
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.54, "yaw":  0.08, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.57, "yaw":  0.14, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_right")

    assert result["passed"] is False
    assert "Keep turning" in result["feedback"]


def test_analyze_challenge_turn_right_passes_on_clear_relative_shift_from_starting_pose() -> None:
    history = [
        {"nose_rel_x": 0.46, "yaw": 0.02, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.48, "yaw": 0.04, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.51, "yaw": 0.07, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.55, "yaw": 0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.60, "yaw": 0.14, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_right")

    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


def test_analyze_challenge_nod_passes_on_large_pitch_range() -> None:
    history = [
        {"nose_rel_x": 0.5, "pitch": 0.02, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.05, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.08, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.31, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.34, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.36, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "nod")

    assert result == {"passed": True, "feedback": "✅ Nod detected!"}


def test_analyze_challenge_blink_passes_with_ear_dip_and_recovery() -> None:
    history = [
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.31, "det_score": 0.95},
        {"ear": 0.29, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.07, "det_score": 0.95},
        {"ear": 0.22, "det_score": 0.95},
        {"ear": 0.26, "det_score": 0.95},
    ]

    result = analyze_challenge(history, "blink")

    assert result == {"passed": True, "feedback": "✅ Blink detected!"}


def test_analyze_challenge_blink_passes_for_short_realistic_webcam_sequence() -> None:
    history = [
        {"ear": 0.29, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.28, "det_score": 0.95},
        {"ear": 0.11, "det_score": 0.95},
        {"ear": 0.17, "det_score": 0.95},
        {"ear": 0.22, "det_score": 0.95},
    ]

    result = analyze_challenge(history, "blink")

    assert result == {"passed": True, "feedback": "✅ Blink detected!"}


def test_analyze_challenge_blink_does_not_pass_without_eye_reopening() -> None:
    history = [
        {"ear": 0.29, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.28, "det_score": 0.95},
        {"ear": 0.12, "det_score": 0.95},
        {"ear": 0.10, "det_score": 0.95},
        {"ear": 0.11, "det_score": 0.95},
    ]

    result = analyze_challenge(history, "blink")

    assert result["passed"] is False


def test_analyze_challenge_blink_falls_back_to_detection_score_when_ear_stays_flat() -> None:
    history = [
        {"ear": 0.10, "det_score": 0.97},
        {"ear": 0.10, "det_score": 0.96},
        {"ear": 0.10, "det_score": 0.97},
        {"ear": 0.10, "det_score": 0.96},
        {"ear": 0.10, "det_score": 0.97},
        {"ear": 0.10, "det_score": 0.88},
        {"ear": 0.10, "det_score": 0.96},
        {"ear": 0.10, "det_score": 0.97},
    ]

    result = analyze_challenge(history, "blink")

    assert result == {"passed": True, "feedback": "✅ Blink detected!"}


def test_run_face_match_returns_error_for_corrupt_reference_embedding() -> None:
    result = run_face_match(_make_face(normed_embedding=np.array([1.0], dtype=np.float32)), "not-base64")

    assert result["passed"] is False
    assert "Reference embedding corrupted" in result["message"]


def test_run_face_match_skips_match_when_live_embedding_is_missing() -> None:
    reference = base64.b64encode(np.array([1.0, 0.0], dtype=np.float32).tobytes()).decode("ascii")

    result = run_face_match(_make_face(normed_embedding=None), reference)

    assert result["passed"] is True
    assert result["display_score"] == pytest.approx(100.0)
    assert "face-match skipped" in result["message"]


def test_run_face_match_returns_failure_message_below_threshold() -> None:
    reference_embedding = np.array([1.0, 0.0], dtype=np.float32)
    reference = base64.b64encode(reference_embedding.tobytes()).decode("ascii")
    live_face = _make_face(normed_embedding=np.array([0.2, 0.0], dtype=np.float32))

    result = run_face_match(live_face, reference)

    assert result["passed"] is False
    assert result["cosine_similarity"] == pytest.approx(0.2)
    assert result["display_score"] == pytest.approx(46.6666666667)
    assert "Face match failed" in result["message"]


# ── look_straight challenge tests ─────────────────────────────────────────────

def _straight_frame(nose_rel_x: float = 0.50, pitch: float = 0.05) -> dict:
    return {"nose_rel_x": nose_rel_x, "pitch": pitch, "det_score": 0.95, "ear": 0.30}


def test_look_straight_passes_when_face_centred_for_required_frames() -> None:
    # Five consecutive frames with nose centred should pass.
    history = [_straight_frame() for _ in range(5)]
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


def test_look_straight_does_not_pass_with_fewer_than_required_frames() -> None:
    history = [_straight_frame() for _ in range(4)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False


def test_look_straight_fails_when_nose_is_off_centre() -> None:
    # Nose too far to image-right → face is angled
    history = [_straight_frame(nose_rel_x=0.65) for _ in range(5)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False
    assert result["feedback"]  # some directional hint


def test_look_straight_passes_with_realistic_pitch() -> None:
    # For any real face pitch = (nose_y - eye_mid_y) / IOD is ~0.4-0.7.
    # The challenge must pass regardless of pitch as long as nose_rel_x is centred.
    history = [_straight_frame(pitch=0.55) for _ in range(5)]
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


def test_turn_right_passes_when_yaw_sufficiently_negative() -> None:
    # Mirrored frame: subject turns THEIR right → nose swings image-RIGHT → yaw POSITIVE
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.57, "yaw":  0.20, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.64, "yaw":  0.38, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]
    result = analyze_challenge(history, "turn_right")
    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


# ── session challenge_results tests ───────────────────────────────────────────

def test_session_advance_challenge_records_results() -> None:
    from basetruth.kyc.session import KYCSession

    session = KYCSession(
        session_id="test-001",
        customer_name="Tester",
        entity_ref="",
        challenges=["look_straight", "nod"],
        reference_embedding_b64=None,
    )

    assert session.challenge_results == []
    session.advance_challenge()  # complete look_straight
    assert len(session.challenge_results) == 1
    assert session.challenge_results[0] == {
        "index": 0,
        "challenge": "look_straight",
        "passed": True,
    }
    session.advance_challenge()  # complete nod
    assert len(session.challenge_results) == 2
    assert session.challenge_results[1]["challenge"] == "nod"


def test_session_to_status_dict_includes_challenge_results() -> None:
    from basetruth.kyc.session import KYCSession

    session = KYCSession(
        session_id="test-002",
        customer_name="Tester",
        entity_ref="",
        challenges=["look_straight"],
        reference_embedding_b64=None,
    )
    session.advance_challenge()
    status = session.to_status_dict()
    assert "challenge_results" in status
    assert status["challenge_results"][0]["challenge"] == "look_straight"