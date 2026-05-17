from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from basetruth.kyc.liveness import analyze_challenge, extract_features, face_geometry_valid, run_face_match


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


def test_analyze_challenge_turn_left_does_not_pass_without_hold() -> None:
    # Under the new hold-and-return design, simply crossing the yaw threshold
    # in a brief swing is no longer enough.  The user must hold the turned
    # position for _TURN_HOLD_FRAMES frames, THEN return to centre.
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.42, "yaw": -0.18, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.35, "yaw": -0.38, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result["passed"] is False


def test_analyze_challenge_turn_left_does_not_pass_on_threshold_touch_without_hold() -> None:
    # Even reaching exactly -_TURN_YAW_THRESHOLD in a single frame is not enough;
    # the user must hold at that position.
    history = [
        {"nose_rel_x": 0.50, "yaw": 0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.46, "yaw": -0.09, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.41, "yaw": -0.16, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result["passed"] is False


def test_analyze_challenge_turn_left_does_not_pass_for_opposite_direction() -> None:
    history = [
        {"nose_rel_x": 0.50, "yaw": 0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.56, "yaw": 0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.61, "yaw": 0.19, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result["passed"] is False


def test_analyze_challenge_turn_left_passes_with_hold_and_return() -> None:
    # Full hold-and-return sequence: straight → turn left → hold 40 frames → look straight.
    # 40 held frames at yaw=-0.20 satisfies _TURN_HOLD_FRAMES; return frame drops |yaw| below
    # _TURN_CENTRE_YAW=0.10.
    history = (
        [{"nose_rel_x": 0.50, "yaw":  0.02, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}]
        + [{"nose_rel_x": 0.37, "yaw": -0.20, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}] * 40
        + [{"nose_rel_x": 0.51, "yaw":  0.03, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}]
    )

    result = analyze_challenge(history, "turn_left")

    assert result == {"passed": True, "feedback": "✅ Turn completed!"}


def test_analyze_challenge_turn_right_requests_more_movement_when_not_far_enough() -> None:
    # 3 frames, yaw reaches 0.14 < _TURN_YAW_THRESHOLD (0.16) → "Keep turning" feedback.
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.54, "yaw":  0.08, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.57, "yaw":  0.14, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_right")

    assert result["passed"] is False
    assert "Keep turning" in result["feedback"]


def test_analyze_challenge_turn_right_passes_with_hold_and_return() -> None:
    # Full hold-and-return sequence for turn_right.
    history = (
        [{"nose_rel_x": 0.49, "yaw": -0.02, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}]
        + [{"nose_rel_x": 0.64, "yaw":  0.21, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}] * 40
        + [{"nose_rel_x": 0.50, "yaw":  0.04, "pitch": 0.0, "det_score": 1.0, "ear": 0.30}]
    )

    result = analyze_challenge(history, "turn_right")

    assert result == {"passed": True, "feedback": "✅ Turn completed!"}


def test_analyze_challenge_nod_does_not_pass_without_hold() -> None:
    # 3 deviated frames is not sufficient to pass the nod challenge
    # (_NOD_HOLD_FRAMES = 6 consecutive frames required).
    history = [
        {"nose_rel_x": 0.5, "pitch": 0.02, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.05, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.08, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.31, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.34, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.5, "pitch": 0.36, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "nod")

    assert result["passed"] is False


def test_analyze_challenge_nod_passes_with_hold_and_return() -> None:
    # Full hold sequence: 3 baseline frames then 40 frames of head-down
    # (pitch deviates by 0.20 from baseline — well above _NOD_DOWN_DELTA 0.08).
    # 40 > _NOD_HOLD_FRAMES (6) so the challenge passes.
    # baseline_pitch = (0.50 + 0.51 + 0.50) / 3 = 0.503… ≈ 0.503
    # down_pitch = 0.30 → deviation = 0.20 ≥ _NOD_DOWN_DELTA (0.08) ✓
    _b = 0.50  # baseline pitch value
    history = (
        [{"nose_rel_x": 0.5, "pitch": _b + d, "yaw": 0.0, "det_score": 1.0, "ear": 0.30}
         for d in [0.00, 0.01, 0.00]]  # 3 baseline frames
        + [{"nose_rel_x": 0.5, "pitch": _b - 0.20, "yaw": 0.0, "det_score": 1.0, "ear": 0.30}] * 40
        + [{"nose_rel_x": 0.5, "pitch": _b,        "yaw": 0.0, "det_score": 1.0, "ear": 0.30}]
    )

    result = analyze_challenge(history, "nod")

    assert result == {"passed": True, "feedback": "✅ Nod completed!"}


def test_analyze_challenge_nod_passes_when_user_starts_moving_immediately() -> None:
    """A genuine nod must still pass when the user starts moving on the first challenge frames.

    This mirrors the real UI flow where some users begin the chin dip immediately after the
    instruction appears, so the baseline must stay anchored to the earliest stable frame instead
    of averaging already-moving frames into the neutral pose.
    """
    history = [
        {"nose_rel_x": 0.50, "pitch": 0.50, "yaw": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.50, "pitch": 0.55, "yaw": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.50, "pitch": 0.58, "yaw": 0.0, "det_score": 1.0, "ear": 0.30},
    ] + [
        {"nose_rel_x": 0.50, "pitch": 0.62, "yaw": 0.0, "det_score": 1.0, "ear": 0.30}
        for _ in range(6)
    ]

    result = analyze_challenge(history, "nod")

    assert result == {"passed": True, "feedback": "✅ Nod completed!"}


def test_analyze_challenge_blink_passes_on_single_blink() -> None:
    # _BLINK_COUNT_REQUIRED = 1 — a single complete EAR dip-recovery cycle is
    # sufficient to pass the blink challenge.
    history = [
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.31, "det_score": 0.95},
        {"ear": 0.29, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.07, "det_score": 0.95},  # eyes close
        {"ear": 0.22, "det_score": 0.95},  # eyes reopen → one blink complete
        {"ear": 0.26, "det_score": 0.95},
    ]

    result = analyze_challenge(history, "blink")

    assert result == {"passed": True, "feedback": "\u2705 Blink detected!"}


def test_analyze_challenge_blink_passes_when_two_distinct_blinks_present() -> None:
    # With _BLINK_COUNT_REQUIRED = 1, two blinks also pass (on the first complete cycle).
    history = [
        {"ear": 0.30, "det_score": 0.95},  # open baseline
        {"ear": 0.31, "det_score": 0.95},
        {"ear": 0.07, "det_score": 0.95},  # blink 1: eyes close
        {"ear": 0.26, "det_score": 0.95},  # blink 1: eyes reopen → passes here
        {"ear": 0.30, "det_score": 0.95},
        {"ear": 0.07, "det_score": 0.95},  # blink 2: eyes close
        {"ear": 0.28, "det_score": 0.95},  # blink 2: eyes reopen
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

def _straight_frame(nose_rel_x: float = 0.50, pitch: float = 0.05, yaw: float = 0.0) -> dict:
    # yaw=0.0 means the face is looking straight at the camera (nose at the eye midpoint).
    return {"nose_rel_x": nose_rel_x, "yaw": yaw, "pitch": pitch, "det_score": 0.95, "ear": 0.30}


def test_look_straight_passes_when_face_centred_for_required_frames() -> None:
    # Ten or more consecutive frames with nose centred should pass (minimum is now 10).
    history = [_straight_frame() for _ in range(10)]
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


def test_look_straight_does_not_pass_with_fewer_than_required_frames() -> None:
    # Nine frames is below the 10-frame minimum; should not pass yet.
    history = [_straight_frame() for _ in range(9)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False


def test_look_straight_fails_when_nose_is_off_centre() -> None:
    # Nose too far to image-right → face is angled. Use 10 frames so the centering
    # check fires (not the insufficient-history early return).
    history = [_straight_frame(nose_rel_x=0.65) for _ in range(10)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False
    assert result["feedback"]  # some directional hint


def test_look_straight_passes_with_realistic_pitch() -> None:
    # For any real face pitch = (nose_y - eye_mid_y) / IOD is ~0.4-0.7.
    # The challenge must pass regardless of pitch as long as nose_rel_x is centred.
    history = [_straight_frame(pitch=0.55) for _ in range(10)]
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


def test_turn_right_does_not_pass_without_hold() -> None:
    # Under the new hold-and-return design, simply crossing the yaw threshold
    # in a brief swing is no longer enough.  The user must hold the turned
    # position for _TURN_HOLD_FRAMES frames, THEN return to centre.
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.57, "yaw":  0.20, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.64, "yaw":  0.38, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]
    result = analyze_challenge(history, "turn_right")
    assert result["passed"] is False


# ── ghost / no-face detection tests ──────────────────────────────────────────

def _ghost_frame(pitch: float = 0.0, yaw: float = 0.0, nose_rel_x: float = 0.50) -> dict:
    """A frame with det_score below the minimum threshold — simulates a background
    object or empty room producing a marginal false-positive detection.
    All challenges must refuse to pass on a history made up solely of such frames."""
    return {"nose_rel_x": nose_rel_x, "yaw": yaw, "pitch": pitch, "det_score": 0.40, "ear": 0.30}


def test_look_straight_does_not_pass_on_low_confidence_ghost_frames() -> None:
    """look_straight must not pass when only low-confidence ghost frames are present."""
    history = [_ghost_frame(nose_rel_x=0.50) for _ in range(5)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False


def test_nod_does_not_pass_on_low_confidence_ghost_frames() -> None:
    """nod must not pass when pitch variance comes from low-confidence ghost frames.
    This is the scenario where the user hides under the table and jitter in the
    background detection causes pitch to vary by more than the threshold."""
    # Pitch range of 0.20 would normally satisfy the nod threshold.
    history = [
        _ghost_frame(pitch=0.00),
        _ghost_frame(pitch=0.05),
        _ghost_frame(pitch=0.10),
        _ghost_frame(pitch=0.20),
        _ghost_frame(pitch=0.15),
        _ghost_frame(pitch=0.08),
    ]
    result = analyze_challenge(history, "nod")
    assert result["passed"] is False


def test_turn_left_does_not_pass_on_low_confidence_ghost_frames() -> None:
    """turn_left must not pass when yaw shift comes from low-confidence ghost frames."""
    history = [
        _ghost_frame(yaw=0.00, nose_rel_x=0.50),
        _ghost_frame(yaw=-0.10, nose_rel_x=0.45),
        _ghost_frame(yaw=-0.20, nose_rel_x=0.40),
    ]
    result = analyze_challenge(history, "turn_left")
    assert result["passed"] is False


def test_blink_does_not_pass_on_low_confidence_ghost_frames() -> None:
    """blink must not pass when the EAR dip comes from low-confidence ghost frames."""
    history = [
        {"ear": 0.30, "det_score": 0.40},
        {"ear": 0.30, "det_score": 0.40},
        {"ear": 0.07, "det_score": 0.40},
        {"ear": 0.22, "det_score": 0.40},
    ]
    result = analyze_challenge(history, "blink")
    assert result["passed"] is False


def test_challenges_pass_normally_with_mixed_ghost_and_real_frames() -> None:
    """Challenges should still pass when high-confidence real frames are mixed with
    low-confidence ghost frames — the ghost frames are simply ignored."""
    # 10 real frontal frames mixed with 2 ghost frames.
    history = (
        [_ghost_frame(nose_rel_x=0.80)]
        + [_straight_frame(nose_rel_x=0.50, yaw=0.0) for _ in range(5)]
        + [_ghost_frame(nose_rel_x=0.20)]
        + [_straight_frame(nose_rel_x=0.50, yaw=0.0) for _ in range(5)]
    )
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


# ── face-size / empty-room gate tests ────────────────────────────────────────

def _tiny_face_frame(pitch: float = 0.0, yaw: float = 0.0, nose_rel_x: float = 0.50) -> dict:
    """A frame with bbox_area_ratio below MIN_FACE_AREA_RATIO — simulates a user who
    has moved away from the camera or a distant background object detected as a face."""
    from basetruth.kyc.liveness import MIN_FACE_AREA_RATIO
    return {
        "nose_rel_x": nose_rel_x,
        "yaw": yaw,
        "pitch": pitch,
        "det_score": 0.90,   # high confidence — passes the det_score gate
        "ear": 0.30,
        "bbox_area_ratio": MIN_FACE_AREA_RATIO * 0.5,  # half the minimum
    }


def test_look_straight_does_not_pass_when_face_is_too_small() -> None:
    """look_straight must not pass when the face bbox is too small.
    This catches the 'user ducked under table and background object detected' case."""
    history = [_tiny_face_frame(nose_rel_x=0.50, yaw=0.0) for _ in range(5)]
    result = analyze_challenge(history, "look_straight")
    assert result["passed"] is False


def test_nod_does_not_pass_when_face_is_too_small() -> None:
    """nod must not pass when pitch variance comes from a tiny-face detection.
    Pitch range of 0.20 would normally satisfy the nod threshold; it must be
    blocked when the face is too small to be a real person in the oval."""
    history = [
        _tiny_face_frame(pitch=0.00),
        _tiny_face_frame(pitch=0.05),
        _tiny_face_frame(pitch=0.10),
        _tiny_face_frame(pitch=0.20),
        _tiny_face_frame(pitch=0.15),
        _tiny_face_frame(pitch=0.08),
    ]
    result = analyze_challenge(history, "nod")
    assert result["passed"] is False


def test_look_straight_passes_normally_when_face_is_large_enough() -> None:
    """look_straight must still pass when bbox_area_ratio meets the minimum."""
    from basetruth.kyc.liveness import MIN_FACE_AREA_RATIO
    history = []
    for _ in range(10):
        frame = _straight_frame(nose_rel_x=0.50, yaw=0.0)
        frame["bbox_area_ratio"] = MIN_FACE_AREA_RATIO * 3  # well above minimum
        history.append(frame)
    result = analyze_challenge(history, "look_straight")
    assert result == {"passed": True, "feedback": "✅ Frontal face captured!"}


def test_nod_passes_normally_when_face_is_large_enough() -> None:
    """nod must still pass when bbox_area_ratio meets the minimum.
    Uses the full hold-and-return sequence: 3 baseline frames, 40 held-down frames,
    then 1 return frame. All frames have bbox_area_ratio above the minimum."""
    from basetruth.kyc.liveness import MIN_FACE_AREA_RATIO
    area = MIN_FACE_AREA_RATIO * 3
    _b = 0.50  # baseline pitch value
    history = (
        [{"nose_rel_x": 0.50, "pitch": _b, "yaw": 0.0, "det_score": 0.90, "ear": 0.30, "bbox_area_ratio": area} for _ in range(3)]
        + [{"nose_rel_x": 0.50, "pitch": _b - 0.20, "yaw": 0.0, "det_score": 0.90, "ear": 0.30, "bbox_area_ratio": area}] * 40
        + [{"nose_rel_x": 0.50, "pitch": _b, "yaw": 0.0, "det_score": 0.90, "ear": 0.30, "bbox_area_ratio": area}]
    )
    result = analyze_challenge(history, "nod")
    assert result == {"passed": True, "feedback": "✅ Nod completed!"}


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


# ── wrong_motion field tests ───────────────────────────────────────────────────

def test_turn_left_wrong_direction_includes_wrong_motion_and_reset() -> None:
    """Turning right during turn_left must produce wrong_motion='turned_right' and reset_needed=True."""
    history = [
        {"nose_rel_x": 0.50, "yaw": 0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.56, "yaw": 0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.61, "yaw": 0.18, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.65, "yaw": 0.22, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.67, "yaw": 0.25, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]
    result = analyze_challenge(history, "turn_left")

    assert result["passed"] is False
    assert result["reset_needed"] is True
    assert result["wrong_motion"] == "turned_right"
    assert "LEFT" in result["feedback"]


def test_turn_right_wrong_direction_includes_wrong_motion_and_reset() -> None:
    """Turning left during turn_right must produce wrong_motion='turned_left' and reset_needed=True."""
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.44, "yaw": -0.11, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.39, "yaw": -0.18, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.35, "yaw": -0.22, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.33, "yaw": -0.25, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]
    result = analyze_challenge(history, "turn_right")

    assert result["passed"] is False
    assert result["reset_needed"] is True
    assert result["wrong_motion"] == "turned_left"
    assert "RIGHT" in result["feedback"]


def test_nod_side_to_side_includes_wrong_motion() -> None:
    """A side-to-side shake during nod must include wrong_motion='side_to_side_shake'."""
    # Yaw oscillates a lot (side-to-side), pitch barely moves.
    history = [
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.02, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.44, "yaw": -0.10, "pitch": 0.03, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.02, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.56, "yaw":  0.10, "pitch": 0.03, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.50, "yaw":  0.00, "pitch": 0.02, "det_score": 1.0, "ear": 0.30},
    ]
    result = analyze_challenge(history, "nod")

    assert result["passed"] is False
    assert result.get("wrong_motion") == "side_to_side_shake"
    assert "nod" in result["feedback"].lower() or "DOWN" in result["feedback"]


def test_challenge_wrong_actions_recorded_in_live_session() -> None:
    """Wrong-direction response from analyze_challenge must be recorded in session.challenge_wrong_actions.

    We test this by directly calling analyze_challenge with a wrong-direction history (as
    process_live_frame_message would) and then simulating what process_live_frame_message
    does with the result: append to challenge_wrong_actions and flag feedback_sticky.
    """
    from basetruth.face_scan.live import FaceScanLiveSession

    session = FaceScanLiveSession(
        session_id="wa-test-001",
        challenges=["turn_left"],
        status="active",
    )

    # Five frames turning right during a turn_left challenge — should trigger reset_needed
    wrong_history = [
        {"nose_rel_x": 0.62, "yaw": 0.20, "pitch": 0.0, "det_score": 0.95, "ear": 0.30},
        {"nose_rel_x": 0.63, "yaw": 0.21, "pitch": 0.0, "det_score": 0.95, "ear": 0.30},
        {"nose_rel_x": 0.64, "yaw": 0.22, "pitch": 0.0, "det_score": 0.95, "ear": 0.30},
        {"nose_rel_x": 0.65, "yaw": 0.23, "pitch": 0.0, "det_score": 0.95, "ear": 0.30},
        {"nose_rel_x": 0.66, "yaw": 0.24, "pitch": 0.0, "det_score": 0.95, "ear": 0.30},
    ]

    # Reproduce what process_live_frame_message does with a reset_needed result
    analysis = analyze_challenge(wrong_history, "turn_left")
    assert analysis.get("reset_needed") is True, "Precondition: wrong-direction must trigger reset"
    assert analysis.get("wrong_motion") == "turned_right"

    # Simulate the recording logic from process_live_frame_message
    if analysis.get("reset_needed") or analysis.get("wrong_motion"):
        session.challenge_wrong_actions.append({
            "challenge": "turn_left",
            "wrong_motion": analysis.get("wrong_motion", "unknown"),
            "frame_index": 5,
        })

    assert len(session.challenge_wrong_actions) == 1
    entry = session.challenge_wrong_actions[0]
    assert entry["challenge"] == "turn_left"
    assert entry["wrong_motion"] == "turned_right"
    assert entry["frame_index"] == 5


def test_challenge_wrong_actions_in_result_payload() -> None:
    """build_live_face_scan_result must include wrong_actions in active_liveness."""
    from basetruth.face_scan.live import FaceScanLiveSession, build_live_face_scan_result
    from unittest.mock import patch, MagicMock

    session = FaceScanLiveSession(
        session_id="wa-test-002",
        challenges=["turn_left"],
        status="active",
    )
    session.current_challenge_idx = 1  # mark all challenges done
    session.challenge_wrong_actions = [
        {"challenge": "turn_left", "wrong_motion": "turned_right", "frame_index": 5},
    ]

    # Patch narrative so we don't call the LLM; return a valid (text, source) tuple
    dummy_narrative = MagicMock(return_value=("Mock narrative text.", "none"))

    with patch("basetruth.face_scan.live._narrative_mod.generate_face_scan_narrative", dummy_narrative):
        result = build_live_face_scan_result(session)

    al = result["checks"]["active_liveness"]
    assert al["wrong_action_count"] == 1
    assert len(al["wrong_actions"]) == 1
    assert al["wrong_actions"][0]["wrong_motion"] == "turned_right"


# ── face_geometry_valid tests ─────────────────────────────────────────────────

def _frontal_face(
    bbox=(0, 0, 100, 120),
    left_eye=(30, 35),
    right_eye=(70, 35),
    nose=(50, 60),
    mouth_l=(35, 85),
    mouth_r=(65, 85),
):
    """Build a synthetic face object with plausible frontal-face geometry."""
    return SimpleNamespace(
        kps=np.array([left_eye, right_eye, nose, mouth_l, mouth_r], dtype=float),
        bbox=np.array(bbox, dtype=float),
        det_score=0.95,
        ear=0.30,
        normed_embedding=None,
    )


def test_face_geometry_valid_accepts_normal_frontal_face() -> None:
    """A well-formed frontal face with all invariants satisfied must pass."""
    face = _frontal_face()
    valid, reason = face_geometry_valid(face)
    assert valid is True, f"Expected valid face but got reason: {reason}"


def test_face_geometry_valid_rejects_nose_above_eyes() -> None:
    """If the nose landmark is above the eye landmarks, reject — this never happens on a real face."""
    # Place nose above the eyes to simulate a hand/palm mis-detection
    face = _frontal_face(nose=(50, 10))   # nose_y=10 < eye_y=35
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "Nose" in reason or "nose" in reason


def test_face_geometry_valid_rejects_eyes_at_very_different_heights() -> None:
    """Eyes at wildly different vertical positions are not a real face."""
    # Spread eyes vertically by 50% of bbox height (way above the 25% threshold).
    # Nose is placed BELOW both eyes so Check 1 (nose below eyes) passes and the
    # function reaches the eye-gap check (Check 2).
    face = _frontal_face(left_eye=(30, 20), right_eye=(70, 80), nose=(50, 90))
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "height" in reason or "gap" in reason


def test_face_geometry_valid_rejects_iod_too_small() -> None:
    """Eyes that are extremely close together are not a real face (e.g. fingertips together)."""
    # Eyes only 5 px apart on a 100 px wide bbox → iod_ratio = 0.05 (below 0.15 threshold)
    face = _frontal_face(left_eye=(48, 35), right_eye=(52, 35))
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "Interocular" in reason or "interocular" in reason


def test_face_geometry_valid_rejects_iod_too_large() -> None:
    """Eyes further apart than the face width are not a real face (e.g. wide-spread fingers)."""
    # Eyes 95 px apart on a 100 px wide bbox → iod_ratio = 0.95 (above 0.65 threshold)
    face = _frontal_face(left_eye=(3, 35), right_eye=(97, 35))
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "Interocular" in reason or "interocular" in reason


def test_face_geometry_valid_rejects_eyes_too_low() -> None:
    """Eye midpoint in the lower half of the bbox cannot be a real face."""
    # Eyes at y=100 in a 120-tall bbox → ratio = 100/120 = 0.83 (above 0.65)
    face = _frontal_face(left_eye=(30, 100), right_eye=(70, 100), nose=(50, 110))
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "low" in reason or "height" in reason


def test_face_geometry_valid_rejects_missing_kps() -> None:
    """A face object with no keypoints should be rejected cleanly."""
    face = SimpleNamespace(kps=None, bbox=np.array([0, 0, 100, 120], dtype=float))
    valid, reason = face_geometry_valid(face)
    assert valid is False
    assert "keypoints" in reason.lower() or "No face" in reason


def test_face_geometry_valid_accepts_slightly_tilted_face() -> None:
    """A face tilted a bit to one side (one eye ~10% bbox_h higher) should still pass."""
    # One eye at y=30, other at y=40 → gap ratio = 10/120 = 0.083 (well under 0.25)
    face = _frontal_face(left_eye=(30, 30), right_eye=(70, 40), nose=(50, 60))
    valid, reason = face_geometry_valid(face)
    assert valid is True, f"Slightly tilted face should be accepted, got: {reason}"


# ── is_face_stable tests ──────────────────────────────────────────────────────

def _stable_call(
    *,
    face_count: int = 1,
    bbox_area_ratio: float = 0.10,
    confidence: float = 0.90,
    nose_rel_x: float = 0.50,
    nose_rel_y: float = 0.50,
    yaw: float = 0.0,
    pitch: float = 0.0,
    texture_score: float = 99.0,
    brightness_mean: float = 128.0,
):
    """Build a minimal face stub and call is_face_stable with the given parameters."""
    from basetruth.kyc.liveness import is_face_stable

    # Provide a face with enough kps to satisfy face_geometry_valid (not used by is_face_stable
    # itself, but we want a consistent stub).
    face = SimpleNamespace(
        kps=np.array([[30, 35], [70, 35], [50, 60], [35, 85], [65, 85]], dtype=float),
        bbox=np.array([0, 0, 100, 120], dtype=float),
        det_score=confidence,
        ear=0.30,
        normed_embedding=None,
    )
    return is_face_stable(
        face=face,
        face_count=face_count,
        bbox_area_ratio=bbox_area_ratio,
        confidence=confidence,
        nose_rel_x=nose_rel_x,
        nose_rel_y=nose_rel_y,
        yaw=yaw,
        pitch=pitch,
        texture_score=texture_score,
        brightness_mean=brightness_mean,
    )


def test_is_face_stable_passes_all_conditions_met() -> None:
    """A well-positioned, high-confidence, single face must pass."""
    ok, feedback = _stable_call()
    assert ok is True
    assert feedback == ""


def test_is_face_stable_rejects_no_face() -> None:
    """Zero detected faces must return False with an appropriate message."""
    ok, feedback = _stable_call(face_count=0)
    assert ok is False
    assert "No face detected" in feedback


def test_is_face_stable_rejects_multiple_faces() -> None:
    """More than one face must return False with a multi-face message."""
    ok, feedback = _stable_call(face_count=2)
    assert ok is False
    assert "Multiple faces" in feedback


def test_is_face_stable_rejects_low_confidence() -> None:
    """Confidence below FACE_STABILITY_CONFIDENCE_MIN must return False."""
    from basetruth.kyc.liveness import FACE_STABILITY_CONFIDENCE_MIN

    ok, feedback = _stable_call(confidence=FACE_STABILITY_CONFIDENCE_MIN - 0.01)
    assert ok is False
    assert "not clearly visible" in feedback.lower() or "visible" in feedback.lower()


def test_is_face_stable_rejects_tiny_face() -> None:
    """A face bbox smaller than FACE_STABILITY_AREA_MIN must return False."""
    from basetruth.kyc.liveness import FACE_STABILITY_AREA_MIN

    ok, feedback = _stable_call(bbox_area_ratio=FACE_STABILITY_AREA_MIN * 0.5)
    assert ok is False
    assert "closer" in feedback.lower() or "small" in feedback.lower()


def test_is_face_stable_rejects_face_too_far_left() -> None:
    """Nose too far to the image-left must return False with a centering message."""
    from basetruth.kyc.liveness import FACE_STABILITY_X_MIN

    ok, feedback = _stable_call(nose_rel_x=FACE_STABILITY_X_MIN - 0.01)
    assert ok is False
    assert "centre" in feedback.lower() or "right" in feedback.lower()


def test_is_face_stable_rejects_face_too_far_right() -> None:
    """Nose too far to the image-right must return False with a centering message."""
    from basetruth.kyc.liveness import FACE_STABILITY_X_MAX

    ok, feedback = _stable_call(nose_rel_x=FACE_STABILITY_X_MAX + 0.01)
    assert ok is False
    assert "centre" in feedback.lower() or "left" in feedback.lower()


def test_face_stable_frames_accumulates_to_required() -> None:
    """Simulated session counter: N good frames → reaches FACE_STABLE_FRAMES_REQUIRED."""
    from basetruth.kyc.liveness import FACE_STABLE_FRAMES_REQUIRED

    stable_count = 0
    for _ in range(FACE_STABLE_FRAMES_REQUIRED):
        ok, _ = _stable_call()
        if ok:
            stable_count += 1

    assert stable_count == FACE_STABLE_FRAMES_REQUIRED


def test_face_stable_frames_resets_on_low_confidence() -> None:
    """Simulated counter: interrupting good frames with one bad frame must reset the count."""
    from basetruth.kyc.liveness import FACE_STABLE_FRAMES_REQUIRED, FACE_STABILITY_CONFIDENCE_MIN

    # Good frames accumulate
    counter = 0
    for _ in range(4):
        ok, _ = _stable_call()
        if ok:
            counter += 1
    assert counter == 4

    # One bad frame resets counter
    ok, _ = _stable_call(confidence=FACE_STABILITY_CONFIDENCE_MIN - 0.05)
    if not ok:
        counter = 0
    assert counter == 0


def test_face_stable_frames_requires_continuous_window() -> None:
    """Counter must only reach threshold if N frames are consecutive without a reset."""
    from basetruth.kyc.liveness import FACE_STABLE_FRAMES_REQUIRED, FACE_STABILITY_CONFIDENCE_MIN

    counter = 0
    for i in range(FACE_STABLE_FRAMES_REQUIRED * 2):
        # Inject a bad frame halfway through to force a reset
        if i == FACE_STABLE_FRAMES_REQUIRED - 1:
            ok, _ = _stable_call(confidence=FACE_STABILITY_CONFIDENCE_MIN - 0.05)
            if not ok:
                counter = 0
        else:
            ok, _ = _stable_call()
            if ok:
                counter += 1
            else:
                counter = 0

    # After a reset midway, the remaining good frames should fill a new window
    assert counter == FACE_STABLE_FRAMES_REQUIRED


# ── is_face_stable: expert-review additions ───────────────────────────────────

def test_is_face_stable_rejects_face_too_high_vertically() -> None:
    """Nose above the vertical band (nose_rel_y < FACE_STABILITY_Y_MIN) must return False."""
    from basetruth.kyc.liveness import FACE_STABILITY_Y_MIN

    ok, feedback = _stable_call(nose_rel_y=FACE_STABILITY_Y_MIN - 0.01)
    assert ok is False
    assert "vertically" in feedback.lower() or "centre" in feedback.lower()


def test_is_face_stable_rejects_face_too_low_vertically() -> None:
    """Nose below the vertical band (nose_rel_y > FACE_STABILITY_Y_MAX) must return False."""
    from basetruth.kyc.liveness import FACE_STABILITY_Y_MAX

    ok, feedback = _stable_call(nose_rel_y=FACE_STABILITY_Y_MAX + 0.01)
    assert ok is False
    assert "vertically" in feedback.lower() or "centre" in feedback.lower()


def test_is_face_stable_accepts_nose_at_vertical_boundaries() -> None:
    """Nose exactly at the Y boundary values must pass (inclusive test)."""
    from basetruth.kyc.liveness import FACE_STABILITY_Y_MIN, FACE_STABILITY_Y_MAX

    ok_low, _ = _stable_call(nose_rel_y=FACE_STABILITY_Y_MIN)
    ok_high, _ = _stable_call(nose_rel_y=FACE_STABILITY_Y_MAX)
    assert ok_low is True
    assert ok_high is True


def test_is_face_stable_rejects_yaw_too_large() -> None:
    """Head turned too far (|yaw| > FACE_STABILITY_YAW_MAX) must return False."""
    from basetruth.kyc.liveness import FACE_STABILITY_YAW_MAX

    ok_pos, fb_pos = _stable_call(yaw=FACE_STABILITY_YAW_MAX + 0.01)
    ok_neg, fb_neg = _stable_call(yaw=-(FACE_STABILITY_YAW_MAX + 0.01))
    assert ok_pos is False
    assert "straight" in fb_pos.lower() or "camera" in fb_pos.lower()
    assert ok_neg is False


def test_is_face_stable_accepts_yaw_within_limit() -> None:
    """Yaw exactly at FACE_STABILITY_YAW_MAX must pass."""
    from basetruth.kyc.liveness import FACE_STABILITY_YAW_MAX

    ok, _ = _stable_call(yaw=FACE_STABILITY_YAW_MAX)
    assert ok is True


def test_is_face_stable_pitch_does_not_affect_stability() -> None:
    """Pitch is intentionally NOT checked by is_face_stable().

    The 5-point InsightFace pitch formula measures (nose_y - eye_mid_y) / IOD,
    which is an anatomical constant (0.4-0.7) for any real face regardless of
    head tilt. Real 3D pitch estimation requires 3D landmarks. Removing the
    pitch check eliminates false rejections that blocked all real users.
    """
    # Large pitch values must NOT cause rejection — pitch is not checked.
    ok_pos, _ = _stable_call(pitch=0.9)
    ok_neg, _ = _stable_call(pitch=0.1)
    assert ok_pos is True, "Large pitch value must not reject a stable face"
    assert ok_neg is True, "Small pitch value must not reject a stable face"


def test_is_face_stable_rejects_low_texture() -> None:
    """A face with very low texture score (flat screen/photo) must return False."""
    from basetruth.kyc.liveness import MIN_FACE_TEXTURE_SCORE

    ok, feedback = _stable_call(texture_score=MIN_FACE_TEXTURE_SCORE - 1.0)
    assert ok is False
    assert "screen" in feedback.lower() or "real" in feedback.lower() or "camera" in feedback.lower()


def test_is_face_stable_accepts_high_texture() -> None:
    """A face with texture score well above threshold must pass."""
    from basetruth.kyc.liveness import MIN_FACE_TEXTURE_SCORE

    ok, _ = _stable_call(texture_score=MIN_FACE_TEXTURE_SCORE + 20.0)
    assert ok is True


def test_is_face_stable_uses_stability_area_min_not_live_gate() -> None:
    """An area between MIN_FACE_AREA_RATIO and FACE_STABILITY_AREA_MIN must fail the stability gate."""
    from basetruth.kyc.liveness import MIN_FACE_AREA_RATIO, FACE_STABILITY_AREA_MIN

    # This area is above the live-challenge gate but below the stability gate.
    area_between = (MIN_FACE_AREA_RATIO + FACE_STABILITY_AREA_MIN) / 2.0
    ok, feedback = _stable_call(bbox_area_ratio=area_between)
    assert ok is False
    assert "closer" in feedback.lower() or "small" in feedback.lower()


# ── compute_face_texture_score tests ─────────────────────────────────────────

def test_compute_face_texture_score_high_for_noisy_image() -> None:
    """A random-noise image (high local std dev) must score above MIN_FACE_TEXTURE_SCORE."""
    pytest.importorskip("cv2")
    import numpy as np
    from basetruth.kyc.liveness import compute_face_texture_score, MIN_FACE_TEXTURE_SCORE

    rng = np.random.default_rng(42)
    img_bgr = rng.integers(0, 256, (120, 100, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 100, 120], dtype=float)
    score = compute_face_texture_score(img_bgr, bbox)
    assert score > MIN_FACE_TEXTURE_SCORE, f"Random image should score high, got {score}"


def test_compute_face_texture_score_low_for_flat_image() -> None:
    """A uniform-colour image (zero variance) must score below MIN_FACE_TEXTURE_SCORE."""
    pytest.importorskip("cv2")
    import numpy as np
    from basetruth.kyc.liveness import compute_face_texture_score, MIN_FACE_TEXTURE_SCORE

    # Constant 128 — absolutely zero local texture.
    img_bgr = np.full((120, 100, 3), 128, dtype=np.uint8)
    bbox = np.array([0, 0, 100, 120], dtype=float)
    score = compute_face_texture_score(img_bgr, bbox)
    assert score < MIN_FACE_TEXTURE_SCORE, f"Flat image should score low, got {score}"


def test_compute_face_texture_score_returns_99_for_tiny_bbox() -> None:
    """A bbox that clips to zero area must return 99.0 (passes by default)."""
    pytest.importorskip("cv2")
    import numpy as np
    from basetruth.kyc.liveness import compute_face_texture_score

    img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    # bbox outside image — clips to nothing
    bbox = np.array([20, 20, 20, 20], dtype=float)
    score = compute_face_texture_score(img_bgr, bbox)
    assert score == 99.0


# ── Issue 2: Adaptive texture threshold ──────────────────────────────────────

def test_get_adaptive_texture_threshold_low_brightness() -> None:
    """Below LOW_BRIGHTNESS_THRESHOLD the relaxed (lower) threshold is returned."""
    from basetruth.kyc.liveness import (
        get_adaptive_texture_threshold,
        LOW_BRIGHTNESS_THRESHOLD, LOW_BRIGHTNESS_TEXTURE_SCORE, MIN_FACE_TEXTURE_SCORE,
    )
    threshold = get_adaptive_texture_threshold(LOW_BRIGHTNESS_THRESHOLD - 1)
    assert threshold == LOW_BRIGHTNESS_TEXTURE_SCORE
    assert threshold < MIN_FACE_TEXTURE_SCORE  # must be relaxed relative to normal


def test_get_adaptive_texture_threshold_normal_brightness() -> None:
    """At or above LOW_BRIGHTNESS_THRESHOLD the full threshold is returned."""
    from basetruth.kyc.liveness import (
        get_adaptive_texture_threshold,
        LOW_BRIGHTNESS_THRESHOLD, MIN_FACE_TEXTURE_SCORE,
    )
    threshold = get_adaptive_texture_threshold(LOW_BRIGHTNESS_THRESHOLD)
    assert threshold == MIN_FACE_TEXTURE_SCORE


def test_is_face_stable_passes_low_texture_when_dark() -> None:
    """A low texture score that would fail in normal light must pass in a dark frame."""
    from basetruth.kyc.liveness import (
        MIN_FACE_TEXTURE_SCORE, LOW_BRIGHTNESS_TEXTURE_SCORE, LOW_BRIGHTNESS_THRESHOLD,
    )
    # Score is between the relaxed (dark) and normal thresholds.
    mid_score = (LOW_BRIGHTNESS_TEXTURE_SCORE + MIN_FACE_TEXTURE_SCORE) / 2.0
    # Dark frame (brightness < LOW_BRIGHTNESS_THRESHOLD) → relaxed threshold applies → passes.
    ok_dark, _ = _stable_call(texture_score=mid_score, brightness_mean=LOW_BRIGHTNESS_THRESHOLD - 10)
    assert ok_dark is True
    # Normal frame (brightness ≥ LOW_BRIGHTNESS_THRESHOLD) → full threshold → fails.
    ok_normal, fb = _stable_call(texture_score=mid_score, brightness_mean=LOW_BRIGHTNESS_THRESHOLD + 10)
    assert ok_normal is False
    assert "screen" in fb.lower() or "real" in fb.lower()


def test_compute_face_brightness_returns_value() -> None:
    """compute_face_brightness must return a float in [0, 255] for valid input."""
    pytest.importorskip("cv2")
    import numpy as np
    from basetruth.kyc.liveness import compute_face_brightness

    img_bgr = np.full((120, 100, 3), 100, dtype=np.uint8)
    bbox = np.array([0, 0, 100, 120], dtype=float)
    brightness = compute_face_brightness(img_bgr, bbox)
    assert 0.0 <= brightness <= 255.0


def test_compute_face_brightness_returns_128_for_bad_bbox() -> None:
    """A bbox that clips to nothing must return 128.0 (neutral default)."""
    pytest.importorskip("cv2")
    import numpy as np
    from basetruth.kyc.liveness import compute_face_brightness

    img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = np.array([20, 20, 20, 20], dtype=float)  # outside image
    brightness = compute_face_brightness(img_bgr, bbox)
    assert brightness == 128.0


# ── Issue 3: Yaw frequency anti-gaming ───────────────────────────────────────

def test_is_yaw_motion_natural_passes_irregular_signal() -> None:
    """An irregular yaw buffer (real micro-movement) must pass the naturalness check."""
    from basetruth.kyc.liveness import is_yaw_motion_natural

    # Monotonically drifting signal — very few or zero sign alternations.
    yaw_buffer = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]
    ok, feedback = is_yaw_motion_natural(yaw_buffer)
    assert ok is True
    assert feedback == ""


def test_is_yaw_motion_natural_rejects_perfectly_alternating_signal() -> None:
    """A perfectly alternating yaw buffer (artificial jitter) must fail."""
    from basetruth.kyc.liveness import is_yaw_motion_natural

    # Every frame flips direction: the classic screen-shaking pattern.
    yaw_buffer = [0.0, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01]
    ok, feedback = is_yaw_motion_natural(yaw_buffer)
    assert ok is False
    assert "natural" in feedback.lower() or "unnatural" in feedback.lower()


def test_is_yaw_motion_natural_passes_short_buffer() -> None:
    """A buffer with fewer than 4 values must always pass (not enough data to test)."""
    from basetruth.kyc.liveness import is_yaw_motion_natural

    ok, _ = is_yaw_motion_natural([0.0, 0.01, -0.01])
    assert ok is True


# ── Issue 5: Challenge timeout ────────────────────────────────────────────────

def test_challenge_timeout_seconds_is_positive() -> None:
    """CHALLENGE_TIMEOUT_SECONDS must be a positive float."""
    from basetruth.kyc.liveness import CHALLENGE_TIMEOUT_SECONDS

    assert isinstance(CHALLENGE_TIMEOUT_SECONDS, float)
    assert CHALLENGE_TIMEOUT_SECONDS > 0

