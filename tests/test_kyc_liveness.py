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
    history = [
        {"nose_rel_x": 0.50, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.58, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.64, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_left")

    assert result == {"passed": True, "feedback": "✅ Turn detected!"}


def test_analyze_challenge_turn_right_requests_more_movement_when_not_far_enough() -> None:
    history = [
        {"nose_rel_x": 0.51, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.47, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
        {"nose_rel_x": 0.44, "pitch": 0.0, "det_score": 1.0, "ear": 0.30},
    ]

    result = analyze_challenge(history, "turn_right")

    assert result["passed"] is False
    assert "Keep turning" in result["feedback"]


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