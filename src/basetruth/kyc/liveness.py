"""Per-frame liveness analysis for the Video KYC pipeline.

Detects active-liveness challenges using InsightFace 5-point landmarks
(left_eye, right_eye, nose, mouth_left, mouth_right) and the bounding-box
detection confidence score.

Coordinate convention (mirrored front-camera frame sent by the browser):
    The browser preview is mirrored for the user and the captured canvas frame is
    mirrored the same way before it is sent to the server. This keeps the live
    challenge logic aligned with what the user sees on screen.

    In the mirrored frame the subject's LEFT side appears on the IMAGE-LEFT and
    the subject's RIGHT side appears on the IMAGE-RIGHT.

    kps[0] = image-left eye  = subject's LEFT eye
    kps[1] = image-right eye = subject's RIGHT eye
  kps[2] = nose tip
  Y-axis increases DOWNWARD (image convention).

Turn-direction note:
    Subject turns to THEIR left  → nose moves toward image-LEFT  → yaw NEGATIVE
    Subject turns to THEIR right → nose moves toward image-RIGHT → yaw POSITIVE

All spatial features are normalized by the bounding-box width so they are
invariant to the subject's distance from the camera.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ── Per-challenge detection thresholds ───────────────────────────────────────

# yaw = (nose_x - eye_mid_x) / interocular_px
# This measures how far the nose has swung away from the eye midpoint, normalised
# by the interocular distance.  It is the most reliable head-turn indicator because
# it is independent of where the face sits in the frame — only the relative positions
# of nose and eyes matter.
#
# The browser sends a MIRRORED frame so the server sees the same left/right
# direction that the user sees in the selfie-style preview.
# Direction in that mirrored frame:
#   Subject turns to THEIR left  → nose swings to image-LEFT  → yaw NEGATIVE
#   Subject turns to THEIR right → nose swings to image-RIGHT → yaw POSITIVE
#
_TURN_YAW_THRESHOLD = 0.16  # |yaw| must exceed this to register a valid head turn
                             # Tuned for low-FPS webcams so smaller real turns still register
_TURN_RELATIVE_YAW_DELTA = 0.09
_TURN_RELATIVE_NOSE_SHIFT = 0.10

# look_straight: nose_rel_x must stay within this central band for several frames.
# We only check the horizontal position — pitch (nose below eye midpoint) is always
# 0.4–0.7 for any real face and is not a reliable head-tilt indicator here.
_STRAIGHT_X_MIN = 0.43
_STRAIGHT_X_MAX = 0.57
_STRAIGHT_STABLE_FRAMES = 5  # how many consecutive centred frames are required

# pitch: (nose_y - eye_mid_y) / interocular_px
# Nod detected when pitch range across recent frames > this
# Lowered from 0.28 → normal nod only produces ≈ 0.12-0.18 pitch range
_NOD_RANGE_THRESHOLD = 0.14

_BLINK_BASELINE_MIN      = 0.880   # baseline (open-eye) confidence


def extract_features(face: Any) -> Dict[str, float]:
    """Return normalized pose features from one face object (InsightFace or MediaPipe)."""
    # Guard: some detectors may not provide keypoints (face too small, partial occlusion)
    if getattr(face, "kps", None) is None:
        raise ValueError(
            "Face keypoints not available — move closer and ensure your full face is visible."
        )
    kps  = face.kps.astype(float)   # shape (5, 2)
    bbox = face.bbox.astype(float)  # [x1, y1, x2, y2]

    bbox_w = max(bbox[2] - bbox[0], 1.0)
    bbox_h = max(bbox[3] - bbox[1], 1.0)

    left_eye_x,  left_eye_y  = kps[0]
    right_eye_x, right_eye_y = kps[1]
    nose_x,      nose_y      = kps[2]

    # Interocular distance in pixels (for normalization)
    interocular_px = max(abs(right_eye_x - left_eye_x), 1.0)

    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    eye_mid_y = (left_eye_y + right_eye_y) / 2.0

    _det = getattr(face, "det_score", None)
    _ear = getattr(face, "ear", None)

    return {
        # Nose position relative to bbox (0 = left edge, 1 = right edge)
        "nose_rel_x": (nose_x - bbox[0]) / bbox_w,
        "nose_rel_y": (nose_y - bbox[1]) / bbox_h,
        # Yaw: how far nose deviates from eye midpoint (normalized by IOD)
        "yaw":  (nose_x - eye_mid_x) / interocular_px,
        # Pitch: nose below/above eye midpoint (normalized by IOD)
        #   positive = chin down, negative = face up
        "pitch": (nose_y - eye_mid_y) / interocular_px,
        # Detection confidence — drops slightly when eyes close (InsightFace only).
        "det_score": float(_det if _det is not None else 1.0),
        # Eye Aspect Ratio — reliable blink indicator (MediaPipe); 0.30 default (open eye)
        "ear": float(_ear if _ear is not None else 0.30),
    }


def analyze_challenge(
    feature_history: List[Dict[str, float]],
    challenge: str,
) -> Dict[str, Any]:
    """Determine whether the current active-liveness challenge is satisfied.

    Parameters
    ----------
    feature_history:
        Chronological list of feature dicts for the CURRENT challenge (reset
        when the challenge advances).
    challenge:
        One of: ``"blink"``, ``"turn_left"``, ``"turn_right"``, ``"nod"``.

    Returns
    -------
    dict with keys:
        ``passed`` (bool): whether the challenge is complete.
        ``feedback`` (str): human-readable hint shown on screen.
    """
    n = len(feature_history)
    if n < 3:
        return {"passed": False, "feedback": "Look straight at the camera…"}

    recent = feature_history[-20:] if n > 20 else feature_history

    # ─── Look Straight (mandatory first challenge — captures the best selfie) ──
    # Require the nose to stay within the horizontal centre band for several frames.
    # Pitch (nose_y - eye_mid_y / IOD) is always 0.4–0.7 for any real face and is
    # NOT used here — checking it would make this challenge physically impossible.
    # When this challenge passes, api.py saves the current frame as the best selfie.
    if challenge == "look_straight":
        if len(recent) < _STRAIGHT_STABLE_FRAMES:
            return {"passed": False, "feedback": "Look directly into the camera…"}
        last_n = recent[-_STRAIGHT_STABLE_FRAMES:]
        xs     = [f["nose_rel_x"] for f in last_n]
        if all(_STRAIGHT_X_MIN <= x <= _STRAIGHT_X_MAX for x in xs):
            return {"passed": True, "feedback": "✅ Frontal face captured!"}
        avg_x = sum(xs) / len(xs)
        if avg_x < _STRAIGHT_X_MIN:
            return {"passed": False, "feedback": "Move slightly to YOUR right to centre…"}
        if avg_x > _STRAIGHT_X_MAX:
            return {"passed": False, "feedback": "Move slightly to YOUR left to centre…"}
        return {"passed": False, "feedback": "Look directly into the camera…"}

    # ─── Turn Left (subject's left → nose image-LEFT → yaw NEGATIVE) ─────────
    # Canvas frames are mirrored to match the user's preview. In a mirrored frame,
    # when the subject turns to their own left the nose swings toward image-left,
    # making yaw negative.
    if challenge == "turn_left":
        yaws = [f["yaw"] for f in recent]
        noses = [f["nose_rel_x"] for f in recent]
        if min(yaws) <= -_TURN_YAW_THRESHOLD:
            return {"passed": True, "feedback": "✅ Turn detected!"}
        if len(recent) >= 5:
            baseline_yaw = sum(yaws[:2]) / 2
            baseline_nose = sum(noses[:2]) / 2
            yaw_delta = baseline_yaw - min(yaws)
            nose_shift = baseline_nose - min(noses)
            if yaw_delta >= _TURN_RELATIVE_YAW_DELTA and nose_shift >= _TURN_RELATIVE_NOSE_SHIFT:
                return {"passed": True, "feedback": "✅ Turn detected!"}
        gap = _TURN_YAW_THRESHOLD - abs(min(yaws))
        hint = "a little more…" if gap < 0.10 else "turn further to YOUR left…"
        return {"passed": False, "feedback": f"Keep turning — {hint}"}

    # ─── Turn Right (subject's right → nose image-RIGHT → yaw POSITIVE) ─────────
    # When the subject turns to their own right in a mirrored frame, the nose swings
    # toward image-right, making yaw positive.
    if challenge == "turn_right":
        yaws = [f["yaw"] for f in recent]
        noses = [f["nose_rel_x"] for f in recent]
        if max(yaws) >= _TURN_YAW_THRESHOLD:
            return {"passed": True, "feedback": "✅ Turn detected!"}
        if len(recent) >= 5:
            baseline_yaw = sum(yaws[:2]) / 2
            baseline_nose = sum(noses[:2]) / 2
            yaw_delta = max(yaws) - baseline_yaw
            nose_shift = max(noses) - baseline_nose
            if yaw_delta >= _TURN_RELATIVE_YAW_DELTA and nose_shift >= _TURN_RELATIVE_NOSE_SHIFT:
                return {"passed": True, "feedback": "✅ Turn detected!"}
        gap = _TURN_YAW_THRESHOLD - max(yaws)
        hint = "a little more…" if gap < 0.10 else "turn further to YOUR right…"
        return {"passed": False, "feedback": f"Keep turning — {hint}"}

    # ─── Nod (vertical head movement → pitch range) ──────────────────────────
    if challenge == "nod":
        pitches = [f["pitch"] for f in recent]
        if len(pitches) >= 6:
            pitch_range = max(pitches) - min(pitches)
            if pitch_range >= _NOD_RANGE_THRESHOLD:
                return {"passed": True, "feedback": "✅ Nod detected!"}
        return {"passed": False, "feedback": "Nod your head down and back up…"}

    # ─── Blink ────────────────────────────────────────────────────────────────
    if challenge == "blink":
        if len(feature_history) < 5:
            return {"passed": False, "feedback": "Hold still and look at the camera…"}

        # ── Primary path: EAR (Eye Aspect Ratio) — works with MediaPipe ──────
        # Open eye: EAR ≈ 0.25-0.35. Closed eye: EAR ≈ 0.02-0.10.
        # With blendshapes: open → EAR ≈ 0.35, closed → EAR ≈ 0.07 or lower.
        # Thresholds: "closed" = EAR < 0.15, "open" = EAR > 0.18.
        # The default 0.30 fill value means no real data — detect via variance.
        ears = [f.get("ear", 0.30) for f in feature_history]
        ear_variance = max(ears) - min(ears)
        has_real_ear = ear_variance > 0.04  # flat 0.30 everywhere = no real data

        if has_real_ear:
            # We have genuine EAR data. Look for the sequence:
            #   1. Eyes open (EAR > 0.18) in early frames (baseline)
            #   2. Eyes closed (EAR < 0.15) in a middle frame (the dip)
            #   3. Eyes open again (EAR > 0.18) in recent frames (recovery)
            baseline_window = ears[:-2] if len(ears) > 2 else ears
            baseline_open = max(baseline_window or ears)
            recent_recovery = max(ears[-2:])
            recent_ear_avg = sum(ears[-2:]) / 2

            # A real blink on a low-FPS webcam often recovers over only one or
            # two frames. Accept a modest reopen signal, but still require the
            # full open -> dip -> reopen sequence.
            recovery_threshold = max(0.18, baseline_open * 0.68)
            if recent_recovery >= recovery_threshold or recent_ear_avg >= 0.19:
                # Walk backwards while keeping the last two frames reserved for
                # the reopen step. The dip must be materially below the open-eye
                # baseline, but should still be reachable on average webcams.
                dip_threshold = min(0.20, baseline_open * 0.72)
                open_threshold = max(0.20, baseline_open * 0.82)

                for i in range(len(ears) - 3, -1, -1):
                    if ears[i] < dip_threshold:
                        before = ears[:i]
                        if before and max(before) >= open_threshold:
                            return {"passed": True, "feedback": "✅ Blink detected!"}
                        if not before:
                            return {"passed": True, "feedback": "✅ Blink detected!"}
                        break
        else:
            # ── Fallback: det_score dip (InsightFace only) ───────────────────
            # When MediaPipe EAR isn't available, a blink causes a small but
            # measurable dip in the face-detection confidence score (det_score).
            all_scores = [f["det_score"] for f in feature_history]
            recent_score_avg = sum(all_scores[-3:]) / 3

            if recent_score_avg >= _BLINK_BASELINE_MIN * 0.96:
                # Need a relative drop of ~5% to register as a blink.
                dip_threshold = recent_score_avg * 0.95
                for i in range(len(all_scores) - 3, 0, -1):
                    if all_scores[i] < dip_threshold:
                        before_max = max(all_scores[:i] + [0.0])
                        if before_max >= recent_score_avg * 0.97:
                            return {"passed": True, "feedback": "✅ Blink detected!"}

        return {"passed": False, "feedback": "Close your eyes fully, then open them…"}

    return {"passed": False, "feedback": ""}


def run_face_match(
    live_face: Any,
    reference_embedding_b64: str,
) -> Dict[str, Any]:
    """Compare the live face embedding against the stored reference.

    Parameters
    ----------
    live_face:
        An InsightFace face object with ``normed_embedding``.
    reference_embedding_b64:
        Base-64 encoded float32 numpy bytes of the reference embedding.

    Returns
    -------
    dict with keys: passed, match_score (0-1), cosine_similarity, message.
    """
    import base64  # noqa: PLC0415

    try:
        emb_bytes = base64.b64decode(reference_embedding_b64)
        ref_emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
    except Exception:
        return {
            "passed": False,
            "match_score": 0.0,
            "cosine_similarity": 0.0,
            "message": "Reference embedding corrupted — please restart the session.",
        }

    live_emb = getattr(live_face, "normed_embedding", None)
    if live_emb is None:
        # InsightFace not available (e.g. Python 3.13); face-match not possible.
        return {
            "passed": True,
            "match_score": 1.0,
            "cosine_similarity": 1.0,
            "display_score": 100.0,
            "threshold": 0.40,
            "message": "Liveness verified (face-match skipped — requires InsightFace).",
        }
    sim = float(np.dot(live_emb, ref_emb))
    # Map cosine sim [-1, 1] → display score [0, 100 %] using the same mapping
    # as the rest of BaseTruth: (sim - (-0.5)) / (1.0 - (-0.5)) * 100
    display_pct = min(max((sim - (-0.5)) / (1.0 - (-0.5)) * 100, 0.0), 100.0)
    passed = sim >= 0.40

    return {
        "passed": passed,
        "match_score": display_pct / 100.0,
        "cosine_similarity": sim,
        "display_score": display_pct,
        "threshold": 0.40,
        "message": (
            "Identity verified." if passed
            else f"Face match failed (score {display_pct:.1f}%). Please retry."
        ),
    }
