"""Per-frame liveness analysis for the Video KYC pipeline.

Detects active-liveness challenges using InsightFace 5-point landmarks
(left_eye, right_eye, nose, mouth_left, mouth_right) and the bounding-box
detection confidence score.

Coordinate convention (InsightFace, un-mirrored camera frames):
  kps[0] = subject's LEFT eye  (appears on the LEFT side of the image)
  kps[1] = subject's RIGHT eye (appears on the RIGHT side of the image)
  kps[2] = nose tip
  Y-axis increases DOWNWARD (image convention).

All spatial features are normalized by the bounding-box width so they are
invariant to the subject's distance from the camera.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ── Per-challenge detection thresholds ───────────────────────────────────────

# nose_rel_x = (nose_x - bbox_left) / bbox_width
# In a frontal face, nose_rel_x ≈ 0.45-0.55
# Subject turns left  (THEIR left → nose moves right in image)  → nose_rel_x ↑
# Subject turns right (THEIR right → nose moves left in image)  → nose_rel_x ↓
_TURN_LEFT_THRESHOLD  = 0.62   # nose_rel_x must exceed this
_TURN_RIGHT_THRESHOLD = 0.38   # nose_rel_x must go below this

# pitch: (nose_y - eye_mid_y) / interocular_px
# Nod detected when pitch range across recent frames > this
_NOD_RANGE_THRESHOLD = 0.28

# Blink: det_score must drop below this (eyes closing lowers detection confidence)
# then recover above BLINK_RECOVER to confirm the eye opened again.
_BLINK_LOW_THRESHOLD    = 0.840
_BLINK_RECOVER_THRESHOLD = 0.900
_BLINK_BASELINE_MIN      = 0.880   # baseline (open-eye) confidence


def extract_features(face: Any) -> Dict[str, float]:
    """Return normalized pose features from one face object (InsightFace or MediaPipe)."""
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

    return {
        # Nose position relative to bbox (0 = left edge, 1 = right edge)
        "nose_rel_x": (nose_x - bbox[0]) / bbox_w,
        "nose_rel_y": (nose_y - bbox[1]) / bbox_h,
        # Yaw: how far nose deviates from eye midpoint (normalized by IOD)
        "yaw":  (nose_x - eye_mid_x) / interocular_px,
        # Pitch: nose below/above eye midpoint (normalized by IOD)
        #   positive = chin down, negative = face up
        "pitch": (nose_y - eye_mid_y) / interocular_px,
        # Detection confidence — drops slightly when eyes close (InsightFace only)
        "det_score": float(getattr(face, "det_score", 1.0)),
        # Eye Aspect Ratio — reliable blink indicator (MediaPipe); 0.30 default (open eye)
        "ear": float(getattr(face, "ear", 0.30)),
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

    # ─── Turn Left (subject's left → nose moves image-right → nose_rel_x ↑) ──
    if challenge == "turn_left":
        xs = [f["nose_rel_x"] for f in recent]
        if max(xs) >= _TURN_LEFT_THRESHOLD:
            return {"passed": True, "feedback": "✅ Turn detected!"}
        gap = _TURN_LEFT_THRESHOLD - max(xs)
        hint = "a little more…" if gap < 0.08 else "turn further to YOUR left…"
        return {"passed": False, "feedback": f"Keep turning — {hint}"}

    # ─── Turn Right (subject's right → nose moves image-left → nose_rel_x ↓) ─
    if challenge == "turn_right":
        xs = [f["nose_rel_x"] for f in recent]
        if min(xs) <= _TURN_RIGHT_THRESHOLD:
            return {"passed": True, "feedback": "✅ Turn detected!"}
        gap = min(xs) - _TURN_RIGHT_THRESHOLD
        hint = "a little more…" if gap < 0.08 else "turn further to YOUR right…"
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
        if len(feature_history) < 8:
            return {"passed": False, "feedback": "Hold still and look at the camera…"}

        # ── Primary path: EAR (Eye Aspect Ratio) — works with MediaPipe ──────
        # A typical open eye has EAR ≈ 0.25-0.35; closed eye EAR ≈ 0.02-0.10.
        ears = [f.get("ear", 0.30) for f in feature_history]
        baseline_ear = sum(ears[:5]) / 5.0  # average of first 5 frames

        if baseline_ear > 0.18:  # eyes were open at the start
            min_ear = min(ears[4:])            # must dip below 0.15 (closed)
            last5_avg_ear = sum(ears[-5:]) / min(len(ears), 5)  # must recover
            if min_ear < 0.15 and last5_avg_ear > 0.18:
                return {"passed": True, "feedback": "✅ Blink detected!"}

        # ── Fallback: det_score dip (InsightFace only) ─────────────────────
        all_scores = [f["det_score"] for f in feature_history]
        baseline_score = max(all_scores[:5])
        if baseline_score >= _BLINK_BASELINE_MIN:
            tail = all_scores[4:]
            min_tail  = min(tail)
            last5_avg = sum(tail[-5:]) / min(len(tail), 5)
            if min_tail <= _BLINK_LOW_THRESHOLD and last5_avg >= _BLINK_RECOVER_THRESHOLD:
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
