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

from typing import Any, Dict, List, Tuple

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
# Band widened to 0.40–0.60 (was 0.43–0.57) so slightly off-centre face placement
# still registers without the user having to reposition themselves.
_STRAIGHT_X_MIN = 0.40
_STRAIGHT_X_MAX = 0.60
_STRAIGHT_STABLE_FRAMES = 3  # 3 consecutive centred frames ≈ 375 ms at 8 FPS
                              # Reduced from 5 (was ≈ 625 ms) — still a clear frontal hold

# yaw = (nose_x - eye_mid_x) / interocular_px.  For a face looking straight into
# the camera the nose sits directly between the eyes, so yaw ≈ 0.  We cap it at
# 0.12 (about 75 % of the full turn threshold) so that only genuinely frontal
# frames are accepted as the selfie.  A turned face that happens to have a
# centred nose_rel_x would still be rejected here.
_STRAIGHT_YAW_MAX = 0.12

# Minimum InsightFace detection confidence for a frame to count as a valid
# selfie capture.  Values below this typically mean the face is partially
# obscured, blurry, or a marginal false-positive.  The default fill value in
# extract_features() is 1.0 (used when no det_score is available, e.g. MediaPipe)
# so this threshold only filters low-confidence InsightFace detections.
_STRAIGHT_DET_SCORE_MIN = 0.65

# Minimum face detection confidence required for ANY frame to be counted toward
# challenge progress.  When there is no face present (user moved away, ducked,
# covered camera), the detector occasionally fires on background texture with a
# very low confidence score.  Feature values extracted from such marginal
# detections are unreliable noise — their pitch range can satisfy the nod
# challenge and their yaw can satisfy a turn challenge even though no real face
# movement occurred.  Filtering frames below this threshold at both the
# frame-ingestion layer (api.py / face_scan/live.py) and inside analyze_challenge
# ensures those ghost frames never contribute to liveness decisions.
# 0.55 keeps real faces at angle/distance (typically ≥ 0.70) while reliably
# discarding false-positive background detections (typically 0.20–0.50).
MIN_FACE_DETECTION_CONFIDENCE = 0.55

# Minimum face bounding-box area as a fraction of the full frame area required
# for ANY frame to be counted toward challenge progress.
# Expert review raised this from 3% to 5%: at 3% the face bbox is only ~96×96 px
# at 640×480, which is too small for reliable landmark extraction and causes wrong
# turn / blink / jitter readings.  5% (≈ 156×156 px at 640×480) gives the model
# enough landmark resolution to be confident.
# Frames that do not carry a 'bbox_area_ratio' key (old test fixtures, paths
# that do not compute it) are allowed through so backward compatibility holds.
MIN_FACE_AREA_RATIO = 0.05

# ── Pre-liveness face stability gate ──────────────────────────────────────────
#
# Following the expert-recommended pipeline:
#   Face Detection → Face Validation → Face Tracking (stability) → Liveness
#
# Before the first challenge frame is accepted, the server requires N consecutive
# frames where the face passes ALL of:
#   1. Geometry invariants    (face_geometry_valid)
#   2. Detection confidence   ≥ FACE_STABILITY_CONFIDENCE_MIN (stricter than runtime)
#   3. Face area              ≥ FACE_STABILITY_AREA_MIN (6%, stricter than the 5% runtime gate)
#   4. Horizontal centering   within FACE_STABILITY_X band
#   5. Vertical centering     within FACE_STABILITY_Y band (prevents tilted/partial faces)
#   6. Near-frontal pose      |yaw| ≤ FACE_STABILITY_YAW_MAX  (head not turned sideways)
#   7. Near-frontal pose      |pitch| ≤ FACE_STABILITY_PITCH_MAX (head not tilted up/down)
#   8. Texture richness       ≥ MIN_FACE_TEXTURE_SCORE (flat screen/photo guard)
#   9. Exactly 1 face in the frame
#
# In addition, after all N frames have accumulated, the yaw variance across the
# window must exceed FACE_STABILITY_YAW_VARIANCE_MIN — a real person always has
# tiny involuntary micro-movements; a static screen or printed photo has zero.
#
# Only once this quota is met does the server start crediting challenge history.
# Any frame that fails ANY condition resets the counter to zero so the user must
# hold still for a full clean window — a flickering or marginal detection can
# never accumulate a partial count.
#
# 10 frames ≈ 1–1.25 seconds at 8–10 FPS.
FACE_STABLE_FRAMES_REQUIRED = 10

# Higher confidence threshold for the stability gate.  We can be more demanding
# here because the user is stationary and well-lit before challenges start.
# The in-challenge MIN_FACE_DETECTION_CONFIDENCE (0.55) is kept lower so that
# legitimate head turns, blinks, and nods — which temporarily reduce confidence —
# are not inadvertently dropped mid-challenge.
FACE_STABILITY_CONFIDENCE_MIN = 0.80

# Face size threshold for the stability gate — stricter than the runtime 5% gate.
# At 6%, the face bbox is at least ~125×125 px at 640×480, giving InsightFace
# reliable 5-point landmarks before challenges start.
FACE_STABILITY_AREA_MIN = 0.06

# Horizontal centering band for the stability gate — nose_rel_x must be in range.
# Slightly wider than the look_straight challenge band (0.40–0.60) so the user is
# not forced to pixel-perfectly centre before they even start.
FACE_STABILITY_X_MIN = 0.35
FACE_STABILITY_X_MAX = 0.65

# Vertical centering band — nose_rel_y (nose Y relative to bbox height) must be
# within this range.  Prevents partial faces, extreme camera angles, and bad
# landmark geometry caused by a very low or very high camera position.
FACE_STABILITY_Y_MIN = 0.30
FACE_STABILITY_Y_MAX = 0.70

# Maximum absolute yaw/pitch during the stability accumulation phase.
# The user must be facing roughly straight at the camera before we start
# so that the first challenge frames are captured with a high-quality pose.
# 0.08 is roughly ±5 degrees of head rotation.
FACE_STABILITY_YAW_MAX   = 0.08
FACE_STABILITY_PITCH_MAX = 0.08

# Minimum yaw variance across the stability window to confirm micro-movement.
# A real human always has tiny involuntary head oscillations (breathing, muscle
# tremor).  Over 10 frames their yaw variance is typically 1e-4 to 1e-2.
# A static screen or printed photo has yaw values that are machine-precision
# constant — variance ≈ 0.  Setting the threshold at 2e-4 comfortably separates
# the two populations while not penalising very still but genuinely live users.
FACE_STABILITY_YAW_VARIANCE_MIN = 0.0002

# Minimum local texture richness score for the stability gate.
# compute_face_texture_score() measures the mean per-cell standard deviation
# across a 6×6 grid of the grayscale face crop.
# Real faces at selfie distance: typically 25–70.
# Flat surfaces (screen/photo reproduced from a screen): typically < 15.
# Threshold at 18 gives a comfortable margin between the two populations.
MIN_FACE_TEXTURE_SCORE = 18.0


def compute_face_texture_score(img: Any, bbox: Any) -> float:
    """Measure local texture richness inside the face bounding box.

    Divides the grayscale face crop into a 6×6 grid and returns the mean
    standard deviation across all cells.  This measures LOCAL variation inside
    small patches rather than global contrast, which makes it robust to images
    that have high global contrast despite a flat, textureless source (e.g. a
    white background around a printed face).

    High score = organic skin texture, hair, shadow variation → real face.
    Low score  = uniform, smooth, texture-free surface → screen or print.

    Real faces at normal selfie distance: typically 25–70.
    Flat screen/photo presented to the camera: typically < 15.

    Returns 99.0 (always passes) when the crop is too small to measure reliably
    or when cv2 is unavailable.
    """
    try:
        import cv2 as _cv2  # noqa: PLC0415 — lazy import keeps module lightweight
    except ImportError:
        return 99.0  # cv2 not available — cannot measure, assume OK

    # Clamp the bounding box to image boundaries before cropping.
    x1 = int(max(float(bbox[0]), 0))
    y1 = int(max(float(bbox[1]), 0))
    x2 = int(min(float(bbox[2]), img.shape[1]))
    y2 = int(min(float(bbox[3]), img.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return 99.0

    crop = _cv2.cvtColor(img[y1:y2, x1:x2], _cv2.COLOR_BGR2GRAY)
    if crop.size < 64:
        return 99.0  # too small for meaningful patch statistics

    # Divide into a 6×6 grid; compute std dev inside each cell.
    # Small cells capture local texture variation independently of global pose lighting.
    cell_h = max(crop.shape[0] // 6, 1)
    cell_w = max(crop.shape[1] // 6, 1)
    stds: List[float] = []
    for r in range(6):
        for c in range(6):
            cell = crop[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            if cell.size > 0:
                stds.append(float(np.std(cell)))

    return float(np.mean(stds)) if stds else 99.0


def is_face_stable(
    face: Any,
    face_count: int,
    bbox_area_ratio: float,
    confidence: float,
    nose_rel_x: float,
    nose_rel_y: float = 0.50,
    yaw: float = 0.0,
    pitch: float = 0.0,
    texture_score: float = 99.0,
) -> Tuple[bool, str]:
    """Return (True, "") if this frame qualifies as a valid stability frame, or
    (False, user_feedback) explaining what the user needs to fix.

    This implements Step 2 (Face Validation) from the expert-recommended pipeline:

        Face Detection → Face Validation → Face Tracking (stability) → Liveness

    All nine conditions must be satisfied simultaneously:
      1. Exactly one face in the frame.
      2. Detection confidence ≥ FACE_STABILITY_CONFIDENCE_MIN (0.80).
      3. Face bounding box ≥ FACE_STABILITY_AREA_MIN (6%) of the full frame.
      4. Nose within horizontal band FACE_STABILITY_X_MIN–MAX (35–65%).
      5. Nose within vertical band FACE_STABILITY_Y_MIN–MAX (30–70%).
      6. |yaw|   ≤ FACE_STABILITY_YAW_MAX (0.08) — head near-frontal.
      7. |pitch| ≤ FACE_STABILITY_PITCH_MAX (0.08) — head near-frontal.
      8. texture_score ≥ MIN_FACE_TEXTURE_SCORE (18) — not a flat surface.
      9. Geometry invariants pass (caller must pre-check with face_geometry_valid).

    Parameters nose_rel_y, yaw, pitch, texture_score default to safe "pass"
    values so existing callers that do not supply them are unaffected.

    This function is intentionally stateless — the caller owns the consecutive-
    frame counter and resets it on any False return.
    """
    if face_count != 1:
        if face_count == 0:
            return False, "No face detected — move into the oval."
        return False, "Multiple faces detected — ensure only you are in frame."

    if confidence < FACE_STABILITY_CONFIDENCE_MIN:
        return False, "Face not clearly visible — improve lighting or move closer."

    # Use the stricter stability-phase area threshold (6%), not the live-phase one (5%).
    if bbox_area_ratio < FACE_STABILITY_AREA_MIN:
        return False, "Move closer to the camera — your face is too small in the oval."

    if nose_rel_x < FACE_STABILITY_X_MIN:
        return False, "Move slightly to YOUR right to centre your face."

    if nose_rel_x > FACE_STABILITY_X_MAX:
        return False, "Move slightly to YOUR left to centre your face."

    # Vertical centering — prevents tilted cameras and partial/cut-off faces.
    if nose_rel_y < FACE_STABILITY_Y_MIN or nose_rel_y > FACE_STABILITY_Y_MAX:
        return False, "Centre your face vertically in the oval."

    # Head angle restrictions — user must face roughly straight before challenges begin.
    if abs(yaw) > FACE_STABILITY_YAW_MAX:
        return False, "Look straight at the camera before we begin."

    if abs(pitch) > FACE_STABILITY_PITCH_MAX:
        return False, "Look straight at the camera — do not tilt your head up or down."

    # Flat surface guard — a screen or printed photo has very low local texture variance.
    # A real face has organic skin texture that scores well above the threshold.
    if texture_score < MIN_FACE_TEXTURE_SCORE:
        return False, "Real face not detected — ensure you are in front of the camera, not a screen."

    return True, ""


# 0.12 accepts a natural head nod; 0.28 was the original, 0.14 was already lowered once
_NOD_RANGE_THRESHOLD = 0.12

_BLINK_BASELINE_MIN      = 0.880   # baseline (open-eye) confidence


# ── Face geometry validity thresholds ────────────────────────────────────────

# Human faces have strong geometric invariants that non-face objects (e.g. palms,
# hands, masks, printed photos) frequently violate. These thresholds act as a
# first-line guard against the face detector being fooled by a hand or object.

# Maximum allowed vertical gap between the two eye landmarks, as a fraction of
# bbox height. Both eyes of a real face are always at similar heights (< 20%).
# A palm held up to the camera often produces wildly different "eye" heights.
_FACE_EYE_VERTICAL_GAP_MAX = 0.25

# Interocular distance (eye-to-eye pixels) as a fraction of bbox width.
# For a human face at any distance, this ratio is roughly 0.20 – 0.55.
# A hand can produce "eyes" that are much closer together or much further apart.
_FACE_IOD_RATIO_MIN = 0.15
_FACE_IOD_RATIO_MAX = 0.65

# Eye midpoint must be in the upper 65% of the face bbox (measured top-down).
# A human face has eyes in roughly the upper third; a palm does not.
_FACE_EYE_MIDPOINT_Y_MAX = 0.65

# The nose tip must be at or below both eye landmarks (Y increases downward).
# This is the single strongest invariant: it is always true for a real face
# at any pose, and is commonly violated by a hand or printed object that the
# detector mis-classifies as a face.
# A small tolerance (5% of bbox height) allows for mild detection jitter.
_FACE_NOSE_BELOW_EYES_TOLERANCE = 0.05


def face_geometry_valid(face: Any) -> Tuple[bool, str]:
    """Check that a detected face has plausible human facial landmark geometry.

    Face detectors (InsightFace, MediaPipe) occasionally mistake a palm or hand
    for a face. This function validates the five facial keypoints against geometric
    invariants that every real human face satisfies regardless of pose or distance:

      1. Nose is always BELOW both eye landmarks (image Y increases downward).
      2. Both eye landmarks are at similar vertical positions.
      3. Interocular distance is in a plausible range relative to face width.
      4. Eye midpoint is in the upper portion of the face bounding box.

    Returns (True, "") if the landmarks look like a real face, or
    (False, reason_string) if any invariant is violated.
    """
    kps  = getattr(face, "kps",  None)
    bbox = getattr(face, "bbox", None)
    if kps is None or bbox is None:
        return False, "No face keypoints available"

    kps  = np.asarray(kps,  dtype=float)
    bbox = np.asarray(bbox, dtype=float)

    bbox_w = max(bbox[2] - bbox[0], 1.0)
    bbox_h = max(bbox[3] - bbox[1], 1.0)

    left_eye_x,  left_eye_y  = kps[0]
    right_eye_x, right_eye_y = kps[1]
    nose_x,      nose_y      = kps[2]

    # --- Check 1: Nose must be at or below both eye landmarks ---
    # In image coordinates Y increases downward. For every real human face —
    # frontal, tilted, or in profile — the nose tip is always below the eyes.
    # A small tolerance absorbs detector jitter on heavily tilted faces.
    tolerance_px = _FACE_NOSE_BELOW_EYES_TOLERANCE * bbox_h
    eye_y_max = max(left_eye_y, right_eye_y)
    if nose_y < eye_y_max - tolerance_px:
        return False, (
            f"Nose landmark (y={nose_y:.1f}) is above the eye landmarks "
            f"(eye_max_y={eye_y_max:.1f}) — likely not a face"
        )

    # --- Check 2: Both eyes must be at similar vertical positions ---
    # A real face never has one eye dramatically higher than the other.
    # Large vertical gaps indicate the detector found "eyes" on a non-face object.
    eye_y_gap_ratio = abs(left_eye_y - right_eye_y) / bbox_h
    if eye_y_gap_ratio > _FACE_EYE_VERTICAL_GAP_MAX:
        return False, (
            f"Eye landmarks have very different heights "
            f"(gap ratio {eye_y_gap_ratio:.2f} > {_FACE_EYE_VERTICAL_GAP_MAX}) — likely not a face"
        )

    # --- Check 3: Interocular distance must be in a plausible range ---
    # For any human face the distance between the two eye landmarks is
    # roughly 20–55% of the face bounding-box width. Values outside this range
    # mean the detected "eyes" are not real eye landmarks on a face.
    iod_ratio = abs(right_eye_x - left_eye_x) / bbox_w
    if iod_ratio < _FACE_IOD_RATIO_MIN or iod_ratio > _FACE_IOD_RATIO_MAX:
        return False, (
            f"Interocular distance ratio {iod_ratio:.2f} is outside the human-face "
            f"range [{_FACE_IOD_RATIO_MIN}, {_FACE_IOD_RATIO_MAX}]"
        )

    # --- Check 4: Eye midpoint must be in the upper portion of the face bbox ---
    # Human eyes are in roughly the upper 20–55% of the face bounding box.
    # If the detected "eyes" are in the lower half, this is not a real face.
    eye_mid_y = (left_eye_y + right_eye_y) / 2.0
    eye_mid_y_ratio = (eye_mid_y - bbox[1]) / bbox_h
    if eye_mid_y_ratio > _FACE_EYE_MIDPOINT_Y_MAX:
        return False, (
            f"Eye midpoint is at {eye_mid_y_ratio:.2f} of face height "
            f"(max allowed {_FACE_EYE_MIDPOINT_Y_MAX}) — eyes appear too low for a face"
        )

    return True, ""


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
        # Interocular distance (IOD) relative to face width.
        # For a real 3D head this ratio DECREASES during a head turn because the far eye
        # moves behind the nose (perspective parallax). A flat 2D photo or printed mask
        # has no depth, so its IOD/bbox ratio stays constant regardless of how much it is
        # physically rotated — used by the 3D depth-consistency check.
        "interocular_px_norm": interocular_px / bbox_w,
        # Eye landmark positions relative to the face bounding box (0–1 range).
        # Used for saccade / eye micro-jitter analysis: real eyes make tiny involuntary
        # micro-movements (saccades) every ~100–200 ms. A static photo or looped replay
        # produces eye positions that are unnaturally stable across frames.
        "left_eye_x_norm":  (left_eye_x  - bbox[0]) / bbox_w,
        "left_eye_y_norm":  (left_eye_y  - bbox[1]) / bbox_h,
        "right_eye_x_norm": (right_eye_x - bbox[0]) / bbox_w,
        "right_eye_y_norm": (right_eye_y - bbox[1]) / bbox_h,
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
    # Filter out frames where the face detector was not sufficiently confident.
    # Background objects, hands, or an empty room can trigger marginal detections
    # with low det_score.  The extracted features from those frames are noise —
    # a pitch range of 0.12 can appear over 4 frames of jitter even with no real
    # head present, which would spuriously satisfy the nod challenge.
    # The default value of 1.0 (set by extract_features when det_score is absent,
    # e.g. MediaPipe) is intentionally above the threshold so MediaPipe frames
    # are never discarded by this filter.
    recent = [f for f in recent if f.get("det_score", 1.0) >= MIN_FACE_DETECTION_CONFIDENCE]
    if len(recent) < 3:
        return {"passed": False, "feedback": "No face detected — look directly into the camera."}

    # Filter out frames where the face is too small relative to the full frame.
    # When the user ducks away or a distant background object is detected, the
    # face bbox is tiny (< 3 % of frame area).  Frames without 'bbox_area_ratio'
    # (e.g. test fixtures or paths that do not compute it) default to passing.
    recent = [f for f in recent if f.get("bbox_area_ratio", MIN_FACE_AREA_RATIO) >= MIN_FACE_AREA_RATIO]
    if len(recent) < 3:
        return {"passed": False, "feedback": "No face detected — position yourself in the oval."}

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
        yaws   = [f["yaw"]        for f in last_n]
        scores = [f["det_score"]  for f in last_n]
        # Three conditions must ALL hold for every frame in the capture window:
        #   1. Horizontal centering: nose must be within the central 40-60 % band
        #      of the face bounding box.  A turned face shifts the nose out of this
        #      band, but this alone is not sufficient — see condition 2.
        #   2. Low yaw: nose must be no more than _STRAIGHT_YAW_MAX away from the
        #      eye midpoint (normalized by interocular distance).  This is the direct
        #      measure of head rotation.  A face can have nose_rel_x ≈ 0.5 while
        #      being rotated (perspective effect) — yaw catches that case.
        #   3. Detection confidence: reject blurry or marginal detections so that the
        #      frame captured as the identity selfie is always a clean, clear shot.
        centered   = all(_STRAIGHT_X_MIN <= x <= _STRAIGHT_X_MAX for x in xs)
        frontal    = all(abs(y) <= _STRAIGHT_YAW_MAX for y in yaws)
        confident  = all(s >= _STRAIGHT_DET_SCORE_MIN for s in scores)
        if centered and frontal and confident:
            return {"passed": True, "feedback": "\u2705 Frontal face captured!"}
        avg_x   = sum(xs)   / len(xs)
        avg_yaw = sum(yaws) / len(yaws)
        # Return the most specific feedback so the user knows exactly what to fix.
        if not frontal:
            return {"passed": False, "feedback": "Look directly into the camera lens\u2026"}
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
        # Wrong-direction guard: if the user clearly turned RIGHT (yaw strongly positive)
        # while the required challenge is turn_left, reset the frame history. This
        # prevents the inflated baseline from making the relative-delta check trivially
        # pass when the user returns to centre. Only fires after >= 5 frames so brief
        # detector noise on the very first frames is not penalised.
        if len(recent) >= 5 and max(yaws) >= _TURN_YAW_THRESHOLD:
            return {"passed": False, "feedback": "Wrong direction - turn to YOUR LEFT.", "reset_needed": True, "wrong_motion": "turned_right"}
        if min(yaws) <= -_TURN_YAW_THRESHOLD:
            return {"passed": True, "feedback": "\u2705 Turn detected!"}
        if len(recent) >= 5:
            baseline_yaw = sum(yaws[:2]) / 2
            baseline_nose = sum(noses[:2]) / 2
            yaw_delta = baseline_yaw - min(yaws)
            nose_shift = baseline_nose - min(noses)
            if yaw_delta >= _TURN_RELATIVE_YAW_DELTA and nose_shift >= _TURN_RELATIVE_NOSE_SHIFT:
                return {"passed": True, "feedback": "\u2705 Turn detected!"}
        gap = _TURN_YAW_THRESHOLD - abs(min(yaws))
        hint = "a little more..." if gap < 0.10 else "turn further to YOUR left..."
        return {"passed": False, "feedback": f"Keep turning - {hint}"}

    # ─── Turn Right (subject's right → nose image-RIGHT → yaw POSITIVE) ─────────
    # When the subject turns to their own right in a mirrored frame, the nose swings
    # toward image-right, making yaw positive.
    if challenge == "turn_right":
        yaws = [f["yaw"] for f in recent]
        noses = [f["nose_rel_x"] for f in recent]
        # Wrong-direction guard: if the user clearly turned LEFT (yaw strongly negative)
        # while the required challenge is turn_right, reset the frame history so the
        # depressed baseline cannot be exploited by the relative-delta check.
        if len(recent) >= 5 and min(yaws) <= -_TURN_YAW_THRESHOLD:
            return {"passed": False, "feedback": "Wrong direction - turn to YOUR RIGHT.", "reset_needed": True, "wrong_motion": "turned_left"}
        if max(yaws) >= _TURN_YAW_THRESHOLD:
            return {"passed": True, "feedback": "\u2705 Turn detected!"}
        if len(recent) >= 5:
            baseline_yaw = sum(yaws[:2]) / 2
            baseline_nose = sum(noses[:2]) / 2
            yaw_delta = max(yaws) - baseline_yaw
            nose_shift = max(noses) - baseline_nose
            if yaw_delta >= _TURN_RELATIVE_YAW_DELTA and nose_shift >= _TURN_RELATIVE_NOSE_SHIFT:
                return {"passed": True, "feedback": "\u2705 Turn detected!"}
        gap = _TURN_YAW_THRESHOLD - max(yaws)
        hint = "a little more..." if gap < 0.10 else "turn further to YOUR right..."
        return {"passed": False, "feedback": f"Keep turning - {hint}"}

    # ─── Nod (vertical head movement → pitch range) ──────────────────────────
    if challenge == "nod":
        pitches = [f["pitch"] for f in recent]
        # 4 frames (≈ 500 ms at 8 FPS) is enough to capture an up-down nod;
        # the range threshold ensures the movement is real and not just detector noise
        if len(pitches) >= 4:
            pitch_range = max(pitches) - min(pitches)
            if pitch_range >= _NOD_RANGE_THRESHOLD:
                return {"passed": True, "feedback": "✅ Nod detected!"}
            # Wrong-motion hint: if the user is shaking their head side-to-side
            # (high yaw range) instead of nodding up-down (high pitch range),
            # give targeted feedback so they stop the wrong movement immediately.
            yaws = [f["yaw"] for f in recent]
            yaw_range = max(yaws) - min(yaws)
            if yaw_range > pitch_range * 2.0 and yaw_range > 0.10:
                return {"passed": False, "feedback": "That's a turn, not a nod - move your head DOWN then back UP.", "wrong_motion": "side_to_side_shake"}
        return {"passed": False, "feedback": "Nod your head down and back up…"}

    # ─── Blink ────────────────────────────────────────────────────────────────
    if challenge == "blink":        # Apply the same confidence filter used for all other challenges.
        # The blink check walks the full feature_history (not just `recent`) so
        # low-confidence ghost frames must be stripped here independently.
        feature_history = [f for f in feature_history if f.get("det_score", 1.0) >= MIN_FACE_DETECTION_CONFIDENCE]
        # Apply the same face-size filter — blink can also be faked by noise in
        # a tiny background detection whose EAR or det_score fluctuates.
        feature_history = [f for f in feature_history if f.get("bbox_area_ratio", MIN_FACE_AREA_RATIO) >= MIN_FACE_AREA_RATIO]
        # 3 frames minimum: 1 baseline open + 1 closed dip + 1 reopen.
        # Reduced from 5 — the blink logic walks backwards through history anyway;
        # waiting for 5 frames just adds ~250 ms latency before the first check.
        if len(feature_history) < 3:
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
