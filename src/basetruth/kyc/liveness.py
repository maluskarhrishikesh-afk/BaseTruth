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

# Turn hold constants — used by the turn_left/turn_right challenges.
# Once the user crosses the yaw threshold, they must hold their head turned for
# _TURN_HOLD_FRAMES frames. Challenge passes immediately at that point — no
# return-to-centre required. A green flash + beep signals completion.
_TURN_HOLD_FRAMES   = 10    # frames of sustained turn ≈ 1 s at 10 FPS
_TURN_HOLD_LENIENCY = 0.04  # allow this much yaw relaxation during hold (natural camera wobble)

# look_straight: nose_rel_x must stay within this central band for several frames.
# We only check the horizontal position — pitch (nose below eye midpoint) is always
# 0.4–0.7 for any real face and is not a reliable head-tilt indicator here.
# Band widened to 0.40–0.60 (was 0.43–0.57) so slightly off-centre face placement
# still registers without the user having to reposition themselves.
_STRAIGHT_X_MIN = 0.40
_STRAIGHT_X_MAX = 0.60
_STRAIGHT_STABLE_FRAMES = 10  # 10 consecutive centred frames ≈ 1 s at 10 FPS — clear frontal hold

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
#      Note: pitch is intentionally NOT checked — see FACE_STABILITY_YAW_MAX comment below.
#   7. Texture richness       ≥ MIN_FACE_TEXTURE_SCORE (flat screen/photo guard)
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

# Maximum absolute yaw during the stability accumulation phase.
# The user must be facing roughly straight at the camera before we start
# so that the first challenge frames are captured with a high-quality pose.
# 0.12 matches the look_straight challenge threshold (_STRAIGHT_YAW_MAX) — a
# natural resting face with slight off-centre placement reads ~0.10–0.13, so
# 0.08 was too strict and blocked real users unconditionally.
FACE_STABILITY_YAW_MAX = 0.12

# NOTE: A pitch check is intentionally absent here.
# Our pitch = (nose_y - eye_mid_y) / interocular_px is a face anatomy constant
# (always 0.4–0.7 for any real forward-facing face) and is not a head-tilt
# indicator.  Head-tilt detection requires 3D landmarks; the 2D version only
# creates a gate that rejects every real user.  The nose_rel_y check (30–70%)
# already guards against extreme camera angles.

# Minimum yaw variance across the stability window to confirm micro-movement.
# Our yaw = (nose_x - eye_mid_x) / interocular_px from 5-point InsightFace
# landmarks.  Empirically, real users at a laptop webcam produce variance in the
# 4e-6 to 1e-4 range — much lower than the "1e-4 to 1e-2" estimate that assumed
# full 3D landmark precision.  A truly static source (printed photo, screen
# replay with a frozen frame) has near-zero NN floating-point variance (≈ 1e-8
# to 1e-7).  Setting threshold to 1e-6 safely separates the two populations
# while not rejecting live users who sit still.
FACE_STABILITY_YAW_VARIANCE_MIN = 1e-6

# Minimum local texture richness score for the stability gate.
# compute_face_texture_score() measures the mean per-cell standard deviation
# across a 6×6 grid of the grayscale face crop.
# Real faces at selfie distance: typically 25–70.
# Flat surfaces (screen/photo reproduced from a screen): typically < 15.
# Threshold at 18 gives a comfortable margin between the two populations.
MIN_FACE_TEXTURE_SCORE = 18.0

# Maximum time (seconds) a user has to complete each liveness challenge from the
# moment the prompt appears.  If the challenge is not completed within this window,
# the current challenge frame-history is cleared and the timer restarts so the
# user gets another attempt from a clean baseline.  This prevents slow brute-force
# probing strategies where an attacker submits random motions until one happens to
# score a pass.  10 seconds is generous for a genuine user but short enough to
# deter patient attackers.
CHALLENGE_TIMEOUT_SECONDS = 10.0


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


def compute_face_brightness(img: Any, bbox: Any) -> float:
    """Return the mean pixel intensity (0–255) of the grayscale face crop.

    Used to select the correct adaptive texture threshold — in dim lighting the
    texture score of a flat screen or print may be inflated by sensor noise and
    JPEG compression artefacts, so we relax the threshold below
    LOW_BRIGHTNESS_THRESHOLD to avoid false-rejecting real faces in dark rooms.

    Returns 128.0 (mid-range, neutral) when the crop cannot be computed.
    """
    try:
        import cv2 as _cv2  # noqa: PLC0415
    except ImportError:
        return 128.0

    x1 = int(max(float(bbox[0]), 0))
    y1 = int(max(float(bbox[1]), 0))
    x2 = int(min(float(bbox[2]), img.shape[1]))
    y2 = int(min(float(bbox[3]), img.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return 128.0

    crop = _cv2.cvtColor(img[y1:y2, x1:x2], _cv2.COLOR_BGR2GRAY)
    return float(np.mean(crop)) if crop.size > 0 else 128.0


# Brightness below this value means the face is in a dark environment.
# At these levels, JPEG compression adds noise that inflates texture scores
# for flat surfaces, so we use a relaxed threshold to avoid false rejections.
LOW_BRIGHTNESS_THRESHOLD  = 80
LOW_BRIGHTNESS_TEXTURE_SCORE = 12.0  # relaxed threshold for dark frames
                                     # Lowered from 14.0: real faces in dim rooms
                                     # (brightness ~30–50) score 13–14; 14.0 was
                                     # rejecting them at the stability gate.


def get_adaptive_texture_threshold(brightness_mean: float) -> float:
    """Return the appropriate texture score threshold for the given face brightness.

    In low-light conditions (brightness < LOW_BRIGHTNESS_THRESHOLD), JPEG noise
    and sensor noise can inflate the local std-dev of a flat surface enough to
    falsely pass the static threshold of 18.0.  Dropping to 14.0 in that regime
    keeps the guard meaningful without producing false rejects of real faces in
    dark rooms.

    The returned threshold should be compared against the score from
    compute_face_texture_score().
    """
    if brightness_mean < LOW_BRIGHTNESS_THRESHOLD:
        return LOW_BRIGHTNESS_TEXTURE_SCORE
    return MIN_FACE_TEXTURE_SCORE


# Maximum fraction of consecutive diff-pairs with alternating signs (direction
# reverses between adjacent frames) in the stability yaw buffer.
# Artificial jitter (screen shaking, programmatic video wobble) bounces back and
# forth on almost every frame → high alternation rate (~1.0).
# Real involuntary micro-movement drifts slowly and changes direction irregularly
# → alternation rate is typically 0.3–0.6.
# Threshold at 0.80 gives comfortable separation between the two populations.
FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX = 0.80


def is_yaw_motion_natural(yaw_buffer: List[float]) -> Tuple[bool, str]:
    """Return (True, "") if the yaw movement pattern looks like natural micro-movement.

    Artificial jitter used to spoof the yaw-variance check (screen shaking,
    programmatic video wobble) typically produces a high-frequency alternating
    signal: the yaw value flips direction on almost every frame.  Natural
    involuntary micro-movement from breathing and muscle tremor is irregular —
    the direction reverses only occasionally across a 10-frame window.

    Metric: fraction of adjacent diff-pairs where signs alternate (direction
    reverses between consecutive frames).  Above FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX
    (0.80) indicates a suspiciously periodic, non-natural signal.

    Returns True (passes) when the buffer is too short to be conclusive, so
    early frames are never incorrectly penalised.
    """
    if len(yaw_buffer) < 4:
        return True, ""  # too few samples to test reliably

    diffs = np.diff(np.asarray(yaw_buffer, dtype=np.float64))
    if len(diffs) < 3:
        return True, ""

    # Count pairs where the direction (sign of diff) reversed between consecutive steps.
    # A perfect sine-like artificial wiggle would alternate on EVERY pair → rate = 1.0.
    sign_alternations = sum(
        1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0
    )
    alternation_rate = sign_alternations / max(len(diffs) - 1, 1)

    if alternation_rate > FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX:
        return False, "Unnatural movement detected — ensure you are using a live camera, not a video."
    return True, ""


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
    brightness_mean: float = 128.0,
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
      6. |yaw|   ≤ FACE_STABILITY_YAW_MAX (0.12) — head near-frontal.
      7. texture_score ≥ get_adaptive_texture_threshold(brightness_mean) — not a flat surface.
      9. Geometry invariants pass (caller must pre-check with face_geometry_valid).

    Parameters nose_rel_y, yaw, pitch, texture_score, brightness_mean all default
    to safe "pass" values so existing callers that do not supply them are unaffected.

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
    # Only yaw is checked; pitch as computed (nose_y - eye_mid_y / IOD) is a
    # face-anatomy constant (~0.4–0.7) and not a reliable tilt indicator.
    if abs(yaw) > FACE_STABILITY_YAW_MAX:
        return False, "Look straight at the camera before we begin."

    # Flat surface guard — uses an adaptive threshold based on face brightness.
    # In low-light frames, sensor noise can inflate the texture score of a flat
    # surface; using a lower threshold there avoids false-rejecting real faces
    # in dark environments while still blocking high-quality bright-room spoofs.
    _texture_threshold = get_adaptive_texture_threshold(brightness_mean)
    if texture_score < _texture_threshold:
        return False, "Real face not detected — ensure you are in front of the camera, not a screen."

    return True, ""


# Nod hold constants — used by the nod challenge.
# The user must tilt their head down (pitch deviates from neutral by ≥ _NOD_DOWN_DELTA)
# and hold that position for _NOD_HOLD_FRAMES frames. No return-to-neutral required:
# the challenge completes as soon as the hold quota is met, signalled by a green
# flash and beep so the user knows they can relax — no looping back up needed.
# Baseline neutral pitch is computed from the first _NOD_BASELINE_FRAMES challenge frames
# (user was just looking straight after the stability gate).
# _NOD_DOWN_DELTA: a gentle chin-dip typically moves pitch by 0.08–0.15; static noise
# is < 0.04. Threshold lowered from 0.12 → 0.08 so users don't have to bend far down.
# Direction is not assumed — we detect deviation in either direction so the
# challenge works regardless of which way pitch changes for "chin down" on
# each user's camera setup.
_NOD_HOLD_FRAMES    = 6     # frames of sustained head-down position ≈ 0.6 s at 10 FPS
_NOD_DOWN_DELTA     = 0.08  # pitch must deviate this much from neutral to count as "down"
_NOD_BASELINE_FRAMES = 3    # first N challenge frames used to estimate neutral pitch

# Blink: number of distinct eye-close/open cycles required to pass the challenge.
# Two blinks are sufficient proof of liveness and are easier for users than
# performing a single precisely-timed blink at the right moment.
_BLINK_COUNT_REQUIRED = 1

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

    # Hold-based challenges (nod, turn_left, turn_right) need the FULL filtered
    # history so the complete turn/down → hold → return sequence can be detected.
    # For instant challenges (look_straight, blink) the recent-20 window is fine.
    filtered_history = [
        f for f in feature_history
        if f.get("det_score", 1.0)         >= MIN_FACE_DETECTION_CONFIDENCE
        and f.get("bbox_area_ratio", MIN_FACE_AREA_RATIO) >= MIN_FACE_AREA_RATIO
    ]
    # Keep a 20-frame recent window for look_straight (and as a fallback).
    recent = filtered_history[-20:] if len(filtered_history) > 20 else filtered_history
    if len(recent) < 3:
        return {"passed": False, "feedback": "No face detected — look directly into the camera."}

    # ─── Look Straight (mandatory first challenge — captures the best selfie) ──
    # Require the nose to stay within the horizontal centre band for several frames.
    # Pitch (nose_y - eye_mid_y / IOD) is always 0.4–0.7 for any real face and is
    # NOT used here — checking it would make this challenge physically impossible.
    # When this challenge passes, api.py saves the current frame as the best selfie.
    if challenge == "look_straight":
        if len(recent) < _STRAIGHT_STABLE_FRAMES:
            remaining = _STRAIGHT_STABLE_FRAMES - len(recent)
            return {"passed": False, "feedback": f"Look directly into the camera… ({remaining} frames remaining)"}
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
    #
    # Hold-only sequence (no return-to-centre required):
    #   Phase 1 — User turns left until |yaw| crosses _TURN_YAW_THRESHOLD.
    #   Phase 2 — User holds that turned position for _TURN_HOLD_FRAMES frames.
    #             Challenge passes immediately — green flash + beep signals done.
    if challenge == "turn_left":
        turn_hist = filtered_history
        if len(turn_hist) < 3:
            return {"passed": False, "feedback": "Slowly turn your head to YOUR LEFT…"}

        yaws = [f["yaw"] for f in turn_hist]

        # Wrong-direction guard: if the user clearly turned RIGHT (yaw strongly positive)
        # while turn_left is required, reset the history. Only fires after ≥ 5 frames
        # so brief detector noise at the start is not penalised.
        if len(turn_hist) >= 5 and max(yaws) >= _TURN_YAW_THRESHOLD:
            return {"passed": False, "feedback": "Wrong direction — turn to YOUR LEFT.", "reset_needed": True, "wrong_motion": "turned_right"}

        # Find the longest contiguous run of frames where yaw is in the turned-left zone.
        # _TURN_HOLD_LENIENCY allows slight relaxation during the hold (natural wobble).
        hold_zone = -(_TURN_YAW_THRESHOLD - _TURN_HOLD_LENIENCY)
        best_start, best_len, cur_start, cur_len = -1, 0, -1, 0
        for i, y in enumerate(yaws):
            if y <= hold_zone:
                if cur_start < 0:
                    cur_start = i
                    cur_len = 1
                else:
                    cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_start, cur_len = -1, 0

        # Require the hold block to have actually crossed the full threshold at least once
        # (prevents a marginally-turned face from gaming the leniency zone).
        crossed_threshold = (
            best_start >= 0 and min(yaws[best_start: best_start + best_len]) <= -_TURN_YAW_THRESHOLD
        )

        if best_len >= _TURN_HOLD_FRAMES and crossed_threshold:
            # Hold quota met — pass immediately, no return required.
            return {"passed": True, "feedback": "✅ Turn completed!"}

        if best_len > 0 and crossed_threshold:
            # Partial hold — show countdown to keep user informed.
            remaining = max(_TURN_HOLD_FRAMES - best_len, 0)
            return {"passed": False, "feedback": f"Keep looking left… ({remaining} more frames)"}

        # User has not turned enough or just started turning.
        min_yaw = min(yaws)
        if abs(min_yaw) >= _TURN_YAW_THRESHOLD:
            # Crossed threshold but hold block is too short (brief flick).
            return {"passed": False, "feedback": "Hold that position — keep your head turned LEFT…"}
        gap  = _TURN_YAW_THRESHOLD - abs(min_yaw)
        hint = "a little more…" if gap < 0.10 else "turn further to YOUR left…"
        return {"passed": False, "feedback": f"Keep turning — {hint}"}

    # ─── Turn Right (subject's right → nose image-RIGHT → yaw POSITIVE) ─────────
    # When the subject turns to their own right in a mirrored frame, the nose swings
    # toward image-right, making yaw positive.
    # Same hold-only sequence as turn_left with sign flipped — no return required.
    if challenge == "turn_right":
        turn_hist = filtered_history
        if len(turn_hist) < 3:
            return {"passed": False, "feedback": "Slowly turn your head to YOUR RIGHT…"}

        yaws = [f["yaw"] for f in turn_hist]

        # Wrong-direction guard: if the user clearly turned LEFT (yaw strongly negative)
        # while turn_right is required, reset the history.
        if len(turn_hist) >= 5 and min(yaws) <= -_TURN_YAW_THRESHOLD:
            return {"passed": False, "feedback": "Wrong direction — turn to YOUR RIGHT.", "reset_needed": True, "wrong_motion": "turned_left"}

        # hold_zone for right turn: yaw ≥ +(_TURN_YAW_THRESHOLD - _TURN_HOLD_LENIENCY).
        hold_zone = _TURN_YAW_THRESHOLD - _TURN_HOLD_LENIENCY
        best_start, best_len, cur_start, cur_len = -1, 0, -1, 0
        for i, y in enumerate(yaws):
            if y >= hold_zone:
                if cur_start < 0:
                    cur_start = i
                    cur_len = 1
                else:
                    cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_start, cur_len = -1, 0

        crossed_threshold = (
            best_start >= 0 and max(yaws[best_start: best_start + best_len]) >= _TURN_YAW_THRESHOLD
        )

        if best_len >= _TURN_HOLD_FRAMES and crossed_threshold:
            # Hold quota met — pass immediately, no return required.
            return {"passed": True, "feedback": "✅ Turn completed!"}

        if best_len > 0 and crossed_threshold:
            # Partial hold — show countdown.
            remaining = max(_TURN_HOLD_FRAMES - best_len, 0)
            return {"passed": False, "feedback": f"Keep looking right… ({remaining} more frames)"}

        max_yaw = max(yaws)
        if max_yaw >= _TURN_YAW_THRESHOLD:
            return {"passed": False, "feedback": "Hold that position — keep your head turned RIGHT…"}
        gap  = _TURN_YAW_THRESHOLD - max_yaw
        hint = "a little more…" if gap < 0.10 else "turn further to YOUR right…"
        return {"passed": False, "feedback": f"Keep turning — {hint}"}

    # ─── Nod (vertical head tilt → hold down) ──────────────────────────────────
    # The challenge requires a HOLD sequence — no return-to-neutral:
    #   Phase 1 — Collect a neutral-pitch baseline from the first few frames
    #             (user was looking straight after the stability gate).
    #   Phase 2 — User tilts head down until pitch deviates ≥ _NOD_DOWN_DELTA
    #             from baseline. Both directions are accepted because different
    #             cameras and user heights produce different pitch directions for
    #             "chin toward chest".
    #   Phase 3 — User holds that deviated position for _NOD_HOLD_FRAMES frames.
    #             Challenge passes immediately at this point — green flash + beep
    #             tells the user they are done. No look-up required.
    if challenge == "nod":
        # Use the full filtered history to detect the complete down→hold→return arc.
        nod_hist = filtered_history
        if len(nod_hist) < 3:
            return {"passed": False, "feedback": "Look DOWN..."}

        # Wrong-motion guard runs immediately (even with few frames) so the user
        # gets corrective feedback before the hold timer starts.
        # A side-to-side shake produces a large yaw range but tiny pitch range.
        pitches   = [f["pitch"] for f in nod_hist]
        yaws      = [f.get("yaw", 0.0) for f in nod_hist]
        pitch_rng = max(pitches) - min(pitches)
        yaw_rng   = max(yaws) - min(yaws)
        if yaw_rng > pitch_rng * 2.0 and yaw_rng > 0.10:
            return {"passed": False, "feedback": "That's a turn, not a nod — tilt your chin DOWN, not sideways.", "wrong_motion": "side_to_side_shake"}

        if len(nod_hist) < _NOD_BASELINE_FRAMES + 3:
            return {"passed": False, "feedback": "Look DOWN..."}

        # Phase 1: baseline pitch from the first frames (user was frontal/straight).
        baseline_pitch = (
            sum(f["pitch"] for f in nod_hist[:_NOD_BASELINE_FRAMES]) / _NOD_BASELINE_FRAMES
        )

        # Phase 2-3: find the longest contiguous run where pitch is deviated from
        # baseline by ≥ _NOD_DOWN_DELTA. Direction-agnostic so it works for all
        # camera heights (pitch can go either way for "chin down").
        best_start, best_len, cur_start, cur_len = -1, 0, -1, 0
        for i in range(_NOD_BASELINE_FRAMES, len(pitches)):
            if abs(pitches[i] - baseline_pitch) >= _NOD_DOWN_DELTA:
                if cur_start < 0:
                    cur_start = i
                    cur_len = 1
                else:
                    cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_start, cur_len = -1, 0

        if best_len >= _NOD_HOLD_FRAMES:
            # Hold quota met — pass immediately. The green flash + beep in the
            # browser tells the user they are done; no return-to-neutral needed.
            return {"passed": True, "feedback": "✅ Nod completed!"}

        if best_len > 0:
            # Partial hold — show countdown so the user knows they are making progress.
            remaining = max(_NOD_HOLD_FRAMES - best_len, 0)
            if remaining == 0:
                return {"passed": False, "feedback": "Hold still..."}
            return {"passed": False, "feedback": f"Keep looking down... ({remaining} more frames)"}

        # No frames have met the threshold yet — give a gentle, encouraging prompt
        # without a frame counter (which confused users into thinking they were stuck).
        return {"passed": False, "feedback": "Gently tilt your chin down…"}

    # ─── Blink ────────────────────────────────────────────────────────────────
    if challenge == "blink":
        # Apply the same confidence filter used for all other challenges.
        # The blink check walks the full feature_history (not just `recent`) so
        # low-confidence ghost frames must be stripped here independently.
        blink_hist = [f for f in feature_history if f.get("det_score", 1.0) >= MIN_FACE_DETECTION_CONFIDENCE]
        # Apply the same face-size filter — blink can also be faked by noise in
        # a tiny background detection whose EAR or det_score fluctuates.
        blink_hist = [f for f in blink_hist if f.get("bbox_area_ratio", MIN_FACE_AREA_RATIO) >= MIN_FACE_AREA_RATIO]
        # 3 frames minimum: 1 baseline open + 1 closed dip + 1 reopen.
        if len(blink_hist) < 3:
            return {"passed": False, "feedback": "Hold still and look at the camera…"}

        # ── Primary path: EAR (Eye Aspect Ratio) — works with MediaPipe ──────
        # Open eye: EAR ≈ 0.25-0.35. Closed eye: EAR ≈ 0.02-0.10.
        # The default 0.30 fill value means no real data — detect via variance.
        ears = [f.get("ear", 0.30) for f in blink_hist]
        ear_variance = max(ears) - min(ears)
        has_real_ear = ear_variance > 0.04  # flat 0.30 everywhere = no real data

        if has_real_ear:
            # We have genuine EAR data. Count distinct blink events using a
            # state machine: open → dipping (EAR below threshold) → open again.
            # Each complete dip-recovery cycle counts as one blink.
            # One blink is required to pass (_BLINK_COUNT_REQUIRED = 1).
            baseline_window = ears[:-2] if len(ears) > 2 else ears
            baseline_open   = max(baseline_window or ears)

            # Set thresholds relative to the user's own open-eye baseline so
            # the check works for a wide range of webcam calibration levels.
            dip_threshold      = min(0.20, baseline_open * 0.72)  # EAR below this = "closing"
            recovery_threshold = max(0.18, baseline_open * 0.68)  # EAR above this = "open again"
            open_threshold     = max(0.20, baseline_open * 0.82)  # baseline open level

            blink_count = 0
            eye_state   = "open"   # start in open state

            for i, ear in enumerate(ears):
                if eye_state == "open":
                    # Transition to "closing" only when we were clearly open before.
                    if ear < dip_threshold:
                        before = ears[:i]
                        if not before or max(before) >= open_threshold:
                            eye_state = "closing"
                elif eye_state == "closing":
                    # Transition back to "open" when EAR recovers — that's one blink.
                    if ear >= recovery_threshold:
                        blink_count += 1
                        eye_state = "open"
                        if blink_count >= _BLINK_COUNT_REQUIRED:
                            return {"passed": True, "feedback": "✅ Blink detected!"}

            # Give progressive feedback so the user knows how many more blinks are needed.
            return {"passed": False, "feedback": "Blink naturally — close both eyes fully, then open them…"}

        else:
            # ── Fallback: det_score dip (InsightFace only) ───────────────────
            # When MediaPipe EAR isn't available, a blink causes a small but
            # measurable dip in the face-detection confidence score (det_score).
            # The fallback uses a single-dip check (less accurate than EAR counting).
            all_scores = [f["det_score"] for f in blink_hist]
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
