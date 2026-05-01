"""Static Face Scan service.

This module implements the first production-grade Face Scan slice: a
deterministic static-image analyzer that returns one canonical response payload
for both the Streamlit UI and the REST API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
import uuid
from typing import Any, Dict, List

import cv2
import numpy as np

from basetruth.logger import get_logger
from basetruth.vision.face import get_face_analyzer, get_mediapipe_faces

log = get_logger(__name__)

FACE_SCAN_SCHEMA_VERSION = "1.0.0"
FACE_SCAN_RULES_VERSION = "face-scan-rules-1.0.0"
FACE_SCAN_MODEL_VERSION = "heuristics-only"


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a numeric value to a bounded range."""
    return max(low, min(high, float(value)))


def _scale_risk(value: float, low: float, high: float, inverse: bool = False) -> float:
    """Map a raw measurement to a 0-100 risk scale.

    We use simple piecewise-linear scaling here because the first production
    slice needs deterministic, explainable behaviour. Thresholds can be tuned
    later with evaluation data.
    """
    if high <= low:
        return 0.0
    if inverse:
        if value <= low:
            return 100.0
        if value >= high:
            return 0.0
        return _clamp((high - value) * 100.0 / (high - low))

    if value <= low:
        return 0.0
    if value >= high:
        return 100.0
    return _clamp((value - low) * 100.0 / (high - low))


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes into an OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _detect_faces(img: np.ndarray) -> List[Any]:
    """Detect faces with InsightFace first, then fall back to MediaPipe.

    We keep this helper small so tests can monkeypatch it cheaply and the
    production path stays aligned with the existing face stack.
    """
    try:
        analyzer = get_face_analyzer()
        faces = analyzer.get(img)
        if faces:
            return list(faces)
    except ImportError:
        log.info("Face Scan static analysis falling back to MediaPipe because InsightFace is unavailable.")
    except Exception as exc:
        log.warning(
            "Face Scan static InsightFace detection failed; using MediaPipe fallback.",
            extra={"error": str(exc)},
        )

    try:
        return list(get_mediapipe_faces(img))
    except Exception as exc:
        log.error("Face Scan static detection failed in both detectors.", extra={"error": str(exc)})
        return []


def _clip_box(box: np.ndarray, img: np.ndarray) -> tuple[int, int, int, int]:
    """Clip a face bounding box to the decoded image size."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _laplacian_variance(gray: np.ndarray) -> float:
    """Return a simple blur-quality metric from the grayscale image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edge_halo_score(gray_crop: np.ndarray) -> float:
    """Estimate whether edge density is unusually concentrated at the face border.

    Replay screens and composited faces often look more stable in the centre of
    the face than around the jawline and hairline. We use that as a coarse risk
    signal, not as a final decision by itself.
    """
    if min(gray_crop.shape[:2]) < 24:
        return 0.0

    edges = cv2.Canny(gray_crop, 80, 160)
    h, w = edges.shape[:2]
    margin = max(2, int(min(h, w) * 0.1))
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[margin : h - margin, margin : w - margin] = False
    centre_mask = ~border_mask
    if not centre_mask.any():
        return 0.0

    border_density = float(edges[border_mask].mean() / 255.0)
    centre_density = float(edges[centre_mask].mean() / 255.0)
    return _clamp((border_density - centre_density) * 220.0)


def _compression_residual_score(face_crop: np.ndarray) -> float:
    """Estimate JPEG-like residual artefacts inside the face crop."""
    if min(face_crop.shape[:2]) < 24:
        return 0.0

    ok, encoded = cv2.imencode(
        ".jpg",
        face_crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok:
        return 0.0

    restored = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if restored is None or restored.shape != face_crop.shape[:2]:
        return 0.0

    residual = float(np.mean(np.abs(face_crop.astype(np.float32) - restored.astype(np.float32))))
    return _clamp((residual - 1.5) * 12.0)


def _landmark_asymmetry_score(face: Any, face_height: float) -> float:
    """Estimate landmark asymmetry from the available eye and mouth points.

    This is only a weak signal. It helps flag unstable geometry, but it should
    not overrule the rest of the evidence on its own.
    """
    kps = getattr(face, "kps", None)
    if kps is None:
        return 0.0

    kps_arr = np.asarray(kps, dtype=np.float32)
    if kps_arr.shape[0] < 5:
        return 0.0

    eye_y_gap = abs(float(kps_arr[0][1] - kps_arr[1][1])) / max(face_height, 1.0)
    mouth_y_gap = abs(float(kps_arr[3][1] - kps_arr[4][1])) / max(face_height, 1.0)
    return _clamp((eye_y_gap * 220.0) + (mouth_y_gap * 180.0))


def _confidence_reason(blur_risk: float, brightness_risk: float, face_size_risk: float, face_count: int) -> str:
    """Explain in simple English why the confidence ended up where it did."""
    if face_count > 1:
        return "Multiple faces were found, so the static scan is less reliable."
    if blur_risk >= 60.0:
        return "Motion blur or soft focus reduced confidence in the static scan."
    if brightness_risk >= 60.0:
        return "Lighting was too uneven for a fully reliable static decision."
    if face_size_risk >= 60.0:
        return "The face occupies too little of the frame for a high-confidence result."
    return "Single clear face with stable lighting and sharp enough detail for a reliable static scan."


def _build_evidence(
    face_count: int,
    blur_risk: float,
    brightness_risk: float,
    edge_halo_score: float,
    compression_score: float,
    asymmetry_score: float,
    confidence: float,
) -> List[str]:
    """Build short evidence bullets for operators and API clients."""
    evidence: List[str] = []

    if face_count == 1:
        evidence.append("One primary face was detected for the static scan.")
    elif face_count > 1:
        evidence.append("Multiple faces were detected, which increases spoof and ambiguity risk.")

    if blur_risk >= 60.0:
        evidence.append("Image sharpness is weak, which reduces confidence in the result.")
    elif blur_risk <= 20.0:
        evidence.append("Facial edges are sharp enough for a stable static analysis.")

    if brightness_risk >= 60.0:
        evidence.append("Lighting is uneven or too dark, so some signals are less reliable.")
    elif brightness_risk <= 25.0:
        evidence.append("Exposure looks balanced enough for a reliable static assessment.")

    if max(edge_halo_score, compression_score) >= 55.0:
        evidence.append("Boundary and compression signals around the face look inconsistent with a clean capture.")
    elif max(edge_halo_score, compression_score) <= 20.0:
        evidence.append("No strong replay-screen or print-photo artefacts were found around the face boundary.")

    if asymmetry_score >= 45.0:
        evidence.append("Landmark geometry shows elevated asymmetry for a single still portrait.")

    if confidence < 35.0:
        evidence.append("Signal quality is too weak for a high-confidence decision.")

    return evidence


def _overall_explanation(verdict: str) -> str:
    """Summarise the strongest technical findings in one sentence."""
    if verdict == "INCONCLUSIVE":
        return "The static face scan found a face, but the image quality is too weak for a reliable authenticity decision."
    if verdict == "DEEPFAKE":
        return "The static face scan found strong presentation or synthetic artefact signals around the face region."
    if verdict == "SUSPICIOUS":
        return "The static face scan found moderate spoof-risk signals, so the image should be reviewed manually."
    return "The static face scan found a clear face with low spoof-risk signals and good enough image quality."


def _honest_review(verdict: str) -> str:
    """Return a simple operator-facing summary."""
    if verdict == "INCONCLUSIVE":
        return "The uploaded photo is not clear enough for a reliable face-authenticity decision. Capture a sharper, better-lit image and try again."
    if verdict == "DEEPFAKE":
        return "The uploaded photo shows strong spoof-risk signals. Treat it as high risk until a live session or manual review clears it."
    if verdict == "SUSPICIOUS":
        return "The uploaded photo is usable, but it contains enough spoof-risk signals to justify manual review or a live challenge."
    return "The uploaded photo appears genuine in this static check, with no strong spoof-risk signals detected."


def run_face_scan_static(
    image_bytes: bytes,
    filename: str,
    environment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a deterministic static Face Scan and return the canonical payload."""
    started_at = time.perf_counter()

    img = _decode_image(image_bytes)
    if img is None:
        return {"error": "Could not decode image.", "error_type": "decode_error"}

    faces = _detect_faces(img)
    if not faces:
        return {"error": "No face found in the image.", "error_type": "no_face"}

    primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = _clip_box(np.asarray(primary_face.bbox), img)
    face_crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    img_area = float(max(img.shape[0] * img.shape[1], 1))
    face_area_ratio = float(max((x2 - x1) * (y2 - y1), 1) / img_area)
    laplacian_var = _laplacian_variance(face_crop_gray)
    brightness_mean = float(np.mean(face_crop_gray))
    det_score = float(getattr(primary_face, "det_score", 0.95) or 0.95)

    blur_risk = _scale_risk(laplacian_var, 25.0, 140.0, inverse=True)
    brightness_risk = _scale_risk(abs(brightness_mean - 128.0), 18.0, 90.0)
    face_size_risk = _scale_risk(face_area_ratio, 0.06, 0.18, inverse=True)
    edge_halo_score = _edge_halo_score(face_crop_gray)
    compression_score = _compression_residual_score(face_crop_gray)
    asymmetry_score = _landmark_asymmetry_score(primary_face, float(y2 - y1))

    presentation_risk = round((0.6 * edge_halo_score) + (0.4 * compression_score), 2)
    synthetic_risk = round(asymmetry_score, 2)
    multi_face_risk = 85.0 if len(faces) > 1 else 0.0
    risk_score = round(
        _clamp((0.45 * presentation_risk) + (0.20 * synthetic_risk) + (0.35 * multi_face_risk)),
        2,
    )

    detection_penalty = _scale_risk(max(0.0, 0.85 - det_score), 0.02, 0.25)
    confidence = round(
        _clamp(
            100.0
            - (0.35 * blur_risk)
            - (0.30 * brightness_risk)
            - (0.20 * face_size_risk)
            - (0.15 * detection_penalty),
            5.0,
            99.0,
        ),
        2,
    )
    if len(faces) > 1:
        confidence = min(confidence, 55.0)
    if laplacian_var < 12.0 or face_area_ratio < 0.03:
        confidence = min(confidence, 25.0)

    if confidence < 35.0:
        verdict = "INCONCLUSIVE"
    elif risk_score >= 75.0:
        verdict = "DEEPFAKE"
    elif risk_score >= 35.0 or len(faces) > 1:
        verdict = "SUSPICIOUS"
    else:
        verdict = "GENUINE"

    runtime_ms = int((time.perf_counter() - started_at) * 1000)
    env = {
        "platform": "upload",
        "browser": None,
        "os": None,
        "camera_resolution": None,
        "observed_fps": None,
        "virtual_camera_suspected": False,
    }
    if environment:
        env.update(environment)

    return {
        "filename": filename,
        "scan_type": "face_scan",
        "mode": "static",
        "schema_version": FACE_SCAN_SCHEMA_VERSION,
        "verdict": verdict,
        "risk_score_0_100": risk_score,
        "confidence_0_100": confidence,
        "confidence_reason": _confidence_reason(
            blur_risk=blur_risk,
            brightness_risk=brightness_risk,
            face_size_risk=face_size_risk,
            face_count=len(faces),
        ),
        "overall_explanation": _overall_explanation(verdict),
        "honest_review": _honest_review(verdict),
        "evidence": _build_evidence(
            face_count=len(faces),
            blur_risk=blur_risk,
            brightness_risk=brightness_risk,
            edge_halo_score=edge_halo_score,
            compression_score=compression_score,
            asymmetry_score=asymmetry_score,
            confidence=confidence,
        ),
        "trace": {
            "decision_trace_id": f"fs_{uuid.uuid4().hex[:12]}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "processing_time_ms": runtime_ms,
            "rules_version": FACE_SCAN_RULES_VERSION,
            "model_version": FACE_SCAN_MODEL_VERSION,
        },
        "environment": env,
        "checks": {
            "face_detection": {
                "status": "pass",
                "face_count": len(faces),
                "primary_face_box": [x1, y1, x2, y2],
                "detector_confidence": round(det_score, 4),
                "face_area_ratio": round(face_area_ratio, 4),
            },
            "quality_assessment": {
                "status": "review" if confidence < 60.0 else "pass",
                "blur_risk_0_100": round(blur_risk, 2),
                "brightness_risk_0_100": round(brightness_risk, 2),
                "face_size_risk_0_100": round(face_size_risk, 2),
                "laplacian_variance": round(laplacian_var, 2),
                "brightness_mean": round(brightness_mean, 2),
            },
            "photo_authenticity": {
                "status": "review" if presentation_risk >= 35.0 else "pass",
                "score_0_100": presentation_risk,
                "signals": {
                    "screen_replay": round(edge_halo_score, 2),
                    "compression_anomaly": round(compression_score, 2),
                    "print_attack": round(max(edge_halo_score, compression_score), 2),
                },
            },
            "deepfake_signals": {
                "status": "review" if synthetic_risk >= 35.0 else "pass",
                "score_0_100": synthetic_risk,
                "signals": {
                    "landmark_asymmetry": round(asymmetry_score, 2),
                },
            },
            "active_liveness": {
                "status": "not_run",
                "passed": False,
                "completed_challenges": [],
                "challenge_count": 0,
                "best_frame_available": False,
            },
        },
        "artifacts": {
            "best_frame_available": False,
            "challenge_snapshots_available": False,
        },
    }