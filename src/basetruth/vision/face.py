"""Computer vision module for identity verification and fraud detection.

This module provides high-accuracy, offline-first face detection and verification
using OpenCV, RetinaFace, and ArcFace ONNX models via InsightFace.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

# We lazy-load insightface so the entire app doesn't crash if it's missing.
_insightface = None
_face_app = None

log = logging.getLogger(__name__)

def _ensure_insightface():
    global _insightface
    if _insightface is None:
        try:
            import os
            import tempfile
            # Prevent matplotlib from trying to write to Docker-only directories.
            # Use the system temp dir so this works on both Windows and Linux.
            _tmp = tempfile.gettempdir()
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(_tmp, "matplotlib"))
            os.environ.setdefault("XDG_CACHE_HOME",  os.path.join(_tmp, "cache"))
            os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(_tmp, "config"))

            import insightface
            _insightface = insightface
        except ImportError:
            raise ImportError(
                "insightface is not installed. "
                "Run: pip install insightface onnxruntime  "
                "(requires Python ≤ 3.12 on Windows; use Docker for Python 3.13+)."
            )

def get_face_analyzer() -> Any:
    """Lazy-initialize InsightFace (RetinaFace + ArcFace). Downloads ~300 MB models on first run."""
    global _face_app
    _ensure_insightface()
    if _face_app is None:
        from insightface.app import FaceAnalysis
        import os

        # Resolve a cross-platform models directory:
        #   Docker:  BASETRUTH_ARTIFACT_ROOT=/app/artifacts  →  /app/your_data/models
        #   Local:   BASETRUTH_ARTIFACT_ROOT not set          →  <repo>/your_data/models
        artifact_root = os.environ.get("BASETRUTH_ARTIFACT_ROOT", "")
        if artifact_root:
            models_dir = str(Path(artifact_root).parent / "your_data" / "models")
        else:
            # Derive from the location of this source file: src/basetruth/vision/face.py
            # Walk up to the repo root (parent of src/)
            _here = Path(__file__).resolve()
            _repo_root = _here.parent.parent.parent.parent  # vision -> basetruth -> src -> repo
            models_dir = str(_repo_root / "your_data" / "models")

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        os.environ["INSIGHTFACE_HOME"] = models_dir

        log.info("Initializing FaceAnalyzer (may download ~300 MB to %s on first run)...", models_dir)
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=models_dir,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("FaceAnalyzer initialized.")

    return _face_app


# ─── MediaPipe Face Mesh fallback (works on Python 3.13, no C extensions) ─────

_mp_landmarker = None
_mp_image_cls = None


class _MediaPipeFace:
    """InsightFace-compatible face object wrapping a MediaPipe FaceLandmarker result.

    Exposes the same ``.kps``, ``.bbox``, and ``.det_score`` attributes used by
    ``extract_features()`` in kyc/liveness.py, plus ``.ear`` for blink detection.
    """

    def __init__(self, kps: np.ndarray, bbox: np.ndarray, ear: float) -> None:
        self.kps = kps                # (5, 2) float32
        self.bbox = bbox              # [x1, y1, x2, y2] float32
        self.det_score: float = 0.95  # MediaPipe is highly confident when face found
        self.ear: float = ear         # Eye Aspect Ratio from EAR landmarks
        self.normed_embedding = None  # Face-match requires InsightFace


def _mp_ear(pts: np.ndarray, idxs: list) -> float:
    """Eye Aspect Ratio from 6 landmark indices (Soukupova & Cech 2016)."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idxs)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (
        2.0 * np.linalg.norm(p1 - p4) + 1e-6
    )


def _get_mp_landmarker():
    """Lazy-init MediaPipe FaceLandmarker (downloads ~3 MB model on first run)."""
    global _mp_landmarker, _mp_image_cls
    if _mp_landmarker is None:
        import os
        import urllib.request
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415

        # Resolve model path alongside InsightFace models
        artifact_root = os.environ.get("BASETRUTH_ARTIFACT_ROOT", "")
        if artifact_root:
            models_dir = str(Path(artifact_root).parent / "your_data" / "models")
        else:
            _here = Path(__file__).resolve()
            _repo_root = _here.parent.parent.parent.parent
            models_dir = str(_repo_root / "your_data" / "models")

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(models_dir, "face_landmarker.task")

        if not os.path.exists(model_path):
            log.info("Downloading MediaPipe face_landmarker.task (~3.6 MB) to %s ...", models_dir)
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            log.info("MediaPipe model downloaded.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,   # enables blink score per-eye
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _mp_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        _mp_image_cls = mp.Image
        log.info("MediaPipe FaceLandmarker initialized.")

    return _mp_landmarker, _mp_image_cls


def get_mediapipe_faces(img_bgr: np.ndarray) -> list:
    """Detect faces using MediaPipe FaceLandmarker (Tasks API).

    Returns a list of ``_MediaPipeFace`` objects with the same interface as
    InsightFace face objects so they can be used by ``extract_features()``.
    """
    import mediapipe as mp  # noqa: PLC0415

    landmarker, mp_image_cls = _get_mp_landmarker()
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_cls(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    faces: list = []
    for lm_list, blendshapes in zip(
        result.face_landmarks,
        result.face_blendshapes if result.face_blendshapes else [[]]*len(result.face_landmarks),
    ):
        pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)

        # 5-point kps matching InsightFace order:
        #   kps[0] = image-left eye  (subject's right in a mirrored webcam)
        #   kps[1] = image-right eye (subject's left  in a mirrored webcam)
        #   kps[2] = nose tip
        #   kps[3] = mouth left corner
        #   kps[4] = mouth right corner
        left_eye  = pts[[33, 133]].mean(axis=0)   # inner+outer of image-left eye
        right_eye = pts[[362, 263]].mean(axis=0)  # inner+outer of image-right eye
        nose_tip  = pts[4]
        mouth_l   = pts[61]
        mouth_r   = pts[291]

        kps = np.array([left_eye, right_eye, nose_tip, mouth_l, mouth_r], dtype=np.float32)

        # Bounding box from all face landmarks
        bbox = np.array(
            [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()],
            dtype=np.float32,
        )

        # EAR using standard 6-point per-eye patterns (used when blendshapes unavailable)
        ear_l = _mp_ear(pts, [33, 160, 158, 133, 153, 144])   # image-left eye
        ear_r = _mp_ear(pts, [362, 385, 387, 263, 380, 373])  # image-right eye
        ear = (ear_l + ear_r) / 2.0

        # If blendshapes available, use eyeBlink scores for more reliable EAR
        if blendshapes:
            bs_map = {b.category_name: b.score for b in blendshapes}
            blink_l = bs_map.get("eyeBlinkLeft", None)
            blink_r = bs_map.get("eyeBlinkRight", None)
            if blink_l is not None and blink_r is not None:
                # Convert blink score (0=open, 1=closed) to EAR-like value
                # EAR open ≈ 0.30, closed ≈ 0.03 → EAR = 0.30 * (1 - blink_score)
                ear = 0.30 * (1.0 - (blink_l + blink_r) / 2.0)

        faces.append(_MediaPipeFace(kps=kps, bbox=bbox, ear=ear))

    return faces


def _draw_face(img: np.ndarray, face: Any) -> np.ndarray:
    """Draw bounding box and landmarks (eyes, nose, mouth) on a face."""
    out = copy.deepcopy(img)
    box = face.bbox.astype(int)
    
    # Draw green bounding box
    cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
    
    # Draw red facial landmarks (usually 5 points from RetinaFace)
    if face.kps is not None:
        kps = face.kps.astype(int)
        for p in kps:
            cv2.circle(out, (p[0], p[1]), 3, (0, 0, 255), 3)
            
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB) # Streamlit expects RGB

def compare_faces(document_bytes: bytes, selfie_bytes: bytes) -> Dict[str, Any]:
    """Compare faces from an ID document and a Selfie.
    
    Returns a dict with:
    - match (bool): True if faces match.
    - confidence (float): 0.0 to 1.0 confidence score.
    - doc_annotated (bytes): PNG bytes of document with bbox.
    - selfie_annotated (bytes): PNG bytes of selfie with bbox.
    - error (str): If any processing error occurs.
    """
    try:
        # Decode bytes to OpenCV matrices
        nparr_doc = np.frombuffer(document_bytes, np.uint8)
        img_doc = cv2.imdecode(nparr_doc, cv2.IMREAD_COLOR)
        if img_doc is None:
            return {"error": "Could not decode document image."}
            
        nparr_selfie = np.frombuffer(selfie_bytes, np.uint8)
        img_selfie = cv2.imdecode(nparr_selfie, cv2.IMREAD_COLOR)
        if img_selfie is None:
            return {"error": "Could not decode selfie image."}

        # Run inference using InsightFace (RetinaFace -> ArcFace)
        app = get_face_analyzer()
        faces_doc = app.get(img_doc)
        faces_selfie = app.get(img_selfie)

        if len(faces_doc) == 0:
            return {"error": "No face found in the Document image."}
        if len(faces_selfie) == 0:
            return {"error": "No face found in the Selfie image."}

        # If multiple faces are detected, choose the primary (largest) face
        def _largest_face(faces_list):
            return max(faces_list, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        primary_doc_face = _largest_face(faces_doc)
        primary_selfie_face = _largest_face(faces_selfie)

        # Compute cosine similarity between the two identity embedding vectors
        # Embeddings are normalized (L2 norm = 1), so inner product represents cosine similarity.
        emb_doc = primary_doc_face.normed_embedding
        emb_selfie = primary_selfie_face.normed_embedding
        
        sim = float(np.dot(emb_doc, emb_selfie))
        
        # Mapping Insightface Cosine Similarity to a percentage score
        # Typically, a score > 0.40 is considered a safe match boundary for ArcFace.
        # However, to be extra certain in fraud, we might want 0.45 or 0.50.
        threshold = 0.40
        is_match = sim >= threshold
        
        # Scale the score for user display (e.g. 0.4 -> 50%, 0.8 -> ~95%)
        # This is an arbitrary linear mapping to make numbers "look" like standard percentages
        display_score = min(max((sim - (-0.5)) / (1.0 - (-0.5)) * 100, 0), 100)

        # Create annotated output images
        disp_doc = _draw_face(img_doc, primary_doc_face)
        disp_selfie = _draw_face(img_selfie, primary_selfie_face)

        return {
            "match": is_match,
            "confidence": min(max(sim, 0.0), 1.0),
            "display_score": display_score,
            "threshold": threshold,
            "doc_annotated_rgb": disp_doc,
            "selfie_annotated_rgb": disp_selfie,
        }

    except Exception as exc:
        log.exception("compare_faces encounted an error: %s", exc)
        return {"error": str(exc)}
