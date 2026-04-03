"""Computer vision module for identity verification and fraud detection.

This module provides high-accuracy, offline-first face detection and verification
using OpenCV, RetinaFace, and ArcFace ONNX models via InsightFace.
"""

import copy
import logging
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
        from pathlib import Path

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
