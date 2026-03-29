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
            # Prevent matplotlib (used internally by insightface) from trying to write
            # to the non-existent or read-only /home/basetruth directory
            os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
            os.environ["XDG_CACHE_HOME"] = "/tmp/cache"
            os.environ["XDG_CONFIG_HOME"] = "/tmp/config"
            
            import insightface
            _insightface = insightface
        except ImportError:
            raise ImportError("Please install insightface and onnxruntime to use face verification.")

def get_face_analyzer() -> Any:
    """Lazy initialize and download the InsightFace models (RetinaFace + ArcFace)."""
    global _face_app
    _ensure_insightface()
    if _face_app is None:
        from insightface.app import FaceAnalysis
        import os
        
        # Override the default download directory (~/.insightface) since the docker user
        # doesn't have a home directory. We write it to the persistent your_data mount instead.
        models_dir = os.path.join(os.environ.get("BASETRUTH_ARTIFACT_ROOT", "/app/artifacts").replace("artifacts", "your_data"), "models")
        os.makedirs(models_dir, exist_ok=True)
        os.environ["INSIGHTFACE_HOME"] = models_dir

        # buffalo_l is the default high-accuracy model pack (includes RetinaFace and ArcFace).
        # It's roughly ~300MB and auto-downloads to INSIGHTFACE_HOME/models/ on first run.
        log.info(f"Initializing FaceAnalyzer (may download ~300MB of ONNX models to {models_dir} on first run)...")
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=models_dir,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"] # Force CPU for universal compat; switch to CUDA if needed
        )
        
        # det_size ensures bounding box scaling works reliably
        # ctx_id=0 uses GPU if available, but since we set CPU provider it will use CPU
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("FaceAnalyzer initialized successfully.")
        
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
