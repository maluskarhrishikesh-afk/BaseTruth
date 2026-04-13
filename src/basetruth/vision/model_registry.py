"""Pluggable model registry for BaseTruth OCR and vision models.

Allows the OCR engine, face analyzer, and VLM backend to be selected at
runtime via environment variables, without changing any code.

Environment variables
---------------------
    BASETRUTH_OCR_ENGINE    : 'paddleocr' | 'auto' (default: 'auto')
  BASETRUTH_FACE_ENGINE   : 'insightface' | 'mediapipe' | 'auto' (default: 'auto')
  BASETRUTH_VLM_ENGINE    : 'gemma_local' | 'gemini_api' | 'none' (default: 'auto')
  BASETRUTH_OCR_CONF_THRESHOLD : float 0–1 (default: 0.70)

Usage
-----
  from basetruth.vision.model_registry import get_ocr_engine, get_face_analyzer

  ocr = get_ocr_engine()
  text, conf = ocr(pil_image)

  analyzer = get_face_analyzer()
  faces = analyzer.get(bgr_frame)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OCR_ENGINE: str = os.environ.get("BASETRUTH_OCR_ENGINE", "auto").lower()
FACE_ENGINE: str = os.environ.get("BASETRUTH_FACE_ENGINE", "auto").lower()
VLM_ENGINE: str = os.environ.get("BASETRUTH_VLM_ENGINE", "auto").lower()
OCR_CONF_THRESHOLD: float = float(
    os.environ.get("BASETRUTH_OCR_CONF_THRESHOLD", "0.70")
)

# ---------------------------------------------------------------------------
# OCR engine registry
# ---------------------------------------------------------------------------

_ocr_instance: Optional[Any] = None


def _make_paddle_ocr() -> Callable:
    """Factory: PaddleOCR engine returning (text, confidence)."""
    from paddleocr import PaddleOCR  # type: ignore
    import numpy as np  # type: ignore

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    _engine = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv4",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    def _run(pil_img: Any) -> Tuple[str, float]:
        img_arr = np.array(pil_img.convert("RGB"))
        result = _engine.predict(img_arr) or []
        texts, confs = [], []
        for page in result:
            if not isinstance(page, dict):
                continue
            rec_texts = page.get("rec_texts") or []
            rec_scores = page.get("rec_scores") or []
            for index, text in enumerate(rec_texts):
                clean_text = str(text or "").strip()
                if not clean_text:
                    continue
                texts.append(clean_text)
                if index < len(rec_scores):
                    confs.append(float(rec_scores[index]))
        combined = "\n".join(texts)
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return combined, mean_conf

    return _run


def get_ocr_engine() -> Callable:
    """Return the PaddleOCR callable based on BASETRUTH_OCR_ENGINE.

    The returned callable has signature: (pil_image) -> (text: str, confidence: float)
    """
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    if OCR_ENGINE in ("paddleocr", "auto"):
        try:
            engine = _make_paddle_ocr()
            log.debug("OCR engine: PaddleOCR")
            _ocr_instance = engine
            return engine
        except ImportError:
            log.warning("PaddleOCR is not installed in the current environment.")
    elif OCR_ENGINE:
        log.warning("Unsupported OCR engine '%s'. BaseTruth now supports only PaddleOCR.", OCR_ENGINE)

    # Last-resort stub
    def _noop(pil_img: Any) -> Tuple[str, float]:
        return "", 0.0

    log.warning("No OCR engine available (install paddleocr and paddlepaddle).")
    _ocr_instance = _noop
    return _noop


def reset_ocr_engine() -> None:
    """Force re-initialisation on the next call (useful for testing)."""
    global _ocr_instance
    _ocr_instance = None


# ---------------------------------------------------------------------------
# Face analyzer registry
# ---------------------------------------------------------------------------

_face_instance: Optional[Any] = None


def get_face_analyzer() -> Any:
    """Return the best available face analyzer based on BASETRUTH_FACE_ENGINE.

    Delegates to basetruth.vision.face which already handles InsightFace vs
    MediaPipe selection — this registry layer just provides a consistent import
    path and allows the preference to be overridden via the environment variable.
    """
    global _face_instance
    if _face_instance is not None:
        return _face_instance

    if FACE_ENGINE in ("insightface", "auto"):
        try:
            from basetruth.vision.face import get_face_analyzer as _gfa  # type: ignore
            _face_instance = _gfa()
            log.debug("Face engine: InsightFace")
            return _face_instance
        except Exception as exc:  # noqa: BLE001
            if FACE_ENGINE == "insightface":
                log.warning("InsightFace unavailable: %s — falling back to MediaPipe.", exc)

    try:
        from basetruth.vision.face import get_mediapipe_analyzer  # type: ignore
        _face_instance = get_mediapipe_analyzer()
        log.debug("Face engine: MediaPipe")
        return _face_instance
    except Exception as exc:  # noqa: BLE001
        log.warning("No face analyzer available: %s", exc)
        return None


def reset_face_analyzer() -> None:
    """Force re-initialisation on the next call."""
    global _face_instance
    _face_instance = None


# ---------------------------------------------------------------------------
# VLM registry
# ---------------------------------------------------------------------------

def get_vlm_engine() -> Optional[str]:
    """Return the preferred VLM backend name ('gemma_local' | 'gemini_api' | None)."""
    if VLM_ENGINE == "none":
        return None
    if VLM_ENGINE == "gemma_local":
        return "gemma_local"
    if VLM_ENGINE == "gemini_api":
        return "gemini_api"

    # auto: check what's available
    import os as _os
    if _os.environ.get("GEMINI_API_KEY"):
        try:
            import google.generativeai  # type: ignore  # noqa: F401
            return "gemini_api"
        except ImportError:
            pass

    gemma_path = _os.environ.get("GEMMA_MODEL_PATH", "")
    if gemma_path:
        try:
            from transformers import AutoProcessor  # type: ignore  # noqa: F401
            return "gemma_local"
        except ImportError:
            pass

    return None


# ---------------------------------------------------------------------------
# Convenience: registry status summary (for dashboard / health endpoint)
# ---------------------------------------------------------------------------

def registry_status() -> dict:
    """Return a dict summarising which model backends are available."""
    status: dict = {
        "ocr_engine_preference": OCR_ENGINE,
        "face_engine_preference": FACE_ENGINE,
        "vlm_engine_preference": VLM_ENGINE,
        "ocr_conf_threshold": OCR_CONF_THRESHOLD,
    }

    # Probe OCR
    try:
        import paddleocr  # type: ignore  # noqa: F401
        status["paddleocr_available"] = True
    except ImportError:
        status["paddleocr_available"] = False

    # Probe VLM
    import os as _os
    status["gemini_api_key_set"] = bool(_os.environ.get("GEMINI_API_KEY"))
    status["gemma_model_path_set"] = bool(_os.environ.get("GEMMA_MODEL_PATH"))

    try:
        import torch  # type: ignore  # noqa: F401
        status["torch_available"] = True
    except ImportError:
        status["torch_available"] = False

    # Probe face analyzers
    try:
        import insightface  # type: ignore  # noqa: F401
        status["insightface_available"] = True
    except ImportError:
        status["insightface_available"] = False

    try:
        import mediapipe  # type: ignore  # noqa: F401
        status["mediapipe_available"] = True
    except ImportError:
        status["mediapipe_available"] = False

    return status
