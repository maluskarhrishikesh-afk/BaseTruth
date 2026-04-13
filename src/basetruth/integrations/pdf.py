from __future__ import annotations

"""
PDF integration helpers for BaseTruth.

Text extraction strategy (in priority order)
---------------------------------------------
1. LiteParse  (called by service.scan_document before this module)
   - Requires ImageMagick on Windows
   - Best quality: OCR + layout-aware structure
2. PyMuPDF (fitz)
   - Pure Python, no external binary, installed via 'pip install pymupdf'
   - Excellent text extraction for text-layer PDFs (payslips, offer letters...)
   - Cannot OCR image-only PDFs (Aadhaar, PAN cards)
3. pypdf
   - Pure Python fallback if pymupdf is not installed
   - Acceptable for text PDFs, returns empty for image-only PDFs
4. PaddleOCR via PyMuPDF rendering
    - OCR tier for image-only PDFs (scanned Aadhaar, PAN cards)
    - Uses the same PaddleOCR runtime as the marksheet pipeline
5. Empty string
   - Returned when all extraction methods fail; metadata forensics still run

How to get full scan for Aadhaar / PAN cards on Windows
---------------------------------------------------------
    Option A (recommended): Install ImageMagick
        https://imagemagick.org/script/download.php#windows
        After install, restart the terminal and retry.

    Option B: Ensure PaddleOCR and PaddlePaddle are installed in the BaseTruth environment.

Public API
----------
  extract_pdf_metadata(path)               -> Dict
  extract_text_from_pdf(path)              -> str
    extract_text_via_ocr(path)               -> Tuple[str, str]  (text, engine)
    ocr_image_bytes_directly(image_bytes)    -> Tuple[str, str]  (text, engine)
    build_liteparse_json_from_text(text, src) -> Dict
"""

import logging
import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)

_paddle_ocr_engine: Any = None


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.  Reads in 1 MiB chunks to keep
    memory usage bounded even for very large PDFs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_raw_signature_markers(pdf_bytes: bytes) -> list[str]:
    """
    Scan raw PDF bytes for known digital-signature structural markers.

    Returns a list of marker names that were found.  A non-empty list means the
    document *claims* to carry a digital signature -- it does not verify whether
    the signature is valid or cryptographically intact.  Full cryptographic
    verification requires pdfsig / qpdf (planned for a future release).
    """
    markers = []
    patterns = {
        "sig_type": rb"/Type\s*/Sig",
        "field_sig": rb"/FT\s*/Sig",
        "byte_range": rb"/ByteRange",
        "signature_contents": rb"/Contents",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, pdf_bytes):
            markers.append(name)
    return markers


def extract_pdf_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract structural and descriptive metadata from a PDF file.

    Two passes are performed:
      Pass 1 (raw bytes) -- PDF header version string and digital-signature markers.
      Pass 2 (pypdf)     -- author, creator, producer, creation date, modification
                            date and page count.  Skipped if pypdf is not installed.

    For non-PDF files a minimal payload is returned indicating that metadata
    inspection is not applicable.

    Returns a flat dict consumed by evaluate_tamper_risk() as its pdf_metadata arg.
    """
    payload: Dict[str, Any] = {
        "available": path.suffix.lower() == ".pdf",
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix.lower() != ".pdf":
        payload["message"] = "Metadata inspection currently focuses on PDF files."
        return payload

    pdf_bytes = path.read_bytes()

    # --- Pass 1: raw byte inspection ---
    payload["pdf_header"] = pdf_bytes[:8].decode("latin-1", errors="ignore")
    payload["signature_markers"] = _extract_raw_signature_markers(pdf_bytes)
    payload["has_digital_signature_markers"] = bool(payload["signature_markers"])
    payload["reader"] = "raw"

    # --- Pass 2: pypdf reader (optional -- graceful fallback if not installed) ---
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        metadata = reader.metadata or {}
        payload["reader"] = "pypdf"
        payload["metadata"] = {
            str(key).lstrip("/"): str(value)
            for key, value in dict(metadata).items()
            if value is not None
        }
        payload["page_count"] = len(reader.pages)
    except (ImportError, OSError, TypeError, ValueError) as exc:  # pragma: no cover
        payload["metadata"] = {}
        payload["metadata_error"] = str(exc)

    return payload


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract all plain text from a PDF using PyMuPDF (preferred) or pypdf (fallback).

    PyMuPDF (fitz) is used when available because it handles complex PDF structures,
    multi-column layouts, and embedded fonts far better than pypdf.  Both methods
    return empty string for image-only PDFs (scanned Aadhaar, PAN cards) -- use
    extract_text_via_ocr() for those documents.

    Returns a single newline-joined string of all page text, or '' on any error.
    """
    # --- Strategy 1: PyMuPDF (fitz) -- best text quality, pure Python ---
    try:
        import fitz  # type: ignore   (PyMuPDF)

        doc = fitz.open(str(path))
        page_texts = [page.get_text() or "" for page in doc]
        doc.close()
        return "\n".join(page_texts)
    except ImportError:
        pass  # PyMuPDF not installed; fall through to pypdf
    except Exception:  # noqa: BLE001
        pass  # Corrupt / encrypted / unsupported PDF; fall through

    # --- Strategy 2: pypdf -- lighter but less accurate ---
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page_texts)
    except Exception:  # noqa: BLE001
        return ""


def extract_text_via_ocr(path: Path) -> Tuple[str, str]:
    """
    OCR a PDF using PaddleOCR to extract text from image-only pages
    (scanned Aadhaar card, PAN card, photo-based PDFs).

    Unlike extract_text_from_pdf(), this function can read PDF pages that contain
    no embedded text at all -- it rasterises each page to an image, then OCRs it.

    Returns
    -------
    (text, engine) where engine is one of:
      'paddleocr'    -- OCR succeeded or ran but found no usable text
      'unavailable'  -- PaddleOCR is not installed
      'error'        -- unexpected exception during OCR
    """
    try:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore

        doc = fitz.open(str(path))
        page_texts: list[str] = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            paddle_text, _paddle_conf = _ocr_with_paddle(img)
            if paddle_text.strip():
                page_texts.append(paddle_text)
        doc.close()
        return "\n".join(page_texts), "paddleocr"

    except ImportError:
        return "", "unavailable"
    except Exception:  # noqa: BLE001
        return "", "error"


def is_image_only_pdf(text: str, page_count: int) -> bool:
    """Return True when extracted text is empty or effectively blank for all pages.

    Used by the service layer to decide whether to attempt OCR and to populate
    the 'is_image_only_pdf' flag in the report artifacts.
    """
    meaningful_chars = sum(1 for ch in text if ch.strip() and ch not in "\n\r\f")
    # Threshold: fewer than 20 meaningful characters per page is considered image-only.
    threshold = max(20, page_count * 20)
    return meaningful_chars < threshold


def build_liteparse_json_from_text(text: str, source_name: str) -> Dict[str, Any]:
    """
    Wrap plain extracted text in the minimal LiteParse-compatible JSON schema.

    LiteParse normally emits:
        { "pages": [ { "page": 1, "text": "..." }, ... ] }

    This function reproduces that structure from plain text so that
    build_structured_summary() can consume it without modification, which means
    the entire downstream analysis pipeline (field extraction, tamper scoring,
    domain validators) runs unchanged -- it just has less raw material to work
    with when the source is image-heavy.

    Page boundaries are estimated by splitting on form-feed characters (\f) or
    four or more consecutive newlines, which is a reasonable heuristic for
    most PDF extraction outputs.
    """
    # Split on crude page boundaries.
    raw_pages = re.split(r"\f|\n{4,}", text)
    pages = [
        {"page": idx + 1, "text": page_text.strip()}
        for idx, page_text in enumerate(raw_pages)
        if page_text.strip()
    ]
    if not pages:
        # Ensure at least one entry so the pipeline never sees an empty pages
        # list and produces a correct "generic / low confidence" result instead
        # of crashing on list index operations.
        pages = [{"page": 1, "text": ""}]
    return {
        "_fallback": True,
        "_fallback_source": "pypdf_text_extraction",
        "source": source_name,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Image file helpers  (for raw .jpg / .png / .tiff etc. — not PDF-wrapped)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)


def is_image_file(path: Path) -> bool:
    """Return True when *path* is a raw image format (not a PDF)."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# PaddleOCR — shared OCR engine for scanned images and PDFs
# ---------------------------------------------------------------------------

def _ocr_with_paddle(img_pil: Any) -> Tuple[str, float]:
    """Run PaddleOCR on a PIL image.

    Returns (text, mean_confidence) where confidence is 0.0–1.0.
    Returns ('', 0.0) if PaddleOCR is not installed.
    """
    global _paddle_ocr_engine
    try:
        from paddleocr import PaddleOCR  # type: ignore
        import numpy as np  # type: ignore

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        if _paddle_ocr_engine is None:
            _paddle_ocr_engine = PaddleOCR(
                lang="en",
                ocr_version="PP-OCRv4",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )

        img_arr = np.array(img_pil.convert("RGB"))
        result = _paddle_ocr_engine.predict(img_arr) or []

        texts: list[str] = []
        confidences: list[float] = []

        for page in result:
            if isinstance(page, dict):
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                for index, text in enumerate(rec_texts):
                    clean_text = str(text or "").strip()
                    if not clean_text:
                        continue
                    texts.append(clean_text)
                    if index < len(rec_scores):
                        confidences.append(float(rec_scores[index]))
                continue

        combined_text = "\n".join(texts)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return combined_text, mean_conf

    except ImportError:
        return "", 0.0
    except Exception as exc:  # noqa: BLE001
        log.debug("PaddleOCR failed: %s", exc)
        return "", 0.0


def _run_shared_paddle_ocr(pil_img: Any, source_label: str) -> Tuple[str, str]:
    """Run the shared PaddleOCR pipeline on one prepared image.

    This keeps image-file OCR and uploaded-image OCR on the same engine and
    preprocessing path, so operators see consistent OCR behaviour everywhere.
    """
    paddle_text, paddle_conf = _ocr_with_paddle(pil_img)
    if paddle_text:
        log.debug(
            "PaddleOCR confidence=%.2f coherence=%.2f for %s",
            paddle_conf,
            _ocr_confidence_score(paddle_text),
            source_label,
        )
    return paddle_text, "paddleocr"


def ocr_image_bytes_directly(image_bytes: bytes) -> Tuple[str, str]:
    """OCR uploaded image bytes using the shared PaddleOCR pipeline.

    PAN-card extraction uses this helper so UI uploads follow the same OCR path
    as saved image files and scanned PDFs.
    """
    from PIL import Image as _PILImage  # type: ignore

    try:
        from basetruth.analysis.preprocess import preprocess_pil_for_ocr

        with _PILImage.open(io.BytesIO(image_bytes)) as raw_img:
            pil_img = preprocess_pil_for_ocr(raw_img.copy())
    except Exception:  # noqa: BLE001
        try:
            with _PILImage.open(io.BytesIO(image_bytes)) as raw_img:
                pil_img = raw_img.copy()
        except Exception:  # noqa: BLE001
            return "", "error"

    return _run_shared_paddle_ocr(pil_img, "memory-image")


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

_PAN_RE = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def _ocr_confidence_score(text: str) -> float:
    """Estimate OCR quality from 0.0–1.0 based on text characteristics.

    Rules (each adds to the score):
    - Non-empty text                       +0.2
    - Contains recognisable words (a-z)    +0.2
    - Contains digits                      +0.1
    - PAN format found                     +0.3
    - Aadhaar format found                 +0.2
    Coherence penalty:
    - > 15% word-tokens are isolated single characters  -0.2
      (indicates garbled layout OCR on complex documents)
    """
    if not text or not text.strip():
        return 0.0

    score = 0.2
    if re.search(r"[a-zA-Z]{3,}", text):
        score += 0.2
    if re.search(r"\d{3,}", text):
        score += 0.1
    if _PAN_RE.search(text.upper()):
        score += 0.3
    if _AADHAAR_RE.search(text):
        score += 0.2

    # Coherence penalty: high rate of isolated single-character alphabetic tokens
    # is a strong indicator of failed layout OCR (e.g. complex Indian certificates).
    word_tokens = [w for w in re.split(r"[\s\n]+", text.strip()) if re.search(r"[a-zA-Z]", w)]
    if word_tokens:
        single_char_count = sum(1 for w in word_tokens if re.fullmatch(r"[a-zA-Z]", w))
        if single_char_count / len(word_tokens) > 0.15:
            score = max(0.0, score - 0.20)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Direct image OCR — PaddleOCR only
# ---------------------------------------------------------------------------

def ocr_image_directly(path: Path) -> Tuple[str, str]:
    """OCR a raw image file using PaddleOCR only.

    Pipeline:
      1. Preprocessing  — deskew + perspective correction + contrast enhance
      2. PaddleOCR      — the only OCR engine used in BaseTruth

    Returns
    -------
    (text, engine) where engine is one of:
      'paddleocr'      -- PaddleOCR ran (text may still be empty if the scan is unreadable)
      'unavailable'    -- PaddleOCR is not available
      'error'          -- unexpected exception
    """
    from PIL import Image as _PILImage  # type: ignore

    try:
        from basetruth.analysis.preprocess import preprocess_pil_for_ocr
        with _PILImage.open(str(path)) as raw_img:
            pil_img = preprocess_pil_for_ocr(raw_img.copy())
    except Exception:  # noqa: BLE001
        try:
            with _PILImage.open(str(path)) as raw_img:
                pil_img = raw_img.copy()
        except Exception:  # noqa: BLE001
            return "", "error"

    return _run_shared_paddle_ocr(pil_img, path.name)


def extract_image_file_metadata(path: Path) -> Dict[str, Any]:
    """Return basic file-level metadata for a raw image file.

    Mirrors the structure returned by :func:`extract_pdf_metadata` so downstream
    code can treat both uniformly.  EXIF / forensic metadata is handled by the
    :mod:`basetruth.analysis.image_forensics` module.
    """
    payload: Dict[str, Any] = {
        "available": False,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "is_image_file": True,
        "image_extension": path.suffix.lower(),
    }

    try:
        from PIL import Image  # type: ignore

        with Image.open(str(path)) as img:
            payload["available"] = True
            payload["image_width"] = img.width
            payload["image_height"] = img.height
            payload["image_mode"] = img.mode
            payload["image_format"] = img.format or path.suffix.upper().lstrip(".")
    except Exception as exc:  # noqa: BLE001
        payload["message"] = f"Could not open image: {exc}"

    return payload


def get_document_image_bytes(path: Path, max_dim: int = 1024) -> bytes | None:
    """Return image bytes (JPEG) suitable for a Gemma4 vision call.

    For raw image files (JPG, PNG, …) the file is read directly and rescaled.
    For PDF files the first page is rasterised via PyMuPDF or pdf2image /
    Poppler.  Returns ``None`` when no image can be produced.

    Parameters
    ----------
    path:
        Path to the source document.
    max_dim:
        Maximum width or height in pixels.  Larger images are resized keeping
        the aspect ratio.  1024 px is a good balance between Gemma4 accuracy
        and request size.
    """
    import io as _io

    def _pil_to_jpeg(pil_img: Any) -> bytes:
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)))
        buf = _io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    try:
        from PIL import Image as _PIL  # type: ignore
    except ImportError:
        return None

    # ── Raw image file ──────────────────────────────────────────────────────
    if is_image_file(path):
        try:
            with _PIL.open(str(path)) as img:
                return _pil_to_jpeg(img.copy())
        except Exception:  # noqa: BLE001
            return None

    # ── PDF — try PyMuPDF first (no Poppler needed) ─────────────────────────
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore  (PyMuPDF)

            doc = fitz.open(str(path))
            if doc.page_count > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)  # 2× zoom → ~144 dpi
                pix = page.get_pixmap(matrix=mat)
                img = _PIL.frombytes("RGB", (pix.width, pix.height), pix.samples)
                doc.close()
                return _pil_to_jpeg(img)
            doc.close()
        except Exception:  # noqa: BLE001
            pass  # Fall through to pdf2image

        # ── PDF — fallback to pdf2image / Poppler ───────────────────────────
        try:
            from pdf2image import convert_from_path  # type: ignore

            images = convert_from_path(str(path), dpi=150, first_page=1, last_page=1)
            if images:
                return _pil_to_jpeg(images[0])
        except Exception:  # noqa: BLE001
            pass

    return None
