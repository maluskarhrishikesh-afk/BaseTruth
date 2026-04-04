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
4. pytesseract + pdf2image (Poppler)
   - OCR tier for image-only PDFs (scanned Aadhaar, PAN cards)
   - Requires: pip install pytesseract pdf2image
   - Also requires: Tesseract OCR binary (tesseract.exe)
     Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Also requires: Poppler binaries on PATH for pdf2image
     Download: https://github.com/oschwartz10612/poppler-windows/releases
   - Gives full field extraction for identity documents
5. Empty string
   - Returned when all extraction methods fail; metadata forensics still run

How to get full scan for Aadhaar / PAN cards on Windows
---------------------------------------------------------
  Option A (recommended): Install ImageMagick
    https://imagemagick.org/script/download.php#windows
    After install, restart the terminal and retry.

  Option B: Install Tesseract + Poppler
    1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
       Add install dir to system PATH (e.g. C:\\Program Files\\Tesseract-OCR)
    2. Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
       Add poppler/bin to system PATH
    3. pip install pytesseract pdf2image

Public API
----------
  extract_pdf_metadata(path)               -> Dict
  extract_text_from_pdf(path)              -> str
  extract_text_via_ocr(path)               -> Tuple[str, str]  (text, engine)
  build_liteparse_json_from_text(text, src) -> Dict
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)


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
    OCR a PDF using pytesseract + pdf2image (Poppler) to extract text from
    image-only pages (scanned Aadhaar card, PAN card, photo-based PDFs).

    Unlike extract_text_from_pdf(), this function can read PDF pages that contain
    no embedded text at all -- it rasterises each page to an image, then OCRs it.

    Requirements (all optional -- function returns ('', 'unavailable') if missing):
      pip install pytesseract pdf2image
      Tesseract OCR binary: https://github.com/UB-Mannheim/tesseract/wiki
      Poppler binaries:     https://github.com/oschwartz10612/poppler-windows/releases

    Returns
    -------
    (text, engine) where engine is one of:
      'pytesseract'  -- OCR succeeded
      'unavailable'  -- pytesseract or Tesseract binary not installed
      'error'        -- unexpected exception during OCR
    """
    # --- pytesseract + pdf2image (Poppler) ---
    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore

        # Convert PDF pages to PIL images.  Poppler must be on PATH.
        images = convert_from_path(str(path), dpi=300)
        page_texts: list[str] = []

        try:
            from basetruth.analysis.preprocess import preprocess_pil_for_ocr as _preprocess
        except Exception:  # noqa: BLE001
            def _preprocess(im):  # type: ignore[misc]
                return im

        for img in images:
            preprocessed = _preprocess(img)
            # Try PaddleOCR first
            paddle_text, paddle_conf = _ocr_with_paddle(preprocessed)
            if paddle_text and paddle_conf >= 0.70:
                page_texts.append(paddle_text)
                continue
            # Fall back to pytesseract
            try:
                text = pytesseract.image_to_string(preprocessed, lang="eng+hin")
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(preprocessed, lang="eng")
            page_texts.append(paddle_text or text or "")
        return "\n".join(page_texts), "pytesseract"

    except ImportError:
        # pytesseract or pdf2image not installed.
        return "", "unavailable"
    except pytesseract.TesseractNotFoundError:  # type: ignore[name-defined]
        # Python packages present but Tesseract binary not on PATH.
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
# PaddleOCR — better accuracy than Tesseract for ID cards
# ---------------------------------------------------------------------------

def _ocr_with_paddle(img_pil: Any) -> Tuple[str, float]:
    """Run PaddleOCR on a PIL image.

    Returns (text, mean_confidence) where confidence is 0.0–1.0.
    Returns ('', 0.0) if PaddleOCR is not installed.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
        import numpy as np  # type: ignore

        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        img_arr = np.array(img_pil.convert("RGB"))
        result = ocr_engine.ocr(img_arr, cls=True)

        texts: list[str] = []
        confidences: list[float] = []

        if result:
            for page in result:
                if page:
                    for line in page:
                        # line is [[box], [text, confidence]]
                        if line and len(line) >= 2:
                            text_part = line[1]
                            if isinstance(text_part, (list, tuple)) and len(text_part) >= 2:
                                texts.append(str(text_part[0]))
                                confidences.append(float(text_part[1]))

        combined_text = "\n".join(texts)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return combined_text, mean_conf

    except ImportError:
        return "", 0.0
    except Exception as exc:  # noqa: BLE001
        log.debug("PaddleOCR failed: %s", exc)
        return "", 0.0


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

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Enhanced direct image OCR — PaddleOCR → Tesseract → VLM fallback
# ---------------------------------------------------------------------------

def ocr_image_directly(path: Path) -> Tuple[str, str]:
    """OCR a raw image file using a best-available engine with VLM fallback.

    Pipeline (in order):
      1. Preprocessing  — deskew + perspective correction + contrast enhance
      2. PaddleOCR      — better than Tesseract for ID card fonts
      3. pytesseract    — used if PaddleOCR unavailable OR confidence low
      4. VLM (Gemma)    — called when no PAN/Aadhaar regex was found in OCR output

    Returns
    -------
    (text, engine) where engine is one of:
      'paddleocr'      -- PaddleOCR succeeded with good confidence
      'pytesseract'    -- Tesseract OCR succeeded
      'gemma_local'    -- Gemma local model used as fallback
      'gemini_api'     -- Gemini API used as fallback
      'unavailable'    -- no OCR engine found
      'error'          -- unexpected exception
    """
    from PIL import Image as _PILImage  # type: ignore

    # --- Step 1: Preprocessing ---
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

    # --- Step 2: PaddleOCR (best for ID documents) ---
    paddle_text, paddle_conf = _ocr_with_paddle(pil_img)
    if paddle_text and paddle_conf >= 0.70:
        log.debug(
            "PaddleOCR confidence=%.2f for %s — using paddle result",
            paddle_conf, path.name,
        )
        return paddle_text, "paddleocr"

    # --- Step 3: pytesseract (always available fallback) ---
    tess_text = ""
    try:
        import pytesseract  # type: ignore

        try:
            tess_text = pytesseract.image_to_string(pil_img, lang="eng+hin") or ""
        except pytesseract.TesseractError:
            tess_text = pytesseract.image_to_string(pil_img, lang="eng") or ""

        # Use Tesseract if it found a PAN/Aadhaar, or if PaddleOCR is unavailable
        if tess_text and (
            not paddle_text
            or _ocr_confidence_score(tess_text) > _ocr_confidence_score(paddle_text)
        ):
            combined = tess_text
        else:
            combined = paddle_text or tess_text

    except ImportError:
        combined = paddle_text
    except Exception as exc:  # noqa: BLE001
        log.debug("pytesseract failed for %s: %s", path.name, exc)
        combined = paddle_text

    # --- Step 4: VLM fallback when no identity fields detected ---
    if not combined or _ocr_confidence_score(combined) < 0.4:
        log.debug(
            "OCR confidence low for %s — trying VLM fallback", path.name
        )
        try:
            from basetruth.integrations.gemma_vlm import extract_text_with_vlm
            vlm_text, vlm_engine = extract_text_with_vlm(path)
            if vlm_text:
                log.debug(
                    "VLM (%s) extracted %d chars for %s",
                    vlm_engine, len(vlm_text), path.name,
                )
                return vlm_text, vlm_engine
        except Exception as exc:  # noqa: BLE001
            log.debug("VLM fallback failed for %s: %s", path.name, exc)

    if combined:
        engine = "paddleocr" if paddle_text and not tess_text else "pytesseract"
        return combined, engine

    return "", "unavailable"


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
