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
        for img in images:
            # OCR with English + Hindi languages when available; fall back to eng.
            try:
                text = pytesseract.image_to_string(img, lang="eng+hin")
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(img, lang="eng")
            page_texts.append(text or "")
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


def ocr_image_directly(path: Path) -> Tuple[str, str]:
    """OCR a raw image file (JPEG, PNG, TIFF…) using pytesseract.

    Unlike :func:`extract_text_via_ocr` this function does NOT need pdf2image /
    Poppler because it feeds the image directly to Tesseract.

    Returns
    -------
    (text, engine) where engine is one of:
      'pytesseract'  -- OCR succeeded
      'unavailable'  -- pytesseract or its binary are not installed
      'error'        -- unexpected exception
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        with Image.open(str(path)) as img:
            try:
                text = pytesseract.image_to_string(img, lang="eng+hin")
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(img, lang="eng")
        return text or "", "pytesseract"

    except ImportError:
        return "", "unavailable"
    except Exception as exc:  # noqa: BLE001
        log.debug("Direct image OCR failed for %s: %s", path.name, exc)
        return "", "error"


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
