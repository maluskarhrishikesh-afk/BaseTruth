from __future__ import annotations

"""
PDF integration helpers for BaseTruth.

Two responsibilities:
  1. extract_pdf_metadata()       -- structural and descriptive metadata from a PDF.
  2. extract_text_from_pdf()      -- plain-text fallback when LiteParse cannot run.
  3. build_liteparse_json_from_text() -- wraps extracted text in the LiteParse JSON
                                        schema so the rest of the pipeline can consume
                                        it without any code changes.

The text-extraction path is used automatically by service.scan_document() when
LiteParse fails (e.g. ImageMagick not installed on Windows, or the PDF is an
image-only scan such as an Aadhaar or PAN card).  The tamper/metadata layer and
domain validators still run in full; field-level extraction will be limited for
image-only documents that have no embedded text layer.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict


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
    Extract all plain text from a PDF using pypdf.

    This is the fallback path used when LiteParse cannot process a document
    (e.g. ImageMagick is unavailable on Windows, or the file requires PDF-to-image
    rasterisation before OCR).  The extracted text is lower-quality than a full
    LiteParse parse but is sufficient for heuristic tamper scoring and arithmetic
    validation on text-based PDFs.

    For image-only PDFs (scanned Aadhaar cards, PAN cards) the text will be empty
    because there is no embedded text layer.  The PDF metadata and structural
    forensics (signature markers, header version, creation-date consistency) still
    run -- so the truth score reflects the document structure even without any
    field-level data.

    Returns a single newline-joined string of all page text, or an empty string if
    pypdf is not installed or the file cannot be read.
    """
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            page_texts.append(text)
        return "\n".join(page_texts)
    except Exception:  # noqa: BLE001
        # Any failure (ImportError, encrypted PDF, corrupt file) is caught here
        # so the caller can proceed gracefully with an empty text string.
        return ""


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
