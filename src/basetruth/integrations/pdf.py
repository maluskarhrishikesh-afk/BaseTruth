from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_raw_signature_markers(pdf_bytes: bytes) -> list[str]:
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
    payload["pdf_header"] = pdf_bytes[:8].decode("latin-1", errors="ignore")
    payload["signature_markers"] = _extract_raw_signature_markers(pdf_bytes)
    payload["has_digital_signature_markers"] = bool(payload["signature_markers"])
    payload["reader"] = "raw"

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
    except (ImportError, OSError, TypeError, ValueError) as exc:  # pragma: no cover - optional dependency path
        payload["metadata"] = {}
        payload["metadata_error"] = str(exc)

    return payload
