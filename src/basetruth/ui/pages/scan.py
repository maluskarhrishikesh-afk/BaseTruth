"""Scan Document page — smart document classification and field extraction.

This page is a pure extraction tool.  It does NOT run forensics and does NOT
save anything to the database.  The workflow is:

1. User uploads any document (PDF, image, etc.)
2. The document goes directly to Gemma AI (gemma-4-31b-it) which simultaneously
   classifies the document type AND extracts all fields in a single LLM call.
   No separate pre-classifier — the LLM handles any document type without
   being limited to a fixed list.
3. Based on the file content:
   - Scanned images / scanned PDFs → PaddleOCR extracts text + bounding-box
     coordinates so the LLM can understand the spatial layout.
   - Structured / digital PDFs → the real PDF file (all pages, full vector
     text) is sent directly to Gemini — no JPEG render, no OCR noise.
4. Extracted fields and the LLM's own confidence score are displayed on screen
   as a styled JSON block.

No data is persisted to PostgreSQL or MinIO from this screen.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import _page_title
from basetruth.logger import get_logger

log = get_logger(__name__)

# ── Supported image file extensions ────────────────────────────────────────
_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── CSS for the JSON result panel ──────────────────────────────────────────
_RESULT_CSS = """
<style>
/* ── Scan result container ─────────────────────────────────────────── */
.bt-scan-result-wrap {
    border-radius: 14px;
    border: 1.5px solid rgba(99, 102, 241, 0.18);
    background: #0d1117;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.22);
}

/* ── Result header bar ─────────────────────────────────────────────── */
.bt-scan-result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 18px;
    background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(139,92,246,0.09) 100%);
    border-bottom: 1px solid rgba(99,102,241,0.16);
}
.bt-scan-result-header .bt-doc-type-badge {
    background: rgba(99,102,241,0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.bt-scan-result-header .bt-scan-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8rem;
    color: #64748b;
}

/* ── JSON body ─────────────────────────────────────────────────────── */
.bt-scan-json {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.83rem;
    line-height: 1.6;
    padding: 18px 22px;
    color: #c9d1d9;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 70vh;
    overflow-y: auto;
}

/* JSON syntax colouring */
.bt-scan-json .jk  { color: #79c0ff; }   /* key   */
.bt-scan-json .jvs { color: #a5d6ff; }   /* string value */
.bt-scan-json .jvn { color: #e3b341; }   /* number value */
.bt-scan-json .jvb { color: #ff7b72; }   /* bool/null */
.bt-scan-json .jp  { color: #8b949e; }   /* punctuation */

/* ── Classification banner ─────────────────────────────────────────── */
.bt-classify-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.15);
}
.bt-classify-banner .bt-classify-icon {
    font-size: 2rem;
    flex-shrink: 0;
}
.bt-classify-banner .bt-classify-detail {
    flex: 1;
}
.bt-classify-banner .bt-classify-type {
    font-size: 1.05rem;
    font-weight: 700;
    color: #a5b4fc;
    text-transform: capitalize;
}
.bt-classify-banner .bt-classify-meta {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Step indicators ───────────────────────────────────────────────── */
.bt-step-row {
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 10px 16px;
    border-radius: 10px;
    background: rgba(99,102,241,0.04);
    border: 1px solid rgba(99,102,241,0.10);
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #94a3b8;
}
.bt-step-row .bt-step-num {
    background: rgba(99,102,241,0.16);
    color: #818cf8;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.75rem;
    flex-shrink: 0;
}
</style>
"""


def _doc_type_icon(doc_type: str) -> str:
    """Return an emoji that represents the document type at a glance."""
    _icons: Dict[str, str] = {
        "payslip": "💰", "bank_statement": "🏦", "form16": "📑",
        "offer_letter": "📄", "employment_letter": "👔", "increment_letter": "📈",
        "gift_letter": "🎁", "marksheet": "📋", "degree_certificate": "🎓",
        "aadhaar": "🪪", "pan_card": "🆔", "passport": "🛂",
        "invoice": "🧾", "receipt": "🧾", "hospital_bill": "🏥",
        "utility_bill": "⚡", "property_agreement": "🏠",
        "experience_letter": "📜", "relieving_letter": "📜",
        "cancelled_cheque": "💳", "photograph": "📷",
        "signature": "✍️", "certificate": "🏅",
    }
    return _icons.get(doc_type.lower().replace(" ", "_"), "📄")


def _syntax_highlight_json(data: Dict[str, Any]) -> str:
    """Convert a dict to a syntax-highlighted HTML string for the dark panel.

    We manually render the JSON (instead of st.json) so it sits inside our
    custom dark container.  Keys: blue, string values: cyan, numbers: amber,
    booleans/null: red.
    """
    import html as _html  # noqa: PLC0415

    raw = json.dumps(data, indent=2, ensure_ascii=False)
    lines = []
    for line in raw.split("\n"):
        # Keys: "some_key":
        line = re.sub(
            r'"([^"]+)"(\s*:)',
            lambda m: f'<span class="jk">"{_html.escape(m.group(1))}"</span><span class="jp">{m.group(2)}</span>',
            line,
        )
        # String values: : "some value"
        line = re.sub(
            r'(:\s*)"([^"]*)"',
            lambda m: f'<span class="jp">{m.group(1)}</span><span class="jvs">"{_html.escape(m.group(2))}"</span>',
            line,
        )
        # Numbers: : 123 or : 1.5
        line = re.sub(
            r"(:\s*)(-?\d+(?:\.\d+)?)",
            lambda m: f'<span class="jp">{m.group(1)}</span><span class="jvn">{m.group(2)}</span>',
            line,
        )
        # Booleans / null
        line = re.sub(
            r"\b(true|false|null)\b",
            lambda m: f'<span class="jvb">{m.group(1)}</span>',
            line,
        )
        lines.append(line)
    return "\n".join(lines)


def _filter_display_fields(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal meta-keys (prefixed with _) from the extracted dict.

    Keys like _unavailable, _extraction_attempts, and _validation_errors are
    for debugging — they should not appear in the user-facing output panel.
    """
    return {k: v for k, v in extracted.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Core extraction logic — shared between the UI page and the REST API
# ---------------------------------------------------------------------------

def _bytes_to_temp_path(file_bytes: bytes, filename: str) -> Path:
    """Write raw bytes to a named temporary file and return its path.

    Several helpers in the pipeline (get_document_image_bytes, PyMuPDF) need
    a seekable file on disk rather than an in-memory buffer.  We create a
    predictably-named temp file so the original filename is preserved (the
    extension drives format detection).  The OS cleans up on process exit.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="bt_scan_"))
    tmp_path = tmp_dir / filename
    tmp_path.write_bytes(file_bytes)
    return tmp_path


# Maps the LLM's string confidence levels to a 0–1 float for the UI banner.
_CONF_MAP: Dict[str, float] = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.4}


def _infer_is_image_based(file_bytes: bytes, filename: str) -> bool:
    """Return True if the document is a scanned image / image-only PDF.

    Strategy:
    - Known image extensions (jpg, png, tiff, etc.) → always True.
    - PDF: open with PyMuPDF and check page-0 embedded text.  Less than 200
      characters of extractable text means it is a scanned page (image-only
      PDF); 200+ chars means it is a digitally-generated structured PDF.
    - Anything else defaults to False (treat as structured).
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return True
    if suffix == ".pdf" or file_bytes[:4] == b"%PDF":
        try:
            import fitz  # PyMuPDF  # noqa: PLC0415
            _doc = fitz.open(stream=file_bytes, filetype="pdf")
            _raw_text = _doc[0].get_text() if _doc.page_count > 0 else ""
            _txt = _raw_text.strip() if isinstance(_raw_text, str) else str(_raw_text).strip()
            _doc.close()
            # Fewer than 200 chars of embedded text → scanned / image-based
            return len(_txt) < 200
        except Exception:
            return False  # if we can't open it, assume structured
    return False


def extract_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract structured fields from a document using Gemma AI.

    This is the single entry point used by both the Streamlit page and the
    REST API endpoint.  There is no separate pre-classifier step — the LLM
    determines the document type itself as part of extraction, so it works
    for any document type without being limited to a fixed list.

    Pipeline
    ────────
    1. Determine is_image_based by inspecting the file type / embedded text.
    2. Call extract_document_fields with doc_type="generic" so the LLM
       classifies and extracts in one pass.  Internally:
       - Scanned images / scanned PDFs → PaddleOCR (PP-OCRv4) extracts text
         with [y=Npx x=Npx] bounding-box coordinates → sent to Gemma AI.
       - Structured digital PDFs → real PDF file (all pages, full vector
         text) sent directly to Gemini — no JPEG render, no OCR noise.
    3. Unwrap the extraction envelope if the LLM returned nested keys.
    4. Read document_type and extraction_confidence from the LLM's own output.

    Return keys
    ───────────
    filename, document_type, is_image_based, confidence, extracted_fields.
    On failure: error key is also set.  Never raises.
    """
    log.info(
        "scan_page: extract_document called",
        extra={"doc_filename": filename, "size_bytes": len(file_bytes)},
    )

    try:
        from basetruth.integrations.document_extract import extract_document_fields  # noqa: PLC0415

        # Step 1: detect whether the file is scanned/image-based or structured.
        # This drives the badge shown in the UI and is purely informational —
        # extract_document_fields handles the routing internally regardless.
        is_image_based = _infer_is_image_based(file_bytes, filename)

        # Step 2: extract — doc_type="generic" lets the LLM determine the type
        # itself as part of extraction rather than from a restricted classifier.
        extracted = extract_document_fields(
            file_bytes,
            doc_type="generic",
            filename=filename,
        )

        # Step 3: unwrap the envelope if the LLM returned a nested structure:
        #   { "document_type": "SSC Marksheet",
        #     "extracted_fields": { ...actual fields... },
        #     "data_quality_notes": [], "extraction_confidence": "HIGH" }
        if "extracted_fields" in extracted and isinstance(extracted.get("extracted_fields"), dict):
            nested = extracted.pop("extracted_fields")
            # Fold remaining envelope keys into the flat field dict so they
            # appear in the JSON panel (e.g. data_quality_notes)
            for k, v in extracted.items():
                if k not in nested:
                    nested[k] = v
            extracted = nested

        # Step 4a: read document_type from the LLM's own response — accurate
        # for any document type, not limited to a pre-defined classifier list.
        doc_type = str(extracted.pop("document_type", "") or "").strip() or "unknown"

        # Step 4b: read extraction_confidence from the LLM's response and
        # convert to a 0–1 float.  The LLM returns HIGH / MEDIUM / LOW.
        conf_raw = str(extracted.get("extraction_confidence", "") or "").strip().upper()
        confidence = _CONF_MAP.get(conf_raw, 0.0)
        if not confidence:
            # Forward-compat: handle numeric confidence strings (e.g. "0.9")
            try:
                confidence = float(conf_raw) if conf_raw else 0.0
            except ValueError:
                confidence = 0.0

        log.info(
            "scan_page: extraction complete",
            extra={
                "doc_filename": filename,
                "doc_type": doc_type,
                "confidence": confidence,
                "is_image_based": is_image_based,
                "field_count": len([k for k, v in extracted.items() if not k.startswith("_") and v]),
            },
        )

        return {
            "filename": filename,
            "document_type": doc_type,
            "is_image_based": is_image_based,
            "confidence": confidence,
            "extracted_fields": extracted,
        }

    except Exception as exc:  # noqa: BLE001
        log.error(
            "scan_page: extraction failed",
            extra={"doc_filename": filename, "error": str(exc)},
            exc_info=True,
        )
        return {
            "filename": filename,
            "document_type": "unknown",
            "is_image_based": False,
            "confidence": 0.0,
            "extracted_fields": {},
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

def _page_scan(_service: BaseTruthService = None) -> None:  # type: ignore[assignment]
    """Render the Scan Document page.

    The _service argument is accepted for compatibility with app.py's call
    convention but is not used by this screen — all logic is self-contained.
    """
    # Inject styles
    st.markdown(_RESULT_CSS, unsafe_allow_html=True)
    st.markdown(_page_title("🔍", "Scan Document"), unsafe_allow_html=True)

    # ── How-to guide ─────────────────────────────────────────────────────
    with st.expander("ℹ️ How this screen works", expanded=False):
        st.markdown(
            """
**Scan Document** classifies any document and extracts its structured data using AI.

<div style="margin:10px 0">
<div class="bt-step-row"><div class="bt-step-num">1</div>
Upload any document — payslip, marksheet, bank statement, offer letter, invoice, hospital bill, challan, etc.</div>
<div class="bt-step-row"><div class="bt-step-num">2</div>
Gemma AI <strong>classifies and extracts in one pass</strong> — no restricted pre-classifier, so it works for any document type.</div>
<div class="bt-step-row"><div class="bt-step-num">3</div>
<strong>Scanned documents:</strong> PaddleOCR (PP-OCRv4) extracts text with pixel bounding boxes so the AI understands the spatial layout — which label belongs to which value column.</div>
<div class="bt-step-row"><div class="bt-step-num">4</div>
<strong>Digital PDFs:</strong> the actual PDF file is sent directly to Gemini — all pages, exact embedded text, no JPEG compression.</div>
<div class="bt-step-row"><div class="bt-step-num">5</div>
The final response JSON includes the LLM-classified document type and extracted fields.</div>
</div>

> **No data is saved.** This screen is for on-demand extraction only.
""",
            unsafe_allow_html=True,
        )

    # ── File uploader ─────────────────────────────────────────────────────
    upload = st.file_uploader(
        "Drop a document here to extract its data",
        type=None,
        accept_multiple_files=False,
        key="scan_doc_upload",
        label_visibility="visible",
        help="Supported: PDF, JPG, PNG, TIFF, BMP, WebP — any document type",
    )

    # Show a placeholder prompt before anything is uploaded
    if upload is None:
        st.markdown(
            """
            <div style="text-align:center; padding:2.5rem 1rem; color:#64748b;">
              <div style="font-size:2.5rem; margin-bottom:0.75rem;">📄</div>
              <div style="font-size:0.95rem; font-weight:600;">Upload a document above to begin extraction.</div>
              <div style="font-size:0.8rem; margin-top:0.5rem; color:#475569;">
                Works with any document — payslips, marksheets, bank statements, invoices,
                offer letters, ID cards, hospital bills, challans, and more.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Run extraction ─────────────────────────────────────────────────────
    file_bytes = upload.read()
    filename = upload.name or "document"

    # Cache result keyed on filename + size so re-renders don't re-run the
    # (potentially slow) LLM + OCR pipeline on every Streamlit re-render.
    cache_key = f"scan_result_{filename}_{len(file_bytes)}"
    if st.session_state.get(cache_key) is None:
        with st.spinner("🧠 Classifying document and extracting fields…"):
            result = extract_document(file_bytes, filename)
        st.session_state[cache_key] = result
    else:
        result = st.session_state[cache_key]

    if not result:
        return

    # ── Error handling ─────────────────────────────────────────────────────
    if result.get("error"):
        st.error(
            f"**Extraction failed:** {result['error']}\n\n"
            "Check that:\n"
            "- PaddleOCR is installed: `pip install paddleocr paddlepaddle`\n"
            "- The Google AI API key is set in `artifacts/config/settings.json`"
        )
        return

    doc_type: str = result.get("document_type", "unknown")
    is_image: bool = result.get("is_image_based", False)
    extracted: Dict[str, Any] = result.get("extracted_fields", {})
    display_fields = _filter_display_fields(extracted)
    user_payload = {
        "filename": filename,
        "document_type": doc_type,
        "is_image_based": is_image,
        "extracted_fields": display_fields,
    }

    # ── Classification banner ──────────────────────────────────────────────
    icon = _doc_type_icon(doc_type)
    scan_method = "Scanned / Image-based" if is_image else "Digital / Structured"
    st.markdown(
        f"""
        <div class="bt-classify-banner">
          <div class="bt-classify-icon">{icon}</div>
          <div class="bt-classify-detail">
            <div class="bt-classify-type">{doc_type.replace("_", " ").title()}</div>
            <div class="bt-classify-meta">
                            {scan_method} &nbsp;·&nbsp; File: <strong>{filename}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Extraction quality note ────────────────────────────────────────────
    if extracted.get("_unavailable"):
        st.warning(
            "⚠️ Gemma AI was unavailable. "
            "Check that the Google AI API key is set in `artifacts/config/settings.json`."
        )
    elif not display_fields:
        st.warning(
            "No fields were extracted. The document may be blank, password-protected, or unsupported."
        )
    else:
        non_empty = len([v for v in display_fields.values() if v not in (None, "", [], {})])
        st.caption(f"✅ **{non_empty} field(s) extracted** — review the structured output below.")

    # ── JSON result panel ──────────────────────────────────────────────────
    if display_fields:
        highlighted = _syntax_highlight_json(user_payload)
        st.markdown(
            f"""
            <div class="bt-scan-result-wrap">
                <div class="bt-scan-result-header">
                    <span class="bt-doc-type-badge">{doc_type.replace("_", " ")}</span>
                    <span class="bt-scan-badge">
                        {'🖼 PaddleOCR + Gemma AI' if is_image else '📑 Embedded Text + Gemma AI'}
                    </span>
                </div>
                <div class="bt-scan-json">{highlighted}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Download + reset row ───────────────────────────────────────────────
    col_dl, col_reset = st.columns([3, 1])
    with col_dl:
        st.download_button(
            "⬇ Download extracted data as JSON",
            data=json.dumps(user_payload, indent=2, ensure_ascii=False),
            file_name=f"{Path(filename).stem}_extracted.json",
            mime="application/json",
            use_container_width=True,
            key="scan_dl_btn",
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True, key="scan_reset_btn",
                     help="Clear result and scan another document"):
            for k in [k for k in st.session_state if k.startswith("scan_")]:
                del st.session_state[k]
            st.rerun()

