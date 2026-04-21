"""Swagger page — exposes API documentation links for operators."""
from __future__ import annotations

import streamlit as st

from basetruth.ui.components import _page_title

# ── CSS shared across all endpoint cards ──────────────────────────────────────
_SWAGGER_CSS = """
<style>
/* ── Base card ────────────────────────────────────────────────────────────── */
.bt-api-card {
    border-radius: 14px;
    border: 1.5px solid rgba(99,102,241,0.18);
    background: #0d1117;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.22);
    margin-bottom: 28px;
}

/* ── Card header bar ──────────────────────────────────────────────────────── */
.bt-api-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.13) 0%, rgba(139,92,246,0.08) 100%);
    border-bottom: 1px solid rgba(99,102,241,0.16);
    flex-wrap: wrap;
}
.bt-method-badge {
    background: #238636;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 4px 11px;
    border-radius: 6px;
    flex-shrink: 0;
}
.bt-api-path {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.01em;
}
.bt-api-summary {
    font-size: 0.82rem;
    color: #64748b;
    margin-left: auto;
    font-style: italic;
}

/* ── Section inside card ──────────────────────────────────────────────────── */
.bt-api-section {
    padding: 14px 22px 6px 22px;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
.bt-api-section:last-child { border-bottom: none; padding-bottom: 18px; }
.bt-section-label {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 10px;
}

/* ── Description text ─────────────────────────────────────────────────────── */
.bt-api-desc {
    font-size: 0.87rem;
    color: #94a3b8;
    line-height: 1.6;
    margin: 0;
}

/* ── Schema table ─────────────────────────────────────────────────────────── */
.bt-schema-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.bt-schema-table th {
    text-align: left;
    color: #475569;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 5px 10px;
    border-bottom: 1px solid rgba(99,102,241,0.14);
}
.bt-schema-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(99,102,241,0.06);
    vertical-align: top;
}
.bt-schema-table tr:last-child td { border-bottom: none; }
.bt-field-name {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    color: #79c0ff;
    white-space: nowrap;
}
.bt-field-type {
    color: #e3b341;
    white-space: nowrap;
}
.bt-field-req  { color: #ef4444; font-weight: 700; font-size: 0.8rem; }
.bt-field-opt  { color: #475569; font-weight: 700; font-size: 0.8rem; }
.bt-field-desc { color: #94a3b8; }

/* ── Content-type pill ────────────────────────────────────────────────────── */
.bt-ct-pill {
    display: inline-block;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.24);
    border-radius: 999px;
    color: #a5b4fc;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 2px 10px;
    margin-bottom: 10px;
}

/* ── Sub-object note ──────────────────────────────────────────────────────── */
.bt-sub-note {
    font-size: 0.78rem;
    color: #475569;
    font-style: italic;
    margin: 4px 0 10px 10px;
}

/* ── Live links bar ───────────────────────────────────────────────────────── */
.bt-links-bar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    padding: 18px 0 6px 0;
}
.bt-link-btn {
    display: inline-block;
    padding: 9px 20px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    text-decoration: none !important;
    letter-spacing: 0.02em;
}
.bt-link-primary {
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc !important;
}
.bt-link-secondary {
    background: rgba(14,116,144,0.12);
    border: 1px solid rgba(14,116,144,0.25);
    color: #67e8f9 !important;
}
</style>
"""


def _schema_row(name: str, type_: str, required: bool, description: str) -> str:
    """Render one HTML table row for a request/response schema field."""
    req_cell = (
        '<span class="bt-field-req">required</span>'
        if required
        else '<span class="bt-field-opt">optional</span>'
    )
    return (
        f"<tr>"
        f'<td><span class="bt-field-name">{name}</span></td>'
        f'<td><span class="bt-field-type">{type_}</span></td>'
        f"<td>{req_cell}</td>"
        f'<td class="bt-field-desc">{description}</td>'
        f"</tr>"
    )


def _verdict_enum_note() -> str:
    return (
        '<span class="bt-field-type">ORIGINAL</span> · '
        '<span class="bt-field-type">UNCERTAIN</span> · '
        '<span class="bt-field-type">LIKELY TAMPERED</span> · '
        '<span class="bt-field-type">TAMPERED</span> · '
        '<span class="bt-field-type">UNAVAILABLE</span>'
    )


def _page_swagger() -> None:
    """Render the Swagger / API specification page."""
    st.markdown(_SWAGGER_CSS, unsafe_allow_html=True)
    st.markdown(_page_title("📘", "Swagger"), unsafe_allow_html=True)

    # ── Live links ──────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="bt-links-bar">
  <a class="bt-link-btn bt-link-primary" href="http://localhost:8000/docs" target="_blank">
    📖 Interactive Swagger UI
  </a>
  <a class="bt-link-btn bt-link-secondary" href="http://localhost:8000/redoc" target="_blank">
    📄 ReDoc
  </a>
  <a class="bt-link-btn bt-link-secondary" href="http://localhost:8000/openapi.json" target="_blank">
    { } OpenAPI JSON
  </a>
</div>
<p style="font-size:0.8rem;color:#475569;margin:6px 0 24px 0;">
  BaseTruth API · Base URL: <code style="color:#a5b4fc;">http://localhost:8000</code>
  · Start the API server before opening these links.
</p>
""",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # ENDPOINT 1 — POST /api/v1/document-extract
    # ════════════════════════════════════════════════════════════════════════
    req_rows_extract = "".join([
        _schema_row("file", "File (binary)", True,
                    "The document to analyse. Accepted formats: PDF, JPG, PNG, TIFF, BMP, WebP."),
    ])
    resp_rows_extract = "".join([
        _schema_row("filename",          "string",  True,  "Original filename as uploaded."),
        _schema_row("document_type",     "string",  True,  "LLM-classified document type (e.g. <em>payslip</em>, <em>bank_statement</em>, <em>passport</em>)."),
        _schema_row("is_image_based",    "boolean", True,  "<code>true</code> for scanned / image PDFs; <code>false</code> for structured / digital PDFs."),
        _schema_row("confidence",        "float",   True,  "Extraction confidence reported by the LLM. Range: <code>0.0</code> – <code>1.0</code> (1.0 = HIGH, 0.5 = MEDIUM, 0.25 = LOW)."),
        _schema_row("extracted_fields",  "object",  True,  "Key–value map of all extracted document fields. Keys vary by document type (e.g. <em>employee_name</em>, <em>net_salary</em>, <em>account_number</em>)."),
    ])

    st.markdown(
        f"""
<div class="bt-api-card">
  <div class="bt-api-header">
    <span class="bt-method-badge">POST</span>
    <span class="bt-api-path">/api/v1/document-extract</span>
    <span class="bt-api-summary">Extract structured fields from a document</span>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Description</div>
    <p class="bt-api-desc">
      Upload any identity or financial document. BaseTruth automatically detects
      whether the file is scanned (image-based) or structured (digital PDF), then
      routes it through the correct OCR + LLM extraction pipeline.<br><br>
      The LLM (<strong>Gemma / Gemini</strong>) classifies the document type and extracts all
      relevant fields in a single pass — no predefined template or document-type hint
      is required. The response includes confidence, scan method, and a flat
      key–value map of all extracted fields.
    </p>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Request · multipart/form-data</div>
    <span class="bt-ct-pill">Content-Type: multipart/form-data</span>
    <table class="bt-schema-table">
      <tr><th>Field</th><th>Type</th><th></th><th>Description</th></tr>
      {req_rows_extract}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Response 200 · application/json</div>
    <span class="bt-ct-pill">Content-Type: application/json</span>
    <table class="bt-schema-table">
      <tr><th>Field</th><th>Type</th><th></th><th>Description</th></tr>
      {resp_rows_extract}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Error Responses</div>
    <table class="bt-schema-table">
      <tr><th>Status</th><th>When</th></tr>
      <tr><td><span class="bt-field-type">422</span></td><td class="bt-field-desc">No file uploaded or wrong field name.</td></tr>
      <tr><td><span class="bt-field-type">500</span></td><td class="bt-field-desc">Extraction pipeline failed (OCR or LLM error). <code>detail</code> field contains the error message.</td></tr>
    </table>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # ENDPOINT 2 — POST /api/v1/forensic-scan
    # ════════════════════════════════════════════════════════════════════════
    req_rows_forensic = "".join([
        _schema_row("file", "File (binary)", True,
                    "The document to analyse. Accepted formats: PDF, JPG, PNG, TIFF, BMP, WebP."),
    ])
    resp_rows_forensic = "".join([
        _schema_row("filename",              "string",  True, "Original filename as uploaded."),
        _schema_row("document_type",         "string",  True, "LLM-classified document type."),
        _schema_row("is_image_based",        "boolean", True, "<code>true</code> for scanned / image PDFs; <code>false</code> for structured PDFs."),
        _schema_row("forensic_verdict",      "string",  True, f"Overall tamper verdict. One of: {_verdict_enum_note()}."),
        _schema_row("forgery_score_0_100",   "float",   True, "Composite forgery risk score. <code>0</code> = pristine, <code>100</code> = highly tampered."),
        _schema_row("overall_explanation",   "string",  True, "Human-readable summary: how many layers fired, and the key evidence items."),
        _schema_row("evidence",              "string[]", True, "List of the most significant suspicious signals across all layers."),
        _schema_row("layers",                "object",  True, "Full per-layer breakdown. See layer schema below."),
    ])
    layer_rows = "".join([
        _schema_row("layer_1_ela",        "LayerResult", False, "<strong>Error Level Analysis</strong> — detects re-compressed regions by measuring JPEG quantisation residuals."),
        _schema_row("layer_2_metadata",   "LayerResult", False, "<strong>Metadata</strong> — inspects EXIF/XMP tags; stripped metadata is a common post-edit signal."),
        _schema_row("layer_3_entropy",    "LayerResult", False, "<strong>Entropy</strong> — Shannon entropy of raw bytes; unusually uniform or high entropy can indicate insertion."),
        _schema_row("layer_4_noise",      "LayerResult", False, "<strong>Noise consistency</strong> — sensor noise should be uniform; localised anomalies suggest splicing."),
        _schema_row("layer_5_dct",        "LayerResult", False, "<strong>DCT / Double-JPEG</strong> — double-compression ghost artefacts visible in DCT coefficient histogram (JPEG only)."),
        _schema_row("layer_6_clone",      "LayerResult", False, "<strong>Clone / copy-move</strong> — detects copy-pasted regions using feature matching."),
        _schema_row("layer_7_color",      "LayerResult", False, "<strong>Colour anomaly</strong> — flags pixels outside the document's natural colour palette."),
        _schema_row("layer_8_edge",       "LayerResult", False, "<strong>Edge density</strong> — unnatural sharp-edge concentrations indicate pasted content."),
        _schema_row("layer_9_saturation", "LayerResult", False, "<strong>Saturation</strong> — localised over-saturation blobs often appear after digital editing."),
        _schema_row("layer_10_font",      "LayerResult", False, "<strong>Font consistency</strong> — stroke-width coefficient of variation catches mixed font origins."),
        _schema_row("layer_11_ai",        "LayerResult", False, "<strong>AI-generation artefacts</strong> — FFT-based frequency analysis to detect AI-generated textures."),
    ])
    layer_result_rows = "".join([
        _schema_row("name",          "string", True, "Human-readable layer name."),
        _schema_row("status",        "string", True, "One of: <span class='bt-field-type'>CLEAN</span> · <span class='bt-field-type'>SUSPICIOUS</span> · <span class='bt-field-type'>N/A</span> · <span class='bt-field-type'>ERROR</span>."),
        _schema_row("plain_english", "string", True, "Plain-English explanation of what this layer found."),
        _schema_row("metrics",       "object", True, "Raw numeric metrics used for scoring (content varies by layer)."),
    ])

    st.markdown(
        f"""
<div class="bt-api-card">
  <div class="bt-api-header">
    <span class="bt-method-badge">POST</span>
    <span class="bt-api-path">/api/v1/forensic-scan</span>
    <span class="bt-api-summary">Tamper-detection forensic analysis</span>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Description</div>
    <p class="bt-api-desc">
      Upload any document to run a <strong>full 11-layer forensic analysis</strong>.
      BaseTruth routes the file to the correct engine automatically — image forensics
      for scanned documents, PDF forensics for structured/digital PDFs.<br><br>
      Each layer examines a different tamper signal: ELA compression artefacts,
      metadata stripping, sensor noise inconsistency, copy-move cloning, colour
      palette anomalies, font stroke variation, and AI-generation patterns.
      The response contains the overall verdict, a composite score, high-level
      evidence strings, and the complete per-layer breakdown with raw metrics.
    </p>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Request · multipart/form-data</div>
    <span class="bt-ct-pill">Content-Type: multipart/form-data</span>
    <table class="bt-schema-table">
      <tr><th>Field</th><th>Type</th><th></th><th>Description</th></tr>
      {req_rows_forensic}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Response 200 · application/json</div>
    <span class="bt-ct-pill">Content-Type: application/json</span>
    <table class="bt-schema-table">
      <tr><th>Field</th><th>Type</th><th></th><th>Description</th></tr>
      {resp_rows_forensic}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">layers — per-layer keys</div>
    <p class="bt-sub-note">Each key in <code>layers</code> corresponds to one forensic layer.</p>
    <table class="bt-schema-table">
      <tr><th>Key</th><th>Type</th><th></th><th>Technique</th></tr>
      {layer_rows}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">LayerResult — object schema</div>
    <p class="bt-sub-note">All layer entries share this structure.</p>
    <table class="bt-schema-table">
      <tr><th>Field</th><th>Type</th><th></th><th>Description</th></tr>
      {layer_result_rows}
    </table>
  </div>

  <div class="bt-api-section">
    <div class="bt-section-label">Error Responses</div>
    <table class="bt-schema-table">
      <tr><th>Status</th><th>When</th></tr>
      <tr><td><span class="bt-field-type">422</span></td><td class="bt-field-desc">No file uploaded or wrong field name.</td></tr>
      <tr><td><span class="bt-field-type">500</span></td><td class="bt-field-desc">Forensic pipeline failed. <code>detail</code> field contains the error message.</td></tr>
    </table>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

