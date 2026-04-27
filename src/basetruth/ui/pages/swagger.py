"""Swagger page — exposes API documentation links for operators."""
from __future__ import annotations

import streamlit as st

from basetruth.ui.components import _page_title

# ── CSS shared across all endpoint cards ──────────────────────────────────────
_SWAGGER_CSS = """
<style>
/* ── Animations ───────────────────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes linkHover {
    0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
    70% { box-shadow: 0 0 0 6px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}

/* ── Base card ────────────────────────────────────────────────────────────── */
.bt-api-card {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(10, 15, 24, 0.97);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 36px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    animation: fadeIn 0.6s ease-out backwards;
}
/* Stagger animations for consecutive cards */
.bt-api-card:nth-child(2) { animation-delay: 0.1s; }
.bt-api-card:nth-child(3) { animation-delay: 0.2s; }

.bt-api-card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(99,102,241,0.4);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
}

/* ── Card header bar ──────────────────────────────────────────────────────── */
.bt-api-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.05) 100%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    flex-wrap: wrap;
    position: relative;
    overflow: hidden;
}
.bt-api-header::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    transition: left 0.7s;
}
.bt-api-card:hover .bt-api-header::after {
    left: 100%;
}

.bt-method-badge {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 5px 12px;
    border-radius: 8px;
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(34,197,94,0.3);
}

.bt-api-path {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #c7d2fe;
    letter-spacing: 0.01em;
}

.bt-api-summary {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-left: auto;
    font-style: italic;
}

/* ── Section inside card ──────────────────────────────────────────────────── */
.bt-api-section {
    padding: 16px 24px 8px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    transition: background 0.3s ease;
}
.bt-api-section:hover {
    background: rgba(255, 255, 255, 0.01);
}
.bt-api-section:last-child { border-bottom: none; padding-bottom: 20px; }

.bt-section-label {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.bt-section-label::before {
    content: '';
    display: block;
    width: 6px;
    height: 6px;
    background: #6366f1;
    border-radius: 50%;
    box-shadow: 0 0 6px #6366f1;
}

/* ── Description text ─────────────────────────────────────────────────────── */
.bt-api-desc {
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.65;
    margin: 0;
}
.bt-api-desc strong {
    color: #cbd5e1;
}

/* ── Schema table ─────────────────────────────────────────────────────────── */
.bt-schema-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    margin-top: 8px;
}
.bt-schema-table th {
    text-align: left;
    color: #a5b4fc;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(99,102,241,0.18);
    background: rgba(15, 23, 42, 0.3);
}
.bt-schema-table th:first-child { border-top-left-radius: 8px; }
.bt-schema-table th:last-child { border-top-right-radius: 8px; }

.bt-schema-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    vertical-align: top;
    transition: background 0.2s ease;
}
.bt-schema-table tr:hover td {
    background: rgba(99,102,241,0.04);
}
.bt-schema-table tr:last-child td { border-bottom: none; }
.bt-schema-table tr:last-child td:first-child { border-bottom-left-radius: 8px; }
.bt-schema-table tr:last-child td:last-child { border-bottom-right-radius: 8px; }

.bt-field-name {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    color: #79c0ff;
    white-space: nowrap;
    font-weight: 600;
}
.bt-field-type {
    color: #e3b341;
    white-space: nowrap;
}
.bt-field-req  { 
    color: #fca5a5; 
    font-weight: 700; 
    font-size: 0.8rem; 
    background: rgba(248, 113, 113, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid rgba(248, 113, 113, 0.2);
}
.bt-field-opt  { 
    color: #cbd5e1; 
    font-weight: 600; 
    font-size: 0.8rem;
    background: rgba(100, 116, 139, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid rgba(100, 116, 139, 0.2);
}
.bt-field-desc { color: #e2e8f0; line-height: 1.55; font-size: 0.92rem; }
.bt-field-desc code {
    background: rgba(255,255,255,0.06);
    padding: 2px 5px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 0.8rem;
}

/* ── Content-type pill ────────────────────────────────────────────────────── */
.bt-ct-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px;
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.1);
}
.bt-ct-pill::before {
    content: '📄';
    font-size: 0.8rem;
}

/* ── Sub-object note ──────────────────────────────────────────────────────── */
.bt-sub-note {
    font-size: 0.85rem;
    color: #94a3b8;
    font-style: italic;
    margin: 0 0 12px 2px;
}

/* ── Live links bar ───────────────────────────────────────────────────────── */
.bt-links-bar {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 20px 0 16px 0;
    animation: fadeIn 0.4s ease-out;
}
.bt-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 24px;
    border-radius: 12px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    text-decoration: none !important;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
}

.bt-link-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}
.bt-link-primary:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
    border-color: rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
    color: #ffffff !important;
}

.bt-link-secondary {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #f8fafc !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
.bt-link-secondary:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-color: rgba(255, 255, 255, 0.25);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    color: #ffffff !important;
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
  <a class="bt-link-btn bt-link-primary" href="http://localhost:8000/api/docs" target="_blank">
    <span style="font-size:1.1rem; opacity:0.9;">📖</span> Interactive Swagger UI
  </a>
  <a class="bt-link-btn bt-link-secondary" href="http://localhost:8000/api/redoc" target="_blank">
    <span style="font-size:1.1rem; opacity:0.9;">📄</span> ReDoc
  </a>
  <a class="bt-link-btn bt-link-secondary" href="http://localhost:8000/api/openapi.json" target="_blank">
    <span style="font-family:monospace; font-weight:bold; opacity:0.8; font-size:1.1rem;">{ }</span> OpenAPI JSON
  </a>
</div>
<p style="font-size:0.85rem;color:#64748b;margin:0 0 28px 4px;font-weight:500;">
  BaseTruth API Server · Base URL: <code style="color:#6366f1;background:rgba(99,102,241,0.1);padding:3px 8px;border-radius:6px;font-weight:600;">http://localhost:8000</code>
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

