"""Forensic Scan page — run tamper analysis on one uploaded document.

This page mirrors the Scan Document user experience but focuses on forensic
integrity checks. It does not write to PostgreSQL or MinIO.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from basetruth.logger import get_logger
from basetruth.service import BaseTruthService
from basetruth.ui.components import _page_title
from basetruth.ui.pages.forensics_utils import ForensicAnalyzer
from basetruth.ui.pages.scan import (
    _RESULT_CSS,
    _doc_type_icon,
    _syntax_highlight_json,
)

log = get_logger(__name__)


_VERDICT_BADGE: Dict[str, str] = {
    "ORIGINAL": "🟢 ORIGINAL",
    "UNCERTAIN": "🟡 UNCERTAIN",
    "LIKELY TAMPERED": "🟠 LIKELY TAMPERED",
    "TAMPERED": "🔴 TAMPERED",
    "UNAVAILABLE": "⚪ UNAVAILABLE",
}

# Map each verdict to the SCAN result-header gradient accent, so the header bar
# changes colour to match the risk level — a quick visual cue at a glance.
_VERDICT_BORDER: Dict[str, str] = {
    "ORIGINAL": "rgba(34,197,94,0.28)",
    "UNCERTAIN": "rgba(234,179,8,0.28)",
    "LIKELY TAMPERED": "rgba(249,115,22,0.28)",
    "TAMPERED": "rgba(239,68,68,0.28)",
    "UNAVAILABLE": "rgba(148,163,184,0.18)",
}


def _verdict_color(verdict: str) -> str:
    """Return a stable accent colour for each forensic verdict level."""
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "ORIGINAL":
        return "#22c55e"
    if verdict_upper == "UNCERTAIN":
        return "#eab308"
    if verdict_upper == "LIKELY TAMPERED":
        return "#f97316"
    if verdict_upper == "TAMPERED":
        return "#ef4444"
    return "#94a3b8"


def _page_forensic_scan(_service: BaseTruthService = None) -> None:  # type: ignore[assignment]
    """Render the Forensic Scan page."""
    st.markdown(_RESULT_CSS, unsafe_allow_html=True)
    st.markdown(_page_title("🧪", "Forensic Scan"), unsafe_allow_html=True)

    with st.expander("ℹ️ How this screen works", expanded=False):
        st.markdown(
            """
<div style="margin:10px 0">
<div class="bt-step-row"><div class="bt-step-num">1</div>
Upload one image or PDF document.</div>
<div class="bt-step-row"><div class="bt-step-num">2</div>
BaseTruth auto-detects whether the file is scanned/image-based or structured/digital.</div>
<div class="bt-step-row"><div class="bt-step-num">3</div>
The correct forensic engine runs: image forensics for scanned files, PDF forensics for structured PDFs.</div>
<div class="bt-step-row"><div class="bt-step-num">4</div>
You get tamper verdict + score + evidence in JSON format.</div>
</div>

> No data is saved from this page. Use this as an instant forensic checker.
""",
            unsafe_allow_html=True,
        )

    upload = st.file_uploader(
        "Drop a document here to run forensic analysis",
        type=None,
        accept_multiple_files=False,
        key="forensic_scan_upload",
        help="Supported: PDF, JPG, PNG, TIFF, BMP, WebP",
    )

    if upload is None:
        st.info("Upload a file to start forensic analysis.")
        return

    file_bytes = upload.read()
    filename = upload.name or "document"
    cache_key = f"forensic_result_{filename}_{len(file_bytes)}"

    if st.session_state.get(cache_key) is None:
        with st.spinner("🧪 Running forensic analysis..."):
            result = ForensicAnalyzer.analyze_document(file_bytes, filename)
        st.session_state[cache_key] = result
    else:
        result = st.session_state[cache_key]

    if result.get("error"):
        st.error(f"Forensic analysis failed: {result['error']}")
        return

    verdict = str(result.get("forensic_verdict", "UNAVAILABLE") or "UNAVAILABLE").upper()
    score = float(result.get("forgery_score_0_100", 0.0) or 0.0)
    document_type = str(result.get("document_type", "document") or "document")
    is_image_based = bool(result.get("is_image_based", False))
    scan_method = "Scanned / Image-based" if is_image_based else "Digital / Structured"
    scan_method_icon = "🖼" if is_image_based else "📑"
    # Show which scorer produced the fraud score so reviewers know the source
    scoring_method = str(result.get("scoring_method", "heuristic") or "heuristic")
    scoring_badge = "🤖 ML Model (XGBoost)" if scoring_method == "ML" else "📐 Heuristic (rule-based)"
    verdict_label = _VERDICT_BADGE.get(verdict, verdict)
    v_color = _verdict_color(verdict)
    v_border = _VERDICT_BORDER.get(verdict, "rgba(148,163,184,0.18)")
    icon = _doc_type_icon(document_type)

    # ── Classification banner (mirrors Scan Document) ──────────────────────
    st.markdown(
        f"""
        <div class="bt-classify-banner">
          <div class="bt-classify-icon">{icon}</div>
          <div class="bt-classify-detail">
            <div class="bt-classify-type" style="color:{v_color};">{verdict_label}</div>
            <div class="bt-classify-meta">
                {document_type.replace("_", " ").title()} &nbsp;·&nbsp; {scan_method}
                &nbsp;·&nbsp; Score: <strong>{score:.1f}/100</strong>
                &nbsp;·&nbsp; {scoring_badge}
                &nbsp;·&nbsp; File: <strong>{filename}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Single dark JSON panel — all forensic info in one place ────────────
    payload: Dict[str, Any] = {
        "filename": filename,
        "document_type": document_type,
        "is_image_based": is_image_based,
        "forensic_verdict": verdict,
        "forgery_score_0_100": score,
        "overall_explanation": result.get("overall_explanation", "") or "",
        "honest_review": result.get("honest_review", "") or "",
        "evidence": result.get("evidence", []) or [],
        # Include the full per-layer breakdown so API consumers can see exactly
        # which layers fired and the raw metrics behind each verdict.
        "layers": result.get("layers", {}) or {},
    }

    # ── Honest Review card — plain-English LLM verdict for the end user ────
    # This section is deliberately placed BEFORE the JSON panel so the reviewer
    # sees the human-readable conclusion first, before diving into the raw data.
    honest_review_text = payload["honest_review"]
    if honest_review_text:
        # Pick a border colour that matches the current verdict severity so the
        # card visually reinforces the risk level at a glance.
        review_border = v_color
        st.markdown(
            f"""
            <div style="
                border-radius: 12px;
                border-left: 4px solid {review_border};
                background: rgba(15, 23, 42, 0.85);
                padding: 16px 22px;
                margin-bottom: 18px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.18);
            ">
                <div style="
                    font-size: 0.72rem;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: {review_border};
                    margin-bottom: 8px;
                ">🔎 Honest Review</div>
                <div style="
                    font-size: 0.92rem;
                    line-height: 1.65;
                    color: #e2e8f0;
                ">{honest_review_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    highlighted = _syntax_highlight_json(payload)
    st.markdown(
        f"""
        <div class="bt-scan-result-wrap" style="border-color:{v_border};">
            <div class="bt-scan-result-header">
                <span class="bt-doc-type-badge">{document_type.replace("_", " ")}</span>
                <span class="bt-scan-badge" style="color:{v_color}; font-weight:700;">{verdict_label}</span>
                <span class="bt-scan-badge">{scan_method_icon} {scan_method}</span>
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
            "⬇ Download forensic result as JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"{Path(filename).stem}_forensic_result.json",
            mime="application/json",
            use_container_width=True,
            key="forensic_scan_dl_btn",
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True, key="forensic_scan_reset_btn",
                     help="Clear result and scan another document"):
            for k in [k for k in st.session_state if k.startswith("forensic_scan_") or k.startswith("forensic_result_")]:
                del st.session_state[k]
            st.rerun()
