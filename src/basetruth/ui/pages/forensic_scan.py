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
    "ORIGINAL-DERIVED": "🔵 ORIGINAL-DERIVED",
    "UNCERTAIN": "🟡 UNCERTAIN",
    "LIKELY TAMPERED": "🟠 LIKELY TAMPERED",
    "TAMPERED": "🔴 TAMPERED",
    "TAMPERED-DERIVED": "🟣 TAMPERED-DERIVED",
    "UNAVAILABLE": "⚪ UNAVAILABLE",
}

# Map each verdict to the SCAN result-header gradient accent, so the header bar
# changes colour to match the risk level — a quick visual cue at a glance.
_VERDICT_BORDER: Dict[str, str] = {
    "ORIGINAL": "rgba(34,197,94,0.28)",
    "ORIGINAL-DERIVED": "rgba(59,130,246,0.28)",
    "UNCERTAIN": "rgba(234,179,8,0.28)",
    "LIKELY TAMPERED": "rgba(249,115,22,0.28)",
    "TAMPERED": "rgba(239,68,68,0.28)",
    "TAMPERED-DERIVED": "rgba(168,85,247,0.28)",
    "UNAVAILABLE": "rgba(148,163,184,0.18)",
}


def _verdict_color(verdict: str) -> str:
    """Return a stable accent colour for each forensic verdict level."""
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "ORIGINAL":
        return "#22c55e"   # green — confirmed genuine
    if verdict_upper == "ORIGINAL-DERIVED":
        return "#3b82f6"   # blue — genuine save-as copy
    if verdict_upper == "UNCERTAIN":
        return "#eab308"   # yellow — heuristic fallback, inconclusive
    if verdict_upper == "LIKELY TAMPERED":
        return "#f97316"   # orange — heuristic fallback, suspicious
    if verdict_upper == "TAMPERED":
        return "#ef4444"   # red — confirmed forgery
    if verdict_upper == "TAMPERED-DERIVED":
        return "#a855f7"   # purple — laundered forgery (save-as of tampered)
    return "#94a3b8"


def _render_visual_clues(visual_clues: Dict, verdict_color: str) -> None:
    """Render Gemma4's visual detective findings as a structured card.

    Gemma4 inspects the document image like a senior fraud investigator and
    returns plain-English observations: font mismatches, cut-and-paste halos,
    colour patches, misaligned fields, irregular stamps, etc.  This is more
    actionable than raw ELA heat maps because each finding is labelled with
    WHERE it is, WHAT was observed, and WHY it is suspicious.
    """
    if not visual_clues:
        return

    # Ollama was offline or the image conversion failed — show a soft info message.
    if visual_clues.get("_unavailable"):
        st.info(
            "ℹ️ Visual intelligence analysis requires Ollama to be running. "
            "Start Ollama to enable Gemma4 visual clue detection.",
        )
        return

    findings  = visual_clues.get("findings") or []
    overall   = (visual_clues.get("overall_assessment") or "").strip()
    no_clues  = visual_clues.get("no_clues_found", False)

    # Colour and icon per suspicion level.
    _LEVEL_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#eab308"}
    _LEVEL_ICON  = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}

    if no_clues or not findings:
        # Clean bill of health — show a green confirmation.
        body_html = (
            "<div style='color:#22c55e;font-size:0.9rem;padding:4px 0;'>"
            "✅ No visual fraud clues detected in this document."
            "</div>"
        )
    else:
        # One card per finding — coloured left-border matches suspicion level.
        rows = []
        for finding in findings:
            area   = finding.get("area", "")
            clue   = finding.get("clue", "")
            level  = str(finding.get("suspicion_level", "LOW")).upper()
            reason = finding.get("reason", "")
            lc = _LEVEL_COLOR.get(level, "#94a3b8")
            li = _LEVEL_ICON.get(level, "⚪")
            rows.append(
                f"<div style='border:1px solid {lc}33;border-left:3px solid {lc};"
                f"border-radius:8px;padding:10px 14px;margin-bottom:10px;"
                f"background:rgba(15,23,42,0.55);'>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>"
                f"<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.07em;"
                f"text-transform:uppercase;color:{lc};'>{li}&nbsp;{level}</span>"
                f"<span style='color:#475569;font-size:0.72rem;'>&middot;</span>"
                f"<span style='font-size:0.82rem;color:#cbd5e1;font-weight:600;'>{area}</span>"
                f"</div>"
                f"<div style='font-size:0.88rem;color:#e2e8f0;margin-bottom:5px;'>{clue}</div>"
                f"<div style='font-size:0.78rem;color:#94a3b8;font-style:italic;'>{reason}</div>"
                f"</div>"
            )
        body_html = "\n".join(rows)

    # Detective summary at the bottom — plain-English paragraph from Gemma4.
    overall_html = (
        f"<div style='font-size:0.88rem;color:#e2e8f0;line-height:1.68;"
        f"border-top:1px solid rgba(255,255,255,0.07);margin-top:14px;padding-top:12px;'>"
        f"<strong style='color:#94a3b8;font-size:0.70rem;text-transform:uppercase;"
        f"letter-spacing:0.07em;'>Detective Summary</strong><br/>{overall}</div>"
        if overall else ""
    )

    st.markdown(
        f"""
        <div style="
            border-radius: 12px;
            border-left: 4px solid {verdict_color};
            background: rgba(15, 23, 42, 0.85);
            padding: 16px 22px 20px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        ">
            <div style="
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {verdict_color};
                margin-bottom: 14px;
            ">🔍 Visual Intelligence — Gemma4 Detective Report</div>
            {body_html}
            {overall_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feature_contributions(contributions: Dict[str, float], verdict_color: str) -> None:
    """Render a horizontal SHAP feature-contribution bar chart in Streamlit.

    Shows the top-10 features by absolute SHAP value.  Red bars mean the
    feature pushed the model toward TAMPERED; green bars mean it pushed toward
    GENUINE.  Bar width is proportional to the absolute value so all bars fit
    in the same scale.

    This uses pure HTML/CSS — no extra plotting library is required.
    """
    if not contributions:
        return

    # Sort by absolute SHAP value descending and keep top 10.
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    max_abs = max(abs(v) for _, v in sorted_items) or 1.0  # avoid divide-by-zero

    # Build one HTML row per feature: label · bar · value.
    rows_html = []
    for name, val in sorted_items:
        pct = min(100.0, abs(val) / max_abs * 100)
        # Red = pushes toward TAMPERED (positive SHAP); green = toward GENUINE (negative).
        bar_color = "#ef4444" if val > 0 else "#22c55e"
        sign = "+" if val > 0 else ""
        label = name.replace("_", " ")
        rows_html.append(
            f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <div style="width:170px;font-size:0.77rem;color:#cbd5e1;text-align:right;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
       title="{name}">{label}</div>
  <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:3px;height:14px;overflow:hidden;">
    <div style="width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:3px;
                transition:width 0.4s;"></div>
  </div>
  <div style="width:54px;font-size:0.75rem;color:{bar_color};font-weight:600;
              text-align:left;">{sign}{val:.3f}</div>
</div>"""
        )

    rows_joined = "\n".join(rows_html)
    st.markdown(
        f"""
        <div style="
            border-radius: 12px;
            border-left: 4px solid {verdict_color};
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
                color: {verdict_color};
                margin-bottom: 10px;
            ">🔬 Feature Contributions (SHAP — top 10 by influence)</div>
            <div style="font-size:0.70rem;color:#64748b;margin-bottom:12px;">
                Red bars push toward <strong style="color:#ef4444">TAMPERED</strong>;
                green bars push toward <strong style="color:#22c55e">GENUINE</strong>.
                Values are log-odds units from XGBoost tree SHAP.
            </div>
            {rows_joined}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    # Show which scorer produced the fraud score so reviewers know the source.
    # scoring_method can be "ML" (image engine) or "ML (XGBoost)" (PDF engine);
    # check with startswith so both values produce the ML badge.
    scoring_method = str(result.get("scoring_method", "heuristic") or "heuristic")
    scoring_badge = "🤖 ML Model (XGBoost)" if scoring_method.upper().startswith("ML") else "📐 Heuristic (rule-based)"
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
    # ── Feature Contributions chart — only when ML scoring was used ────────
    # Shows which XGBoost features drove the score via tree SHAP values.
    # Silently skipped when the heuristic ran or SHAP was unavailable.
    feature_contributions = result.get("feature_contributions") or {}
    if feature_contributions and scoring_method.upper().startswith("ML"):
        _render_feature_contributions(feature_contributions, v_color)

    # ── Gemma4 Visual Detective — plain-English fraud clue findings ────────
    # Gemma4 scanned the document image looking for visual anomalies: font
    # mismatches, cut-and-paste halos, colour patches, misaligned fields, etc.
    # This is more actionable than raw ELA maps because findings are in plain
    # language that any reviewer can act on without forensic training.
    visual_clues = result.get("visual_clues") or {}
    _render_visual_clues(visual_clues, v_color)

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
