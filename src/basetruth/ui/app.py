from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from basetruth.datasources import DatasourceConfig, DatasourceRegistry
from basetruth.service import BaseTruthService


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

_RISK_COLORS = {
    "high":   ("rgba(220,38,38,0.10)",  "#dc2626", "rgba(220,38,38,0.30)"),
    "medium": ("rgba(217,119,6,0.10)",  "#d97706", "rgba(217,119,6,0.30)"),
    "low":    ("rgba(21,128,61,0.10)",  "#16a34a", "rgba(21,128,61,0.30)"),
    "review": ("rgba(29,78,216,0.10)",  "#2563eb", "rgba(29,78,216,0.30)"),
}

_STATUS_COLORS = {
    "new": "#64748b",
    "triage": "#7c3aed",
    "investigating": "#2563eb",
    "pending_client": "#d97706",
    "closed": "#16a34a",
}

_DISPOSITION_ICONS = {
    "open": "🔓",
    "monitor": "👁",
    "escalate": "⚠️",
    "cleared": "✅",
    "fraud_confirmed": "🚨",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ============================================================
   BASETRUTH UI — Modern Elegant Theme v2
   ============================================================ */

/* ---- Typography ------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ---- CSS Custom Properties (light defaults) --------------- */
:root {
    --bt-surface: #f8fafc;
    --bt-surface-raised: #ffffff;
    --bt-surface-border: #e2e8f0;
    --bt-text-muted: #94a3b8;
    --bt-note-bg: #f1f5f9;
    --bt-note-accent: #6366f1;
    --bt-card-shadow: 0 1px 4px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
    --bt-divider: #e2e8f0;
}

/* ---- Dark mode via Streamlit theme attribute -------------- */
[data-testid="stApp"][data-theme="dark"] {
    --bt-surface: #1e293b;
    --bt-surface-raised: #0f172a;
    --bt-surface-border: #334155;
    --bt-text-muted: #64748b;
    --bt-note-bg: #1e293b;
    --bt-card-shadow: 0 1px 4px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.25);
    --bt-divider: #334155;
}

/* ---- Dark mode via OS preference -------------------------- */
@media (prefers-color-scheme: dark) {
    :root {
        --bt-surface: #1e293b;
        --bt-surface-raised: #0f172a;
        --bt-surface-border: #334155;
        --bt-text-muted: #64748b;
        --bt-note-bg: #1e293b;
        --bt-card-shadow: 0 1px 4px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.25);
        --bt-divider: #334155;
    }
}

/* ---- Layout ----------------------------------------------- */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
}

/* ---- Sidebar root ----------------------------------------- */
/* NOTE: No min/max-width here — those break the native resize handle */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.18) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    padding-left: 0.875rem !important;
    padding-right: 0.875rem !important;
}

/* ---- Sidebar brand ---------------------------------------- */
.bt-brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem 0.5rem 1rem;
    gap: 3px;
}
.bt-brand-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 10px;
    box-shadow: 0 4px 16px rgba(99,102,241,0.45);
}
.bt-brand-name {
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.bt-brand-sub {
    font-size: 9.5px;
    color: #475569;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1px;
}

/* ---- Sidebar nav buttons ---------------------------------- */
[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 2px !important;
    border: none !important;
    width: 100% !important;
    background: transparent !important;
    color: #64748b !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #e2e8f0 !important;
}

/* Active nav item — primary button type */
[data-testid="stSidebar"] [data-testid="baseButton-primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.40) !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.50) !important;
}

/* Sidebar secondary/default button — keep clean, no border */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    border: none !important;
    background: transparent !important;
    color: #64748b !important;
}

/* Sidebar misc text */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption {
    color: #475569 !important;
    font-size: 11px !important;
}
[data-testid="stSidebar"] label {
    color: #475569 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.05) !important;
    border-color: #1e293b !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    opacity: 1 !important;
    margin: 0.75rem 0 !important;
}

/* ---- Metric cards ----------------------------------------- */
[data-testid="stMetric"] {
    background: var(--bt-surface-raised) !important;
    border-radius: 14px !important;
    border: 1px solid var(--bt-surface-border) !important;
    border-top: 3px solid #6366f1 !important;
    padding: 1.1rem 1.4rem !important;
    box-shadow: var(--bt-card-shadow) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    opacity: 0.55 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
}

/* ---- Divider ---------------------------------------------- */
hr {
    margin: 1.25rem 0 !important;
    border-color: var(--bt-divider) !important;
    opacity: 1 !important;
}

/* ---- Form inputs ------------------------------------------ */
.stTextInput > div > input,
.stTextArea > div > textarea {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.14) !important;
}

/* ---- Main action buttons ---------------------------------- */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"],
.stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.30) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ---- Expander -------------------------------------------- */
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ---- Download button ------------------------------------- */
[data-testid="stDownloadButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ---- Dataframe ------------------------------------------- */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--bt-surface-border) !important;
    box-shadow: var(--bt-card-shadow) !important;
}

/* ---- Score card ------------------------------------------- */
.bt-score-card {
    background: var(--bt-surface-raised);
    border: 1px solid var(--bt-surface-border);
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    box-shadow: var(--bt-card-shadow);
}

/* ---- PDF section card ------------------------------------ */
.bt-pdf-card {
    background: var(--bt-surface);
    border: 1px solid var(--bt-surface-border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 0.75rem;
    box-shadow: var(--bt-card-shadow);
}
.bt-pdf-card h4 {
    margin: 0 0 0.25rem;
    font-size: 1rem;
    font-weight: 700;
}
.bt-pdf-card p {
    margin: 0 0 1rem;
    font-size: 0.85rem;
    color: var(--bt-text-muted);
}

/* ---- Headers --------------------------------------------- */
h1 { letter-spacing: -0.03em !important; font-weight: 800 !important; }
h2 { letter-spacing: -0.02em !important; font-weight: 700 !important; }
h3 { letter-spacing: -0.01em !important; font-weight: 600 !important; }

/* ---- Top metrics strip top fix --------------------------- */
[data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMetric"] {
    margin-top: 0 !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_artifact_root() -> Path:
    return Path.cwd() / "artifacts"


def _get_service() -> BaseTruthService:
    artifact_root = Path(str(st.session_state.get("artifact_root", _default_artifact_root())))
    return BaseTruthService(artifact_root)


def _badge(level: str, label: str = "") -> str:
    bg, fg, border = _RISK_COLORS.get(str(level).lower(), _RISK_COLORS["low"])
    text = label or str(level).upper()
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {border};'
        f'padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700;'
        f'letter-spacing:0.04em;white-space:nowrap;">{text}</span>'
    )


def _status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(str(status).lower(), "#64748b")
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600;">'
        f'{status.upper().replace("_", " ")}</span>'
    )


def _score_card(score: Any, risk_level: str) -> str:
    try:
        n = int(score)
    except (TypeError, ValueError):
        n = 0
    _, fg, _ = _RISK_COLORS.get(str(risk_level).lower(), _RISK_COLORS["low"])
    bar_pct = max(0, min(100, n))
    return f"""
<div class="bt-score-card">
  <div style="font-size:52px;font-weight:800;color:{fg};line-height:1.05;letter-spacing:-0.02em;">{n}</div>
  <div style="font-size:11px;color:var(--bt-text-muted);letter-spacing:0.1em;text-transform:uppercase;margin-top:4px;">TRUTH SCORE</div>
  <div style="background:var(--bt-surface-border);border-radius:999px;height:6px;margin:12px 0 8px;">
    <div style="background:{fg};width:{bar_pct}%;height:100%;border-radius:999px;transition:width 0.5s ease;"></div>
  </div>
  <div style="margin-top:6px;">{_badge(risk_level)}</div>
</div>"""


def _signal_icon(sig: Dict[str, Any]) -> str:
    if sig.get("passed") is True:
        return "✅"
    sev = str(sig.get("severity", "info")).lower()
    return {"high": "🚨", "medium": "⚠️", "low": "🔷"}.get(sev, "ℹ️")


def _save_uploaded_files(files: List[Any], temp_dir: Path) -> List[Path]:
    saved = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = temp_dir / file.name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def _display_truth_score(value: Any) -> str:
    return "" if value in {None, ""} else str(value)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

_PAGES: Dict[str, str] = {
    "🏠  Dashboard": "dashboard",
    "🔍  Scan": "scan",
    "📦  Bulk Scan": "bulk",
    "📁  Cases": "cases",
    "📊  Reports": "reports",
    "🔗  Datasources": "datasources",
    "⚙️  Settings": "settings",
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="bt-brand">
              <div class="bt-brand-icon">🛡</div>
              <div class="bt-brand-name">BaseTruth</div>
              <div class="bt-brand-sub">Document Integrity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        current_page = st.session_state.get("page", "dashboard")
        for label, key in _PAGES.items():
            is_active = current_page == key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
                st.session_state["page"] = key
                st.rerun()

        st.divider()
        st.caption("Artifact root")
        st.text_input(
            "artifact_root_sidebar",
            key="artifact_root",
            value=str(st.session_state.get("artifact_root", _default_artifact_root())),
            label_visibility="collapsed",
        )

    return str(st.session_state.get("page", "dashboard"))


# ---------------------------------------------------------------------------
# Shared result rendering
# ---------------------------------------------------------------------------

def _render_report_summary(report: Dict[str, Any]) -> None:
    tamper = report.get("tamper_assessment", {})
    structured = report.get("structured_summary", {})
    key_fields = structured.get("key_fields", {})
    risk_level = str(tamper.get("risk_level", "low"))
    score = tamper.get("truth_score", 0)

    col_score, col_detail = st.columns([1, 2])
    with col_score:
        st.markdown(_score_card(score, risk_level), unsafe_allow_html=True)

    with col_detail:
        doc_type = structured.get("document", {}).get("type", "generic")
        st.markdown(
            f"**Document type:** {doc_type.replace('_', ' ').title()}  \n"
            f"**Source:** {report.get('source', {}).get('name', '')}  \n"
            f"**Verdict:** {tamper.get('verdict', '')}",
        )
        flat_fields = {k: v for k, v in key_fields.items() if not isinstance(v, (dict, list)) and v is not None}
        if flat_fields:
            field_rows = [{"Field": k.replace("_", " ").title(), "Value": str(v)} for k, v in flat_fields.items()]
            try:
                import pandas as pd

                st.dataframe(pd.DataFrame(field_rows), hide_index=True, width="stretch")
            except Exception:
                st.json(flat_fields)

    signals = tamper.get("signals", [])
    if signals:
        with st.expander(f"Forensic signals  ({len(signals)} total)", expanded=False):
            for sig in signals:
                icon = _signal_icon(sig)
                name = str(sig.get("name", "")).replace("_", " ").replace("::", " › ")
                score_part = f"  — score {sig.get('score', 0)}" if sig.get("score", 0) else ""
                st.markdown(f"{icon} **{name}**{score_part}")
                st.caption(sig.get("summary", ""))
                if sig.get("details"):
                    st.json(sig["details"])


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def _page_dashboard(service: BaseTruthService) -> None:
    st.header("Dashboard")

    reports = service.list_reports()
    cases = service.list_cases()
    ver_reports = [r for r in reports if r.get("kind") == "verification"]
    scores = [r.get("truth_score") for r in ver_reports if isinstance(r.get("truth_score"), int)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Cases", len(cases))
    col2.metric("Documents Scanned", len(ver_reports))
    high_risk = sum(1 for r in ver_reports if r.get("risk_level") == "high")
    col3.metric("High Risk", high_risk, delta=None)
    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    col4.metric("Avg Truth Score", f"{avg_score}/100" if avg_score is not None else "—")

    st.divider()

    if not ver_reports:
        st.info("No documents scanned yet. Use **Scan** or **Bulk Scan** to get started.")
        return

    chart_col, recent_col = st.columns([1, 2])

    with chart_col:
        st.subheader("Risk Distribution")
        risk_counts: Dict[str, int] = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
        for r in ver_reports:
            level = str(r.get("risk_level", "low")).lower()
            key = {"high": "High Risk", "medium": "Medium Risk", "low": "Low Risk"}.get(level, "Low Risk")
            risk_counts[key] = risk_counts.get(key, 0) + 1
        try:
            import pandas as pd

            chart_df = pd.DataFrame({"Count": list(risk_counts.values())}, index=list(risk_counts.keys()))
            st.bar_chart(chart_df)
        except ImportError:
            st.json(risk_counts)

    with recent_col:
        st.subheader("Recent Scans")
        recent = sorted(ver_reports, key=lambda r: str(r.get("generated_at", "")), reverse=True)[:10]
        for r in recent:
            name = r.get("source_name", "unknown")
            level = r.get("risk_level", "low")
            score_val = r.get("truth_score", "")
            score_display = f"**{score_val}**" if isinstance(score_val, int) else "—"
            badge_html = _badge(level)
            col_a, col_b, col_c = st.columns([4, 2, 1])
            col_a.write(name)
            col_b.markdown(badge_html, unsafe_allow_html=True)
            col_c.markdown(score_display)

    if cases:
        st.divider()
        st.subheader("Open Cases")
        open_cases = [c for c in cases if c.get("disposition") not in {"cleared", "fraud_confirmed"}][:8]
        if open_cases:
            try:
                import pandas as pd

                rows = [
                    {
                        "Case": c.get("case_key", ""),
                        "Type": c.get("document_type", ""),
                        "Docs": str(c.get("document_count", 0)),
                        "Risk": str(c.get("max_risk_level", "low")).title(),
                        "Status": str(c.get("status", "new")).replace("_", " ").title(),
                        "Assignee": c.get("assignee", "—"),
                    }
                    for c in open_cases
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            except ImportError:
                for c in open_cases:
                    st.write(c.get("case_key", ""))
        else:
            st.info("No open cases.")

    if len(scores) >= 2:
        st.divider()
        st.subheader("Truth Score Trend")
        trend_data = sorted(
            [(str(r.get("source_name", "")), int(r.get("truth_score", 0))) for r in ver_reports if isinstance(r.get("truth_score"), int)],
            key=lambda item: item[0],
        )
        try:
            import pandas as pd

            trend_df = pd.DataFrame({"Truth Score": [v for _, v in trend_data]}, index=[n for n, _ in trend_data])
            st.line_chart(trend_df)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Scan page (single document)
# ---------------------------------------------------------------------------

def _page_scan(service: BaseTruthService) -> None:
    st.header("Scan Document")
    st.markdown("Upload a document or point to an existing file to run a full BaseTruth integrity scan.")

    upload = st.file_uploader(
        "Drop a file here — PDF, JSON (LiteParse or structured), or image",
        type=None,
        accept_multiple_files=False,
        label_visibility="visible",
    )
    path_input = st.text_input("Or enter an existing file path on disk")
    st.caption(
        "Note: scanned identity documents (Aadhaar, PAN card) are image-only PDFs with no text layer. "
        "LiteParse requires ImageMagick for OCR on those files. "
        "If ImageMagick is not installed, BaseTruth uses a text-extraction fallback that extracts "
        "metadata and structure but cannot read the printed text from image-only pages."
    )

    if st.button("Run scan ->", type="primary"):
        report: Dict[str, Any] | None = None
        scan_error: str = ""
        with st.spinner("Running BaseTruth scan..."):
            try:
                if upload is not None:
                    temp_dir = Path(tempfile.mkdtemp(prefix="bt_upload_"))
                    saved_path = _save_uploaded_files([upload], temp_dir)[0]
                    report = service.scan_document(saved_path)
                elif path_input.strip():
                    report = service.scan_document(path_input.strip())
                else:
                    st.warning("Provide an uploaded file or a local file path.")
            except FileNotFoundError as exc:
                scan_error = f"File not found: {exc}"
            except Exception as exc:  # noqa: BLE001
                scan_error = (
                    f"Scan failed: {exc}\n\n"
                    "Common causes:\n"
                    "- ImageMagick is not installed (needed for image-only PDFs such as "
                    "Aadhaar / PAN card scans).  Install from https://imagemagick.org/\n"
                    "- The PDF is password-protected or corrupt.\n"
                    "- An unexpected LiteParse or OCR error occurred."
                )

        if scan_error:
            st.error(scan_error)
            return

        if report:
            artifacts = report.get("artifacts", {})
            summary = report.get("structured_summary", {})
            is_fallback = artifacts.get("parse_fallback") or summary.get("parse_fallback")
            is_image_only = artifacts.get("is_image_only_pdf") or summary.get("is_image_only_pdf")
            ocr_engine = artifacts.get("ocr_engine", "")

            if is_fallback:
                fallback_reason = (
                    artifacts.get("parse_fallback_reason")
                    or summary.get("parse_fallback_reason", "")
                ).split("|")[0].strip()  # First segment only -- drops the verbose stacktrace

                if is_image_only and ocr_engine == "pytesseract":
                    # OCR ran and succeeded.
                    st.info(
                        "**Image-only PDF detected** -- LiteParse required ImageMagick which is "
                        "not installed. BaseTruth used **Tesseract OCR** as a fallback and "
                        "successfully extracted text from the document.  "
                        "Field extraction quality may differ from a full LiteParse scan."
                    )
                elif is_image_only and ocr_engine == "unavailable":
                    # Image-only PDF and no OCR available -- worst case for identity docs.
                    st.warning(
                        "**Image-only PDF -- OCR required for full extraction.**  "
                        "This document (e.g. Aadhaar card, PAN card) contains no embedded "
                        "text layer. PDF metadata forensics ran, but field-level extraction "
                        "was not possible without OCR.\n\n"
                        "**To fix this, choose ONE option:**\n\n"
                        "**Option A (recommended) -- Install ImageMagick:**  \n"
                        "Download from https://imagemagick.org/script/download.php#windows  \n"
                        "Restart the terminal after install, then re-scan.\n\n"
                        "**Option B -- Install Tesseract + Poppler:**  \n"
                        "1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki  \n"
                        "   Add its folder to your system PATH  \n"
                        "2. Poppler: https://github.com/oschwartz10612/poppler-windows/releases  \n"
                        "   Add poppler/bin to your system PATH  \n"
                        "3. In the BaseTruth folder run:  \n"
                        "   `.venv\\Scripts\\pip install pytesseract pdf2image`  \n"
                        "4. Re-scan the document."
                    )
                elif is_image_only:
                    st.warning(
                        "**Image-only PDF** -- no embedded text found after OCR attempt.  "
                        "PDF metadata forensics ran in full.  "
                        f"Reason: {fallback_reason or 'unknown'}."
                    )
                else:
                    # LiteParse failed but text extraction succeeded (text-based PDF).
                    st.warning(
                        "**Partial scan** -- LiteParse could not process this document "
                        f"({fallback_reason or 'reason unknown'}).  "
                        "BaseTruth used text-extraction fallback (PyMuPDF).  "
                        "PDF metadata forensics and structural checks ran in full.  "
                        "To enable full LiteParse scans: install ImageMagick from "
                        "https://imagemagick.org/script/download.php#windows"
                    )
            st.divider()
            st.subheader("Scan Result")
            _render_report_summary(report)
            if artifacts.get("verification_json_path"):
                st.caption(f"Report saved to: {artifacts['verification_json_path']}")

            stem = Path(report.get("source", {}).get("name", "report")).stem
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "⬇ Download JSON report",
                    data=json.dumps(report, indent=2, ensure_ascii=False),
                    file_name=f"{stem}_verification.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_dl2:
                pdf_path_str = artifacts.get("pdf_report_path", "")
                if pdf_path_str and Path(pdf_path_str).exists():
                    st.download_button(
                        "⬇ Download PDF report",
                        data=Path(pdf_path_str).read_bytes(),
                        file_name=f"{stem}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button("PDF report (generating...)", disabled=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Bulk scan page
# ---------------------------------------------------------------------------

def _page_bulk(service: BaseTruthService) -> None:
    st.header("Bulk Scan")
    st.markdown("Scan multiple documents at once and optionally run a cross-month payslip comparison.")

    uploads = st.file_uploader(
        "Upload multiple documents",
        type=None,
        accept_multiple_files=True,
    )
    folder_input = st.text_input("Or scan all supported files from a folder on disk")
    compare_payslips = st.checkbox("Run cross-month payslip comparison after scan", value=True)

    if st.button("Run bulk scan →", type="primary"):
        with st.spinner("Scanning…"):
            paths: List[Path] = []
            if uploads:
                temp_dir = Path(tempfile.mkdtemp(prefix="bt_bulk_"))
                paths.extend(_save_uploaded_files(list(uploads), temp_dir))
            if folder_input.strip():
                paths.extend(service.collect_supported_files(folder_input.strip()))
            if not paths:
                st.warning("Provide uploaded files or a folder path.")
                st.stop()

            _new_reports: List[Dict[str, Any]] = []
            _new_errors: List[str] = []
            prog = st.progress(0)
            for i, p in enumerate(paths):
                try:
                    _new_reports.append(service.scan_document(p))
                except Exception as exc:  # noqa: BLE001
                    _new_errors.append(f"{p.name}: {exc}")
                prog.progress((i + 1) / len(paths))

        # Persist results so they survive reruns triggered by any button inside this page
        st.session_state["bt_bulk_reports"] = _new_reports
        st.session_state["bt_bulk_errors"] = _new_errors
        st.session_state["bt_bulk_compare"] = compare_payslips
        # Clear any previously generated bundle PDF on fresh scan
        st.session_state.pop("bt_bundle_pdf_bytes", None)

    # ── Results section ────────────────────────────────────────────────────
    # All display + action buttons live OUTSIDE the scan-button block so they
    # persist across every Streamlit rerun triggered by any widget click.
    if "bt_bulk_reports" not in st.session_state:
        return

    reports: List[Dict[str, Any]] = st.session_state["bt_bulk_reports"]
    errors: List[str] = st.session_state["bt_bulk_errors"]
    compare_payslips = st.session_state.get("bt_bulk_compare", compare_payslips)

    st.success(f"Scanned {len(reports)} document(s).")
    if errors:
        with st.expander(f"{len(errors)} document(s) had errors -- click to expand"):
            for err in errors:
                st.error(err)

    try:
        import pandas as pd

        summary_rows = []
        for r in reports:
            ss = r.get("structured_summary", {})
            doc = ss.get("document", {})
            kf = ss.get("key_fields", {})
            # Principal: employee name (payslip), account holder (bank), employer (letter)
            principal = (
                kf.get("employee_name")
                or kf.get("account_holder")
                or kf.get("employer_name")
                or kf.get("company_name")
                or ""
            )
            # Key amount: gross earnings (payslip), annual CTC (letter), or opening balance (bank)
            key_amount = (
                kf.get("gross_earnings")
                or kf.get("gross_monthly_salary")
                or kf.get("annual_ctc")
                or kf.get("opening_balance")
                or ""
            )
            if key_amount:
                try:
                    key_amount = f"Rs {int(str(key_amount).replace(',', '')):,}"
                except (ValueError, TypeError):
                    key_amount = str(key_amount)
            summary_rows.append({
                "File": r.get("source", {}).get("name", ""),
                "Type": doc.get("type", ""),
                "Principal": str(principal)[:25] if principal else "—",
                "Key Amount": str(key_amount) if key_amount else "—",
                "Score": _display_truth_score(r.get("tamper_assessment", {}).get("truth_score")),
                "Risk": str(r.get("tamper_assessment", {}).get("risk_level", "")).title(),
                "Confidence": f"{int(doc.get('type_confidence', 0) * 100)}%",
                "Parse Method": ss.get("parse_method", "?"),
                "Fallback": "⚠️ Yes" if ss.get("parse_fallback") else "No",
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    except ImportError:
        st.json([r.get("source", {}).get("name", "") for r in reports])

    # Per-document detailed results with download buttons
    st.subheader("Document Results")
    for r in reports:
        fname = r.get("source", {}).get("name", "unknown")
        stem = Path(fname).stem
        risk = str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
        score = r.get("tamper_assessment", {}).get("truth_score", 100)
        risk_icon = {"high": "🚨", "critical": "🚨", "medium": "⚠️", "review": "🔷"}.get(risk, "✅")
        with st.expander(f"{risk_icon} {fname}  —  Score: {score}/100  |  Risk: {risk.title()}"):
            _render_report_summary(r)
            r_artifacts = r.get("artifacts", {})
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download JSON report",
                    data=json.dumps(r, indent=2, ensure_ascii=False),
                    file_name=f"{stem}_verification.json",
                    mime="application/json",
                    key=f"json_{stem}",
                    use_container_width=True,
                )
            with dl2:
                pdf_path_str = r_artifacts.get("pdf_report_path", "")
                if pdf_path_str and Path(pdf_path_str).exists():
                    st.download_button(
                        "⬇ Download PDF report",
                        data=Path(pdf_path_str).read_bytes(),
                        file_name=f"{stem}_report.pdf",
                        mime="application/pdf",
                        key=f"pdf_{stem}",
                        use_container_width=True,
                    )
                else:
                    st.button("PDF (not available)", disabled=True, key=f"pdf_na_{stem}", use_container_width=True)

    # Classifier diagnostics — one expander per document
    with st.expander("Classifier diagnostics (click to inspect per-file classification)"):
        for r in reports:
            ss = r.get("structured_summary", {})
            doc = ss.get("document", {})
            parser = ss.get("parser", {})
            fname = r.get("source", {}).get("name", "?")
            doc_type = doc.get("type", "generic")
            parse_method = ss.get("parse_method", "?")
            fallback = ss.get("parse_fallback", False)
            fallback_reason = ss.get("parse_fallback_reason", "")
            scores = doc.get("classification_scores", {})
            markers = doc.get("matched_markers", {})
            text_len = parser.get("text_length", 0)
            preview = doc.get("text_preview", "")

            st.markdown(f"**{fname}** → `{doc_type}` ({int(doc.get('type_confidence', 0)*100)}% confidence)")
            col1, col2 = st.columns(2)
            col1.metric("Parse method", parse_method)
            col1.metric("Text extracted (chars)", text_len)
            col2.metric("Classification winner", doc_type)
            if scores:
                col2.write("All type scores:")
                col2.json(scores)
            if markers.get(doc_type):
                st.caption("Matched markers: " + ", ".join(markers[doc_type]))
            if fallback:
                st.warning(f"Parse fallback used: {fallback_reason}")
            if text_len == 0:
                st.error("No text was extracted from this document — classification will be generic. Check that LiteParse (Node.js) is installed or Tesseract OCR is available.")
            elif text_len < 100:
                st.warning(f"Very little text extracted ({text_len} chars). Classification may be unreliable.")
            if preview:
                st.caption("Text preview: " + preview[:200])
            st.divider()

    if compare_payslips:
        comparison = service.compare_payslip_summaries_from_reports(reports)
        if comparison.get("anomalies"):
            st.subheader(f"Payslip anomalies — {len(comparison['anomalies'])} detected")
            for anomaly in comparison["anomalies"]:
                sev = str(anomaly.get("severity", "low"))
                icon = "🚨" if sev == "high" else "⚠️" if sev == "medium" else "🔷"
                with st.expander(
                    f"{icon} {anomaly.get('type', '').replace('_', ' ').title()}  "
                    f"— {anomaly.get('from_period', '')} → {anomaly.get('to_period', '')}"
                ):
                    st.json(anomaly.get("details", {}))
        else:
            st.success("No payslip anomalies detected across this document set.")

    # --- Cross-document income reconciliation ---
    st.divider()
    reconciliation = service.reconcile_income_documents(reports)
    income_anomalies = reconciliation.get("anomalies", [])
    evidence = reconciliation.get("evidence", {})

    st.subheader("Cross-Document Income Reconciliation")

    # Evidence summary table
    evidence_rows = []
    if evidence.get("payslip_avg_monthly_gross"):
        evidence_rows.append({
            "Source": f"Payslips ({evidence.get('payslip_count', 0)} docs)",
            "Monthly Gross": f"₹{evidence['payslip_avg_monthly_gross']:,}",
            "Annual (×12)": f"₹{evidence.get('payslip_annualised_gross', 0):,}",
        })
    if evidence.get("letter_annual_ctc"):
        monthly = evidence.get("letter_gross_monthly")
        evidence_rows.append({
            "Source": evidence.get("letter_source", "Offer letter"),
            "Monthly Gross": f"₹{monthly:,}" if monthly else "—",
            "Annual (×12)": f"₹{evidence['letter_annual_ctc']:,}",
        })
    if evidence.get("form16_annual_gross"):
        evidence_rows.append({
            "Source": evidence.get("form16_source", "Form 16"),
            "Monthly Gross": "—",
            "Annual (×12)": f"₹{evidence['form16_annual_gross']:,}",
        })
    if evidence.get("bank_avg_salary_credit"):
        evidence_rows.append({
            "Source": f"Bank statement ({evidence.get('bank_salary_credit_count', 0)} credits)",
            "Monthly Gross": f"₹{evidence['bank_avg_salary_credit']:,} (net credit)",
            "Annual (×12)": f"₹{evidence['bank_avg_salary_credit'] * 12:,}",
        })

    if evidence_rows:
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, use_container_width=True)
        except ImportError:
            for row in evidence_rows:
                st.write(row)

    if income_anomalies:
        st.error(f"⚠️ {len(income_anomalies)} income inconsistenc{'y' if len(income_anomalies) == 1 else 'ies'} detected — possible income inflation fraud")
        for anomaly in income_anomalies:
            sev = str(anomaly.get("severity", "low"))
            icon = "🚨" if sev == "high" else "⚠️"
            with st.expander(
                f"{icon} {anomaly.get('type', '').replace('_', ' ').title()}  "
                f"— {anomaly.get('from_period', '')} → {anomaly.get('to_period', '')}"
            ):
                details = anomaly.get("details", {})
                explanation = details.get("explanation", "")
                if explanation:
                    st.warning(explanation)
                st.json({k: v for k, v in details.items() if k != "explanation"})
    elif evidence_rows:
        st.success("✅ Income figures are consistent across all documents.")

    # --- Case Bundle PDF Report ---
    st.divider()
    st.subheader("📄 Case Report PDF")
    st.caption(
        "Generate a single plain-English PDF report covering all documents, income "
        "reconciliation, and an overall verdict — suitable for a loan officer review."
    )
    st.markdown('<div class="bt-pdf-card">', unsafe_allow_html=True)
    case_report_title = st.text_input(
        "Report title",
        value="Mortgage Application Review",
        key="bundle_pdf_title",
    )
    _gen_pdf_clicked = st.button(
        "📄 Generate Case Report PDF",
        type="primary",
        use_container_width=True,
        key="gen_bundle_pdf",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if _gen_pdf_clicked:
        try:
            from basetruth.reporting.pdf import render_case_bundle_pdf
            with st.spinner("Generating case report PDF…"):
                _pdf_bytes = render_case_bundle_pdf(
                    reports=reports,
                    reconciliation=reconciliation,
                    case_title=case_report_title or "Mortgage Application Review",
                )
            st.session_state["bt_bundle_pdf_bytes"] = _pdf_bytes
            st.session_state["bt_bundle_pdf_title"] = case_report_title or "case_report"
        except Exception as _e:
            st.error(f"Could not generate bundle PDF: {_e}")
            st.session_state.pop("bt_bundle_pdf_bytes", None)

    # Show download button if PDF bytes are ready in session state
    if "bt_bundle_pdf_bytes" in st.session_state:
        _pdf_title = st.session_state.get("bt_bundle_pdf_title", "case_report")
        _safe_fname = "".join(c if c.isalnum() or c in "-_ " else "_" for c in _pdf_title).strip() + ".pdf"
        st.download_button(
            label="⬇ Download Case Report PDF",
            data=st.session_state["bt_bundle_pdf_bytes"],
            file_name=_safe_fname,
            mime="application/pdf",
            key="dl_bundle_pdf",
            use_container_width=True,
        )
        st.success("✅ Case report PDF is ready — click above to download.")


# ---------------------------------------------------------------------------
# Cases page
# ---------------------------------------------------------------------------

def _page_cases(service: BaseTruthService) -> None:
    st.header("Cases")

    cases = service.list_cases()
    if not cases:
        st.info("No cases yet. Scan documents first and cases are grouped automatically.")
        return

    # --- Filters ---
    fcol1, fcol2, fcol3 = st.columns(3)
    filter_status = fcol1.selectbox(
        "Status", ["All"] + ["new", "triage", "investigating", "pending_client", "closed"]
    )
    filter_risk = fcol2.selectbox("Risk level", ["All", "high", "medium", "low"])
    filter_assign = fcol3.text_input("Assignee contains")

    filtered = cases
    if filter_status != "All":
        filtered = [c for c in filtered if c.get("status") == filter_status]
    if filter_risk != "All":
        filtered = [c for c in filtered if c.get("max_risk_level") == filter_risk]
    if filter_assign.strip():
        filtered = [c for c in filtered if filter_assign.strip().lower() in str(c.get("assignee", "")).lower()]

    st.caption(f"Showing {len(filtered)} of {len(cases)} cases")

    # --- Case list ---
    try:
        import pandas as pd

        case_rows = [
            {
                "Case Key": c.get("case_key", ""),
                "Type": c.get("document_type", ""),
                "Docs": c.get("document_count", 0),
                "Risk": str(c.get("max_risk_level", "low")).title(),
                "Status": str(c.get("status", "new")).replace("_", " ").title(),
                "Priority": str(c.get("priority", "normal")).title(),
                "Assignee": c.get("assignee", "—"),
                "Notes": c.get("note_count", 0),
            }
            for c in filtered
        ]
        st.dataframe(pd.DataFrame(case_rows), hide_index=True, width="stretch")
    except ImportError:
        for c in filtered:
            st.write(c.get("case_key", ""))

    st.divider()

    if not filtered:
        return

    case_key_options = [c.get("case_key", "") for c in filtered]
    selected_key = st.selectbox("Open case for detail / workflow", options=case_key_options)

    try:
        case_detail = service.get_case_detail(selected_key)
    except KeyError:
        st.warning("Case not found.")
        return

    workflow = case_detail["workflow"]
    case_meta = case_detail["case"]

    col_meta, col_form = st.columns([1, 2])

    with col_meta:
        st.markdown("#### Case overview")
        st.markdown(
            f"**Type:** {case_meta.get('document_type', '').replace('_', ' ').title()}  \n"
            f"**Documents:** {case_meta.get('document_count', 0)}  \n"
            f"**Max risk:** {_badge(case_meta.get('max_risk_level', 'low'))  }",
            unsafe_allow_html=True,
        )
        if workflow.get("labels"):
            label_html = "  ".join(_badge("review", lab) for lab in workflow["labels"])
            st.markdown(label_html, unsafe_allow_html=True)

    with col_form:
        st.markdown("#### Investigator workflow")
        statuses = ["new", "triage", "investigating", "pending_client", "closed"]
        dispositions = ["open", "monitor", "escalate", "cleared", "fraud_confirmed"]
        priorities = ["low", "normal", "high", "critical"]

        with st.form("case_workflow_form"):
            wf_cols = st.columns(3)
            current_status = str(workflow.get("status", "new"))
            current_disp = str(workflow.get("disposition", "open"))
            current_prio = str(workflow.get("priority", "normal"))

            status = wf_cols[0].selectbox(
                "Status", statuses,
                index=statuses.index(current_status) if current_status in statuses else 0,
            )
            disposition = wf_cols[1].selectbox(
                "Disposition", dispositions,
                index=dispositions.index(current_disp) if current_disp in dispositions else 0,
            )
            priority = wf_cols[2].selectbox(
                "Priority", priorities,
                index=priorities.index(current_prio) if current_prio in priorities else 1,
            )
            assignee = st.text_input("Assignee / Investigator", value=str(workflow.get("assignee", "")))
            labels_text = st.text_input(
                "Labels (comma separated)", value=", ".join(workflow.get("labels", []))
            )
            note_author = st.text_input("Note author", value="analyst")
            note_text = st.text_area("Add a note", placeholder="Observations, evidence, next steps…")

            if st.form_submit_button("Save workflow update", type="primary"):
                service.update_case(
                    selected_key,
                    status=status,
                    disposition=disposition,
                    priority=priority,
                    assignee=assignee,
                    labels=[item.strip() for item in labels_text.split(",") if item.strip()],
                    note_text=note_text,
                    note_author=note_author,
                )
                st.success("Workflow updated.")
                st.rerun()

    # --- Note history ---
    notes = workflow.get("notes", [])
    if notes:
        st.divider()
        st.markdown(f"#### Notes ({len(notes)})")
        for note in reversed(notes):
            ts = str(note.get("created_at", ""))[:19].replace("T", " ")
            author = note.get("author", "")
            st.markdown(
                f'<div style="background:var(--bt-note-bg);border-left:3px solid var(--bt-note-accent);'
                f'padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:10px;">'
                f'<span style="font-size:11px;color:var(--bt-text-muted);">{ts} · {author}</span><br>'
                f'{note.get("text", "")}</div>',
                unsafe_allow_html=True,
            )

    # --- Linked reports ---
    st.divider()
    st.markdown(f"#### Linked reports ({len(case_detail.get('reports', []))})")
    for report in case_detail.get("reports", []):
        with st.expander(
            f"{_signal_icon({'severity': report.get('tamper_assessment', {}).get('risk_level', 'low'), 'passed': False})} "
            f"{report.get('source', {}).get('name', 'report')}"
        ):
            _render_report_summary(report)


# ---------------------------------------------------------------------------
# Reports page
# ---------------------------------------------------------------------------

def _page_reports(service: BaseTruthService) -> None:
    st.header("Reports")

    reports = service.list_reports()
    if not reports:
        st.info("No reports found. Run a scan first.")
        return

    # --- Filters ---
    fc1, fc2 = st.columns(2)
    filter_kind = fc1.selectbox("Kind", ["All", "verification", "comparison"])
    filter_risk = fc2.selectbox("Risk level", ["All", "high", "medium", "low", "review"])

    filtered = reports
    if filter_kind != "All":
        filtered = [r for r in filtered if r.get("kind") == filter_kind]
    if filter_risk != "All":
        filtered = [r for r in filtered if r.get("risk_level") == filter_risk]

    st.caption(f"Showing {len(filtered)} of {len(reports)} reports")

    try:
        import pandas as pd

        rows = [
            {
                "Source": item.get("source_name", ""),
                "Kind": item.get("kind", ""),
                "Case Key": item.get("case_key", ""),
                "Risk": str(item.get("risk_level", "")).title(),
                "Score": _display_truth_score(item.get("truth_score")),
                "Generated": str(item.get("generated_at", ""))[:19].replace("T", " "),
                "path": item.get("path", ""),
            }
            for item in filtered
        ]
        display_df = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in rows])
        st.dataframe(display_df, hide_index=True, width="stretch")
    except ImportError:
        for r in filtered:
            st.write(r.get("source_name", ""))
        rows = [{"path": r.get("path", ""), "Source": r.get("source_name", "")} for r in filtered]

    paths = [r.get("path", "") for r in filtered if r.get("path")]
    if paths:
        selected_path = st.selectbox(
            "Open report", options=paths,
            format_func=lambda p: Path(p).name if p else p,
        )
        if selected_path and Path(selected_path).exists():
            payload = json.loads(Path(selected_path).read_text(encoding="utf-8"))
            col_dl, _ = st.columns([1, 4])
            col_dl.download_button(
                "⬇  Download",
                data=json.dumps(payload, indent=2, ensure_ascii=False),
                file_name=Path(selected_path).name,
                mime="application/json",
            )
            with st.expander("Report JSON", expanded=True):
                st.json(payload)


# ---------------------------------------------------------------------------
# Connector auth settings helpers
# ---------------------------------------------------------------------------

def _connector_settings_fields(kind: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if kind == "s3":
        col1, col2 = st.columns(2)
        settings["bucket"] = col1.text_input("S3 bucket *", value=str(existing.get("bucket", "")))
        settings["prefix"] = col2.text_input("Prefix", value=str(existing.get("prefix", "")))
        col3, col4 = st.columns(2)
        settings["region_name"] = col3.text_input("AWS region", value=str(existing.get("region_name", "")))
        settings["profile_name"] = col4.text_input("AWS profile", value=str(existing.get("profile_name", "")))
        st.caption("Auth: uses the named AWS profile, or the standard environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).")
    elif kind == "google_drive":
        settings["folder_id"] = st.text_input("Drive folder ID *", value=str(existing.get("folder_id", "")))
        settings["service_account_file"] = st.text_input(
            "Service account JSON path (leave empty for ADC)",
            value=str(existing.get("service_account_file", "")),
        )
        st.caption("Auth: service-account JSON for server environments, or application-default credentials for local use.")
    elif kind == "sharepoint":
        col1, col2 = st.columns(2)
        settings["site_id"] = col1.text_input("SharePoint site ID *", value=str(existing.get("site_id", "")))
        settings["drive_id"] = col2.text_input("Drive ID *", value=str(existing.get("drive_id", "")))
        settings["folder_path"] = st.text_input("Folder path", value=str(existing.get("folder_path", "")))
        settings["token_env_var"] = st.text_input(
            "Environment variable holding the Microsoft Graph bearer token",
            value=str(existing.get("token_env_var", "BASETRUTH_SHAREPOINT_TOKEN")),
        )
        st.caption("Auth: set the named env var to a valid Microsoft Graph bearer token before syncing.")
    return settings


# ---------------------------------------------------------------------------
# Datasources page
# ---------------------------------------------------------------------------

def _page_datasources(service: BaseTruthService) -> None:
    st.header("Datasources")
    registry = DatasourceRegistry(service.artifact_root)
    sources = registry.list_sources()

    if sources:
        st.subheader("Registered datasources")
        try:
            import pandas as pd

            st.dataframe(
                pd.DataFrame([s.to_dict() for s in sources])
                .drop(columns=["settings"], errors="ignore"),
                hide_index=True,
                width="stretch",
            )
        except ImportError:
            st.json([s.to_dict() for s in sources])

        selected_name = st.selectbox("Select datasource", options=[s.name for s in sources])
        selected_cfg = registry.get_source(selected_name)

        with st.expander("Connector auth and config", expanded=False):
            st.json({"path": selected_cfg.path, "settings": selected_cfg.settings or {}})

        col_sync, col_scan, _ = st.columns([1, 1, 4])
        if col_sync.button("Sync"):
            with st.spinner("Syncing…"):
                result = registry.sync_source(selected_name)
            if result.get("status") == "success":
                st.success(result.get("message", "Sync complete."))
            else:
                st.warning(str(result.get("message", result)))
            st.json(result)

        if col_scan.button("Sync + Scan"):
            with st.spinner("Syncing then scanning…"):
                result = registry.sync_source(selected_name)
                scan_reports: List[Dict[str, Any]] = []
                if result.get("status") == "success":
                    for fpath in result.get("copied_files", []):
                        scan_reports.append(service.scan_document(fpath))
            st.success(f"Synced {result.get('copied_count', 0)} file(s), scanned {len(scan_reports)} document(s).")

        st.divider()

    st.subheader("Register a new datasource")
    with st.form("datasource_form"):
        col_name, col_kind = st.columns(2)
        name = col_name.text_input("Datasource name *")
        kind = col_kind.selectbox("Type", options=["folder", "manifest", "s3", "google_drive", "sharepoint"])
        path = st.text_input("Source path (folder / manifest / leave empty to derive from fields below)")
        col_rec, col_ext = st.columns(2)
        recursive = col_rec.checkbox("Recursive", value=True)
        extensions = col_ext.text_input("Extensions", value=".pdf,.json,.png,.jpg,.jpeg")
        description = st.text_area("Description", height=60)

        settings: Dict[str, Any] = {}
        if kind in {"s3", "google_drive", "sharepoint"}:
            st.markdown("**Connector settings**")
            existing_cfg = registry.get_source(name) if name and name in [s.name for s in sources] else None
            settings = _connector_settings_fields(kind, dict(existing_cfg.settings or {}) if existing_cfg else {})

        if st.form_submit_button("Save datasource", type="primary"):
            if not name.strip():
                st.error("Datasource name is required.")
            else:
                resolved_path = registry.build_path_from_settings(kind, settings, path)
                registry.upsert_source(
                    DatasourceConfig(
                        name=name.strip(),
                        kind=kind,
                        path=resolved_path,
                        recursive=recursive,
                        extensions=[e.strip() for e in extensions.split(",") if e.strip()],
                        description=description.strip(),
                        settings=settings or None,
                    )
                )
                st.success(f"Datasource '{name}' saved.")
                st.rerun()

    st.markdown(
        """
        > **Operating model** — BaseTruth syncs documents from client sources into a read-only snapshot
        > under its managed workspace. This preserves chain-of-custody, leaves the client system untouched,
        > and produces deterministic evidence trails for every scanned document.
        """
    )


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

def _page_settings() -> None:
    st.header("Settings")

    st.subheader("Artifact root")
    st.markdown(
        "All reports, structured summaries, and case records are stored under this directory. "
        "Change it in the sidebar to point BaseTruth at a different workspace."
    )
    artifact_root = str(st.session_state.get("artifact_root", _default_artifact_root()))
    st.code(artifact_root, language=None)
    if Path(artifact_root).exists():
        items = list(Path(artifact_root).rglob("*"))
        st.metric("Items in artifact root", len(items))

    st.divider()
    st.subheader("Product information")
    st.markdown(
        """
        | Property | Value |
        |---|---|
        | Product | **BaseTruth** |
        | Version | 0.1.0 |
        | Python | `basetruth` package |
        | UI runtime | Streamlit |
        | REST API | `uvicorn basetruth.api:app` |
        """
    )

    st.divider()
    st.subheader("Quick start commands")
    st.code(
        "# CLI scan\npython -m basetruth.cli scan --input /path/to/doc.pdf\n\n"
        "# Compare payslips across months\npython -m basetruth.cli compare-payslips --input-dir /path/to/payslips\n\n"
        "# Start UI\nstreamlit run src/basetruth/ui/app.py\n\n"
        "# Start REST API\nuvicorn basetruth.api:app --host 0.0.0.0 --port 8502",
        language="bash",
    )


# ---------------------------------------------------------------------------
# Index metrics (sidebar summary)
# ---------------------------------------------------------------------------

def _render_index_metrics() -> None:
    service = _get_service()
    reports = service.list_reports()
    cases = service.list_cases()
    ver_reports = [r for r in reports if r.get("kind") == "verification"]
    cols = st.columns(4)
    cols[0].metric("Cases", len(cases))
    cols[1].metric("Scanned", len(ver_reports))
    cols[2].metric(
        "High Risk",
        sum(1 for r in ver_reports if r.get("risk_level") == "high"),
    )
    cols[3].metric(
        "Comparisons",
        sum(1 for r in reports if r.get("kind") == "comparison"),
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="BaseTruth",
        page_icon="🛡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    if "artifact_root" not in st.session_state:
        st.session_state["artifact_root"] = str(_default_artifact_root())
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

    page = _sidebar()
    service = _get_service()

    # Top-level index metrics strip (shown on every page except settings).
    if page != "settings":
        _render_index_metrics()
        st.divider()

    if page == "dashboard":
        _page_dashboard(service)
    elif page == "scan":
        _page_scan(service)
    elif page == "bulk":
        _page_bulk(service)
    elif page == "cases":
        _page_cases(service)
    elif page == "reports":
        _page_reports(service)
    elif page == "datasources":
        _page_datasources(service)
    elif page == "settings":
        _page_settings()
    else:
        st.warning(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
