from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from basetruth.datasources import DatasourceConfig, DatasourceRegistry
from basetruth.service import BaseTruthService

# DB layer — imported lazily so the app still starts even without a DB.
try:
    from basetruth.db import db_available, init_db
    from basetruth.store import (
        db_dashboard_stats,
        db_stats,
        db_table_counts,
        db_table_rows,
        get_all_entities_with_scans,
        get_entity_latest_pdf,
        get_entity_scans,
        get_scan_pdf,
        list_cases_from_db,
        list_recent_scans,
        minio_available,
        minio_bucket_stats,
        minio_get_object,
        minio_list_objects,
        minio_truncate_bucket,
        minio_upload,
        reset_db,
        search_entities,
        update_case_in_db,
        update_entity,
    )
    _DB_IMPORTS_OK = True
except Exception:  # noqa: BLE001
    _DB_IMPORTS_OK = False
    def db_available() -> bool: return False  # type: ignore[misc]
    def init_db() -> bool: return False  # type: ignore[misc]

try:
    from basetruth.logger import log_path as _log_path
    _LOGGER_OK = True
except Exception:
    _LOGGER_OK = False
    def _log_path(): return None  # type: ignore[misc,return-value]


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
    background: var(--secondary-background-color, var(--bt-surface-raised)) !important;
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
    /* var(--text-color) is Streamlit-injected and auto-adapts to dark/light mode */
    color: var(--text-color, #64748b) !important;
    opacity: 0.65 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    /* var(--text-color) is Streamlit-injected and auto-adapts to dark/light mode */
    color: var(--text-color, #0f172a) !important;
}

/* ---- Metric text — dark mode hard overrides (fallback) ---- */
/* Used when Streamlit's --text-color is not injected (rare) */
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetricValue"],
[data-theme="dark"] [data-testid="stMetricValue"],
html[data-theme="dark"] [data-testid="stMetricValue"],
body[data-theme="dark"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetricLabel"],
[data-theme="dark"] [data-testid="stMetricLabel"],
html[data-theme="dark"] [data-testid="stMetricLabel"],
body[data-theme="dark"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    opacity: 1 !important;
}
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetric"],
[data-theme="dark"] [data-testid="stMetric"],
html[data-theme="dark"] [data-testid="stMetric"] {
    background: #0f172a !important;
    border-color: #334155 !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; opacity: 1 !important; }
    [data-testid="stMetric"] { background: #0f172a !important; border-color: #334155 !important; }
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
h1 {
    letter-spacing: -0.03em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #06b6d4 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
h2 { letter-spacing: -0.02em !important; font-weight: 700 !important; }
h3 { letter-spacing: -0.01em !important; font-weight: 600 !important; }

/* ---- Top metrics strip top fix --------------------------- */
[data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMetric"] {
    margin-top: 0 !important;
}

/* ---- Search card ----------------------------------------- */
.bt-search-card {
    background: var(--secondary-background-color, #ffffff);
    border: 1px solid var(--bt-surface-border, #e2e8f0);
    border-radius: 14px;
    padding: 1.1rem 1.25rem 0.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.bt-search-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
    display: block;
}

/* ---- Identity / entity record card ----------------------- */
.bt-entity-card {
    border-left: 4px solid #6366f1;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0 1rem;
    transition: box-shadow 0.2s ease;
}
.bt-entity-card:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.15) !important;
}
.bt-entity-field-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 3px;
}
.bt-entity-name {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: var(--text-color, #0f172a);
}
.bt-entity-field-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-color, #1e293b);
}

/* ---- Dark mode: entity card text override ----------------- */
/* Without this, field values are invisible in dark mode      */
/* (dark-navy fallback on dark-navy background).              */
[data-testid="stApp"][data-theme="dark"] .bt-entity-name,
[data-testid="stApp"][data-theme="dark"] .bt-entity-field-value {
    color: #f1f5f9 !important;
}
@media (prefers-color-scheme: dark) {
    .bt-entity-name     { color: #f1f5f9 !important; }
    .bt-entity-field-value { color: #f1f5f9 !important; }
}

/* ---- Auto-generated PDF banner --------------------------- */
.bt-pdf-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ---- Select box ------------------------------------------ */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}

/* ---- Alert/info boxes ------------------------------------ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ---- Upload area ----------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px !important;
    border-style: dashed !important;
}

/* ---- Tabs ----------------------------------------------- */
[data-testid="stTabs"] [data-testid="stTab"] {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: #6366f1 !important;
}

/* ============================================================
   ADDITIONAL POLISH — dark mode, alignment, modern touches
   ============================================================ */

/* ---- Guarantee --text-color is always defined ------------- */
/* Streamlit injects this, but we define fallbacks so inline   */
/* var(--text-color) works even in edge-case versions.         */
:root { --text-color: #1e293b; }
@media (prefers-color-scheme: dark) { :root { --text-color: #f1f5f9; } }
[data-testid="stApp"][data-theme="dark"] { --text-color: #f1f5f9; }

/* ---- Records search bar: align button to input bottom ----- */
/* Bottom-aligns all columns so inputs and button sit level.   */
.bt-search-card [data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
    gap: 0.75rem !important;
}
/* Remove any extra top-margin on the button widget wrapper    */
.bt-search-card [data-testid="column"]:last-child > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ---- Dark mode: entity card background override ----------- */
[data-testid="stApp"][data-theme="dark"] .bt-entity-card {
    background: #1e293b !important;
    border-color: rgba(99,102,241,0.30) !important;
}
@media (prefers-color-scheme: dark) {
    .bt-entity-card {
        background: #1e293b !important;
        border-color: rgba(99,102,241,0.30) !important;
    }
}

/* ---- Custom scrollbar (webkit) ----------------------------- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.28);
    border-radius: 999px;
    transition: background 0.2s ease;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

/* ---- Alert / info / warning / error / success ------------- */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-size: 0.875rem !important;
    line-height: 1.55 !important;
}

/* ---- Progress bar ------------------------------------------ */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #6366f1, #8b5cf6) !important;
    border-radius: 999px !important;
    transition: width 0.3s ease !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 999px !important;
    overflow: hidden !important;
}

/* ---- File uploader ----------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    border-color: rgba(99,102,241,0.35) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1 !important;
    background: rgba(99,102,241,0.04) !important;
}

/* ---- Expander ---------------------------------------------- */
[data-testid="stExpander"] summary svg { color: #6366f1 !important; }
[data-testid="stExpander"] summary {
    transition: background 0.15s ease !important;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(99,102,241,0.06) !important;
}

/* ---- Checkbox/toggle --------------------------------------- */
[data-testid="stCheckbox"] label span[data-testid="stCheckboxLabel"] {
    font-size: 0.875rem !important;
}

/* ---- Caption and help text --------------------------------- */
[data-testid="stCaptionContainer"] {
    font-size: 11.5px !important;
    line-height: 1.55 !important;
    opacity: 0.75 !important;
}

/* ---- Selectbox dropdown ------------------------------------ */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ---- Spinner ----------------------------------------------- */
[data-testid="stSpinner"] > div {
    border-top-color: #6366f1 !important;
}

/* ---- Download button accent -------------------------------- */
[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#4f46e5 0%,#6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.30) !important;
}

/* ---- Sub-headers ------------------------------------------- */
.main [data-testid="stMarkdownContainer"] h3 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.25rem !important;
}

/* ---- Page title gradient (h1) reaffirm --------------------- */
.main h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.15 !important;
}

/* ---- Section sub-headers (h2 from st.subheader) ------------ */
/* These are smaller than page titles; keep them clean.         */
.main [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    border-bottom: 1px solid var(--bt-divider, #e2e8f0) !important;
    padding-bottom: 0.35rem !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.6rem !important;
}

/* ---- Sidebar version footer -------------------------------- */
[data-testid="stSidebar"] .stCaption:last-of-type {
    position: absolute;
    bottom: 1.25rem;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 10px !important;
    color: #334155 !important;
}

/* ---- Glass morphism for metric cards in dark mode ---------- */
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetric"] {
    background: rgba(15,23,42,0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-color: rgba(99,102,241,0.20) !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetric"] {
        background: rgba(15,23,42,0.72) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-color: rgba(99,102,241,0.20) !important;
    }
}

/* ---- Entity card hover glow -------------------------------- */
.bt-entity-card {
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
.bt-entity-card:hover {
    border-left-color: #8b5cf6 !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.18), 0 2px 8px rgba(0,0,0,0.08) !important;
}

/* ---- Badge pill tighter radius ----------------------------- */
span[style*="border-radius:4px"][style*="font-weight:700"] {
    border-radius: 6px !important;
}

/* ---- Divider styling --------------------------------------- */
hr {
    border: none !important;
    border-top: 1px solid var(--bt-divider) !important;
    margin: 1.5rem 0 !important;
}

/* ---- Dataframe container ----------------------------------- */
[data-testid="stDataFrame"] > div {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ---- Smooth page transitions ------------------------------- */
.main .block-container > div {
    animation: bt-fadein 0.22s ease forwards;
}
@keyframes bt-fadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Sidebar active nav - polished glow -------------------- */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    position: relative;
    overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.15);
    pointer-events: none;
}

/* ---- Score card glow --------------------------------------- */
.bt-score-card {
    transition: box-shadow 0.25s ease !important;
}
.bt-score-card:hover {
    box-shadow: 0 6px 24px rgba(99,102,241,0.15) !important;
}

/* ---- PDF banner modern look -------------------------------- */
.bt-pdf-banner {
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: box-shadow 0.2s ease;
}
.bt-pdf-banner:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.18);
}

/* ---- Records search card: robust column alignment ---------- */
/* Ensures the "Search →" button aligns to the bottom of the   */
/* selectbox / text-input on all Streamlit versions.           */
.bt-search-card [data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
    gap: 0.5rem !important;
}
.bt-search-card [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}
.bt-search-card [data-testid="column"] > div:first-child {
    margin-top: auto !important;
}
/* Remove Streamlit's default top-margin on the button column  */
.bt-search-card [data-testid="column"]:last-child .stButton {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.bt-search-card [data-testid="column"]:last-child .stButton > button {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
}

/* ---- Page section subheaders ------------------------------ */
.main h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* ---- Sidebar nav: tighten divider spacing ------------------ */
[data-testid="stSidebar"] hr {
    margin: 0.5rem 0 !important;
}

/* ---- Dashboard: chart container subtle border -------------- */
[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ---- Upload dropzone polish -------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    transition: all 0.2s ease !important;
    background: rgba(99,102,241,0.02) !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within {
    background: rgba(99,102,241,0.06) !important;
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ---- Dark mode: PDF banner title text ---------------------- */
[data-testid="stApp"][data-theme="dark"] .bt-pdf-banner div[style*="color:var(--text-color"] {
    color: #f1f5f9 !important;
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
    "🗂️  Records": "records",
    "📊  Reports": "reports",
    "🔗  Datasources": "datasources",
    "📋  Logs": "logs",
    "🗄️  Database": "database",
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
        # DB connection status pill
        if _DB_IMPORTS_OK:
            _db_up = db_available()
            _dot = "🟢" if _db_up else "🔴"
            _label = "PostgreSQL connected" if _db_up else "DB offline (file mode)"
            st.markdown(
                f'<div style="font-size:11px;color:#475569;text-align:center;">'
                f'{_dot} {_label}</div>',
                unsafe_allow_html=True,
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
    st.markdown("# Dashboard")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
The Dashboard gives you an **at-a-glance health check** of all document processing.

- **Pending Review** — cases with high or medium risk that need your decision. Go to **Cases** to approve or reject them.
- **Documents Scanned** — total document verifications stored in the database.
- **High Risk** — documents where the Truth Score is critically low.
- **Avg Truth Score** — average score across all scans (100 = perfect integrity).

Use **Scan** (single file) or **Bulk Scan** (entire loan folder) to add new documents.
"""
        )

    if _DB_IMPORTS_OK and db_available():
        # ── Live DB stats ────────────────────────────────────────────────
        stats = db_dashboard_stats()
        if not stats:
            st.warning("Could not load dashboard statistics from the database.")
            return

        avg_s = stats.get("avg_score")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Entities", stats.get("entities", 0),
                      help="Unique applicants stored in the database.")
            if st.button("View →", key="dash_goto_records", use_container_width=True):
                st.session_state["page"] = "records"
                st.rerun()
        with m2:
            st.metric("Docs Scanned", stats.get("total_scans", 0),
                      help="Total document scans in the database.")
            if st.button("View →", key="dash_goto_reports", use_container_width=True):
                st.session_state["page"] = "reports"
                st.rerun()
        with m3:
            st.metric("Pending Review", stats.get("pending_review", 0),
                      help="Open cases requiring Approve / Reject.")
            if st.button("Review →", key="dash_goto_cases_pending", use_container_width=True, type="primary" if stats.get("pending_review", 0) > 0 else "secondary"):
                st.session_state["page"] = "cases"
                st.rerun()
        with m4:
            st.metric("High Risk", stats.get("high_risk", 0),
                      help="Documents with high tamper risk.")
            if st.button("View →", key="dash_goto_cases_risk", use_container_width=True, type="primary" if stats.get("high_risk", 0) > 0 else "secondary"):
                st.session_state["page"] = "cases"
                st.rerun()
        with m5:
            st.metric("Auto Approved", stats.get("auto_approved", 0),
                      help="Low-risk documents automatically cleared.")
            if st.button("View →", key="dash_goto_cases_auto", use_container_width=True):
                st.session_state["page"] = "cases"
                st.rerun()
        with m6:
            st.metric("Avg Score",
                      f"{avg_s}/100" if avg_s is not None else "—",
                      help="Average Truth Score across all scans (100 = perfect).")
            if st.button("View →", key="dash_goto_records_score", use_container_width=True):
                st.session_state["page"] = "records"
                st.rerun()

        st.divider()

        if stats.get("total_scans", 0) == 0:
            st.info("No documents scanned yet. Use **Scan** or **Bulk Scan** to get started.")
            return

        chart_col, info_col = st.columns([1, 2])

        with chart_col:
            st.subheader("Risk Distribution")
            risk_counts = {
                "High Risk": stats.get("high_risk", 0),
                "Medium Risk": stats.get("medium_risk", 0),
                "Low Risk": stats.get("low_risk", 0),
            }
            try:
                import pandas as pd
                st.bar_chart(pd.DataFrame({"Count": list(risk_counts.values())}, index=list(risk_counts.keys())))
            except ImportError:
                st.json(risk_counts)

        with info_col:
            st.subheader(f"Applicants ({stats.get('entities', 0)})")
            entities_list = stats.get("risk_by_entity", [])
            if entities_list:
                try:
                    import pandas as pd
                    df = pd.DataFrame(entities_list)[["entity_ref", "name", "scans"]]
                    df.columns = ["Reference", "Name", "Documents"]
                    st.dataframe(df, hide_index=True, use_container_width=True, height=280)
                except ImportError:
                    for e in entities_list:
                        st.write(f"{e['entity_ref']} — {e['name']} ({e['scans']} docs)")
            else:
                st.info("No entities yet.")

        # ── Pending cases quick-view ───────────────────────────────────
        if stats.get("pending_review", 0) > 0:
            st.divider()
            st.subheader(f"⛔ Cases Requiring Your Review ({stats['pending_review']})")
            st.caption("Go to **Cases** to Approve or Reject.")
            cases = service.list_cases()
            needs_review_cases = [c for c in cases if c.get("needs_review")]
            if needs_review_cases:
                try:
                    import pandas as pd
                    rows = [
                        {
                            "Case": c.get("case_key", ""),
                            "Type": c.get("document_type", "").replace("_", " ").title(),
                            "Docs": str(c.get("document_count", 0)),
                            "Risk": str(c.get("max_risk_level", "low")).title(),
                            "Status": str(c.get("status", "new")).replace("_", " ").title(),
                        }
                        for c in needs_review_cases
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                except ImportError:
                    for c in needs_review_cases:
                        st.write(c.get("case_key", ""))
        else:
            st.divider()
            st.success("✅ All cases resolved — nothing pending review.")
    else:
        # ── DB offline fallback (file-based) ─────────────────────────────
        st.info("📴 **Database offline** — showing file-based stats. Connect PostgreSQL for accurate counts.", icon=None)
        reports = service.list_reports()
        ver_reports = [r for r in reports if r.get("kind") == "verification"]
        scores = [r.get("truth_score") for r in ver_reports if isinstance(r.get("truth_score"), int)]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents Scanned", len(ver_reports))
        col2.metric("High Risk", sum(1 for r in ver_reports if r.get("risk_level") == "high"))
        col3.metric("Avg Truth Score", f"{round(sum(scores)/len(scores),1)}/100" if scores else "—")
        col4.metric("Reports on disk", len(reports))


# ---------------------------------------------------------------------------
# Entity-linking widget (shared by Scan + Bulk Scan pages)
# ---------------------------------------------------------------------------

def _render_entity_link_widget(key_prefix: str) -> tuple[str | None, dict | None]:
    """Render the 'Associate with a person' UI panel.

    Returns
    -------
    forced_ref : str | None
        entity_ref of an existing entity chosen by the user, or None.
    extra_identity : dict | None
        Identity fields typed by the user (used as hints when forced_ref is None).
    """
    _widget_expanded = True if key_prefix == "bulk" else False
    with st.expander("👤 Associate documents with a person (recommended)", expanded=_widget_expanded):
        st.markdown(
            """
Linking documents to an applicant **prevents duplicate entity records** and keeps all
their documents grouped under one profile in the Records screen.

**How it works:**
- *Search an existing person* — type their name, PAN, Aadhaar, email, or reference number.  
  All documents you scan will be linked to that profile.
- *Or enter identifying details* — if this is a new applicant, type their PAN / email / phone
  below. BaseTruth will auto-match future documents to the same person using these fields.
""",
        )

        link_mode = st.radio(
            "Link mode",
            ["Search existing person", "Enter applicant details manually"],
            horizontal=True,
            key=f"{key_prefix}_link_mode",
        )

        forced_ref: str | None = None
        extra_identity: dict | None = None

        if link_mode == "Search existing person":
            if _DB_IMPORTS_OK and db_available():
                search_q = st.text_input(
                    "Search by name / PAN / Aadhaar / email / phone / BT-ref",
                    key=f"{key_prefix}_entity_search",
                    placeholder="e.g. Aarini Parekh, MVWNV2212G, BT-000003…",
                )
                if search_q.strip():
                    matches = search_entities(search_q.strip(), "all", limit=10)
                    if matches:
                        opts = {
                            f"{m['entity_ref']}  —  {m['first_name']} {m['last_name']}  "
                            f"({m.get('pan_number') or m.get('email') or 'no id'})": m["entity_ref"]
                            for m in matches
                        }
                        chosen_label = st.selectbox(
                            "Select person",
                            list(opts.keys()),
                            key=f"{key_prefix}_entity_select",
                        )
                        forced_ref = opts[chosen_label]
                        st.success(f"✅ Scans will be linked to **{chosen_label.split('—')[0].strip()}**")
                    else:
                        st.info("No matching person found. Switch to 'Enter details manually' to create one.")
            else:
                st.warning("Database is offline — entity search unavailable.")

        else:  # Manual entry
            mc1, mc2 = st.columns(2)
            e_fn = mc1.text_input("First name", key=f"{key_prefix}_ei_fn", placeholder="Aarini")
            e_ln = mc2.text_input("Last name", key=f"{key_prefix}_ei_ln", placeholder="Parekh")
            mc3, mc4 = st.columns(2)
            e_pan = mc3.text_input("PAN number", key=f"{key_prefix}_ei_pan", placeholder="MVWNV2212G")
            e_aadh = mc4.text_input("Aadhaar number", key=f"{key_prefix}_ei_aadh", placeholder="1234 5678 9012")
            mc5, mc6 = st.columns(2)
            e_email = mc5.text_input("Email", key=f"{key_prefix}_ei_email")
            e_phone = mc6.text_input("Phone", key=f"{key_prefix}_ei_phone")
            if any([e_fn, e_ln, e_pan, e_aadh, e_email, e_phone]):
                extra_identity = {
                    "first_name": e_fn.strip(),
                    "last_name": e_ln.strip(),
                    "pan_number": e_pan.strip().upper(),
                    "aadhar_number": e_aadh.replace(" ", "").strip(),
                    "email": e_email.strip().lower(),
                    "phone": e_phone.strip(),
                }
                st.success("✅ Identity hints will be used to group documents under the right person.")

    return forced_ref, extra_identity


# ---------------------------------------------------------------------------
# Scan page (single document)
# ---------------------------------------------------------------------------

def _page_scan(service: BaseTruthService) -> None:
    st.markdown("# Scan Document")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Scan** verifies a single document end-to-end.

1. Upload the file (PDF, image, or LiteParse JSON) or paste a local path.
2. *(Optional)* Expand **"Associate with a person"** to link the result to an applicant's
   profile — this prevents duplicate entries in Records when the document doesn't include
   PAN or Aadhaar.
3. Hit **Run scan** and review the Truth Score, risk level, and forensic signals.
4. Download the JSON or PDF report to share with colleagues or attach to a loan file.
"""
        )

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

    forced_ref, extra_identity = _render_entity_link_widget("scan")

    if st.button("Run scan ->", type="primary"):
        report: Dict[str, Any] | None = None
        scan_error: str = ""
        with st.spinner("Running BaseTruth scan..."):
            try:
                if upload is not None:
                    temp_dir = Path(tempfile.mkdtemp(prefix="bt_upload_"))
                    saved_path = _save_uploaded_files([upload], temp_dir)[0]
                    report = service.scan_document(
                        saved_path,
                        forced_entity_ref=forced_ref or None,
                        extra_identity=extra_identity or None,
                    )
                elif path_input.strip():
                    report = service.scan_document(
                        path_input.strip(),
                        forced_entity_ref=forced_ref or None,
                        extra_identity=extra_identity or None,
                    )
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
    st.markdown("# Bulk Scan")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Bulk Scan** lets you process an entire mortgage / loan application folder in one go.

1. Upload all the applicant's documents at once (payslips, bank statements, PAN card, Aadhaar, etc.).
2. Expand **"Associate documents with a person"** — enter the applicant's PAN, email, or phone so
   every document in this batch links to the same profile in Records.  Without this, documents
   that don't embed PAN / Aadhaar may create separate "unknown" entity entries.
3. Tick **"Run cross-month payslip comparison"** if the batch includes multi-month payslips.
4. Hit **Run bulk scan**.  A case-report PDF is auto-generated and saved to `artifacts/case_reports/`.
"""
        )

    uploads = st.file_uploader(
        "Upload multiple documents",
        type=None,
        accept_multiple_files=True,
    )
    folder_input = st.text_input("Or scan all supported files from a folder on disk")
    compare_payslips = st.checkbox("Run cross-month payslip comparison after scan", value=True)

    bulk_forced_ref, bulk_extra_identity = _render_entity_link_widget("bulk")

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
            # Track entity_ref: after the first scan we reuse the same entity_ref
            # for all remaining documents in this batch (prevents duplicate entities).
            _batch_entity_ref: str | None = bulk_forced_ref or None
            prog = st.progress(0)
            for i, p in enumerate(paths):
                try:
                    r = service.scan_document(
                        p,
                        forced_entity_ref=_batch_entity_ref or None,
                        extra_identity=bulk_extra_identity or None,
                    )
                    _new_reports.append(r)
                    # Capture entity_ref from first successful scan and reuse it
                    if _batch_entity_ref is None and r.get("_entity_ref"):
                        _batch_entity_ref = r["_entity_ref"]
                except Exception as exc:  # noqa: BLE001
                    _new_errors.append(f"{p.name}: {exc}")
                prog.progress((i + 1) / len(paths))

        # Persist results so they survive reruns triggered by any button inside this page
        st.session_state["bt_bulk_reports"] = _new_reports
        st.session_state["bt_bulk_errors"] = _new_errors
        st.session_state["bt_bulk_compare"] = compare_payslips
        st.session_state["bt_bulk_entity_ref"] = _batch_entity_ref
        # Clear any previously generated bundle PDF on fresh scan
        st.session_state.pop("bt_bundle_pdf_bytes", None)
        st.session_state.pop("bt_bundle_pdf_path", None)

        # ── Auto-generate case report PDF immediately after scan ──────────────
        if _new_reports:
            try:
                import datetime
                from basetruth.reporting.pdf import render_case_bundle_pdf
                with st.spinner("Generating case report PDF…"):
                    _auto_reconciliation = service.reconcile_income_documents(_new_reports)
                    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _auto_title = f"Case Report — {_ts}"
                    _pdf_bytes = render_case_bundle_pdf(
                        reports=_new_reports,
                        reconciliation=_auto_reconciliation,
                        case_title=_auto_title,
                    )
                
                # Flag cases for cross-document anomalies
                if _auto_reconciliation.get("anomalies"):
                    for _err in _auto_reconciliation["anomalies"]:
                        st.session_state["bt_bulk_errors"].append(f"Anomaly Detected: {_err.get('details', {}).get('explanation', _err.get('type'))}")
                    
                    for _rep in _new_reports:
                        _ent_ref = _rep.get("_entity_ref")
                        _doc_type = _rep.get("structured_summary", {}).get("document", {}).get("type", "generic")
                        _c_key = f"{_doc_type}::{_ent_ref}" if _ent_ref else service._case_key_for_report(_rep)
                        
                        service.update_case(
                            _c_key,
                            status="triage",
                            priority="high",
                            disposition="open",
                            note_text="Cross-document reconciliation uncovered a discrepancy flagged in the bulk case report.",
                            note_author="system"
                        )
                # Save to filesystem for regulatory / audit access
                _reports_dir = service.artifact_root / "case_reports"
                _reports_dir.mkdir(parents=True, exist_ok=True)
                _pdf_path = _reports_dir / f"{_ts}_case_report.pdf"
                _pdf_path.write_bytes(_pdf_bytes)
                st.session_state["bt_bundle_pdf_bytes"] = _pdf_bytes
                st.session_state["bt_bundle_pdf_title"] = f"{_ts}_case_report"
                st.session_state["bt_bundle_pdf_path"] = str(_pdf_path)
                # Also upload case report to MinIO under entity folder
                if _DB_IMPORTS_OK and _batch_entity_ref:
                    try:
                        minio_upload(
                            f"{_batch_entity_ref}/case_reports/{_ts}_case_report.pdf",
                            _pdf_bytes,
                            "application/pdf",
                        )
                    except Exception:
                        pass  # non-fatal
            except Exception as _pdf_err:
                st.session_state["bt_bundle_pdf_bytes"] = None
                st.warning(f"Scan complete — PDF generation failed: {_pdf_err}")

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

        # ── Retroactively flag auto-approved cases for review ─────────────
        _batch_entity_ref = st.session_state.get("bt_bulk_entity_ref")
        if _batch_entity_ref:
            _anomaly_types = [a.get("type", "") for a in income_anomalies]
            _anomaly_count = len(income_anomalies)
            _reopened = service.flag_entity_cases_for_review(
                _batch_entity_ref,
                reason=f"{_anomaly_count} cross-document income inconsistenc{'y' if _anomaly_count == 1 else 'ies'} detected",
                anomaly_types=_anomaly_types,
            )
            if _reopened > 0:
                st.warning(
                    f"🔓 **{_reopened} previously auto-approved case(s) have been reopened** for "
                    f"entity {_batch_entity_ref} due to cross-document anomalies. "
                    f"Go to **Cases** to review them."
                )
    elif evidence_rows:
        st.success("✅ Income figures are consistent across all documents.")

    # --- Case Report PDF (auto-generated after scan) ---
    st.divider()
    _pdf_bytes_ready = st.session_state.get("bt_bundle_pdf_bytes")
    _pdf_path_saved = st.session_state.get("bt_bundle_pdf_path", "")
    if _pdf_bytes_ready:
        _pdf_title = st.session_state.get("bt_bundle_pdf_title", "case_report")
        _safe_fname = "".join(c if c.isalnum() or c in "-_ " else "_" for c in _pdf_title).strip() + ".pdf"
        st.markdown(
            f'<div class="bt-pdf-banner">'
            f'<span style="font-size:1.5rem;">📄</span>'
            f'<div><div style="font-weight:700;font-size:0.95rem;color:var(--text-color,#1e293b);">'
            f'Case Report PDF generated automatically</div>'
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:2px;">'
            f'Saved to: {_pdf_path_saved or "artifacts/case_reports/"}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇  Download Case Report PDF",
            data=_pdf_bytes_ready,
            file_name=_safe_fname,
            mime="application/pdf",
            key="dl_bundle_pdf",
            use_container_width=True,
            type="primary",
        )
    else:
        st.info(
            "📄 **Case Report PDF** — A PDF report will be generated automatically "
            "after scanning documents. It covers all documents, income reconciliation, "
            "and an overall verdict suitable for loan officer or regulator review."
        )


# ---------------------------------------------------------------------------
# Cases page
# ---------------------------------------------------------------------------

def _page_cases(service: BaseTruthService) -> None:
    st.markdown("# Cases")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
A **case** is created automatically whenever a document is scanned.

- **Needs Review** — your action queue for high / medium risk documents.
  Press **✅ Approve** or **❌ Reject** directly on the card.
- **Resolved** — cases you have already decided on (Approved or Rejected).
- **Auto-Approved** — low-risk documents cleared automatically; no action needed.

When PostgreSQL is connected, cases are read from the database (accurate, reset-safe).
Falling back to local files when the database is offline.
"""
        )

    # Prefer DB-driven cases when available
    use_db = _DB_IMPORTS_OK and db_available()
    if use_db:
        cases = list_cases_from_db()
    else:
        cases = service.list_cases()

    if not cases:
        st.info("No cases yet. Scan documents first and cases will appear here automatically.")
        return

    # ── Filter ────────────────────────────────────────────────────────────────
    cases_filter = st.text_input(
        "🔍 Filter cases",
        placeholder="Entity name, BT-reference, case key, or document type…",
        key="cases_filter",
    ).strip().lower()
    if cases_filter:
        cases = [
            c for c in cases
            if cases_filter in (c.get("entity_name") or "").lower()
            or cases_filter in (c.get("entity_ref") or "").lower()
            or cases_filter in (c.get("document_type") or "").lower()
            or cases_filter in (c.get("case_key") or "").lower()
        ]
        if not cases:
            st.info("No cases match your filter.")
            return

    needs_review = [c for c in cases if c.get("needs_review")]
    resolved     = [c for c in cases if c.get("disposition") in ("cleared", "fraud_confirmed")]
    auto_ok      = [c for c in cases if not c.get("needs_review") and c.get("disposition") not in ("cleared", "fraud_confirmed")]

    tab_labels = [
        f"⛔ Needs Review ({len(needs_review)})",
        f"✅ Resolved ({len(resolved)})",
    ]
    if auto_ok:
        tab_labels.append(f"🔵 Auto-Approved ({len(auto_ok)})")

    tabs = st.tabs(tab_labels)

    def _render_grouped(case_list: list, show_actions: bool) -> None:
        """Group a list of cases by entity and render each entity as a section."""
        from collections import defaultdict  # noqa: PLC0415
        by_entity: dict = defaultdict(list)
        for c in case_list:
            by_entity[c.get("entity_ref") or "unlinked"].append(c)
        for ref in sorted(by_entity.keys()):
            entity_cases = by_entity[ref]
            name = entity_cases[0].get("entity_name", "") or ref
            header = f"👤 **{name}** &nbsp; `{ref}` &nbsp;—&nbsp; {len(entity_cases)} case(s)"
            with st.expander(header, expanded=show_actions):
                for case in entity_cases:
                    _render_case_card(service, case, show_actions=show_actions, use_db=use_db)

    with tabs[0]:
        if not needs_review:
            st.success("🎉 No cases pending review — all documents have been assessed.")
        else:
            _render_grouped(needs_review, show_actions=True)

    with tabs[1]:
        if not resolved:
            st.info("No resolved cases yet.")
        else:
            _render_grouped(resolved, show_actions=False)

    if auto_ok:
        with tabs[2]:
            _render_grouped(auto_ok, show_actions=False)


def _render_case_card(
    service: BaseTruthService,
    case: Dict[str, Any],
    *,
    show_actions: bool,
    use_db: bool = False,
) -> None:
    """Render a single case as an expandable card with Approve / Reject buttons.

    When *use_db* is True the Advanced panel is populated from data already
    present in the case dict (no file-system read needed), making page render
    very fast even with many cases.
    """
    case_key    = case.get("case_key", "")
    risk        = case.get("max_risk_level", "low")
    disposition = case.get("disposition", "open")
    doc_type    = case.get("document_type", "").replace("_", " ").title()
    doc_count   = case.get("document_count", 0)
    entity_ref  = case.get("entity_ref", "")
    entity_name = case.get("entity_name", "")
    risk_icon   = {"high": "🚨", "medium": "⚠️", "low": "✅"}.get(risk, "🔷")
    disp_icon   = _DISPOSITION_ICONS.get(disposition, "")

    # Rich header: show person name when available
    name_part = f"  —  {entity_name}" if entity_name else ""
    ref_part  = f"  ({entity_ref})" if entity_ref and entity_ref != "unlinked" else ""
    header = (
        f"{risk_icon} {doc_type}{name_part}{ref_part}"
        f"  |  {doc_count} doc(s)  |  {disp_icon} {disposition.replace('_', ' ').title()}"
    )

    with st.expander(header, expanded=show_actions and risk == "high"):
        # ── Approve / Reject buttons at the very top ──────────────────────
        if show_actions:
            btn_c1, btn_c2, _ = st.columns([1, 1, 3])
            if btn_c1.button("✅  Approve", key=f"approve_{case_key}", use_container_width=True, type="primary"):
                service.update_case(case_key, status="closed", disposition="cleared",
                                    note_text="Manually approved by analyst.", note_author="analyst")
                st.toast("✅ Case approved.", icon="✅")
                st.rerun()
            if btn_c2.button("❌  Reject", key=f"reject_{case_key}", use_container_width=True):
                service.update_case(case_key, status="closed", disposition="fraud_confirmed",
                                    note_text="Rejected by analyst — fraud confirmed.", note_author="analyst")
                st.toast("❌ Case rejected.", icon="❌")
                st.rerun()
            st.divider()
        else:
            verdict_color = "#16a34a" if disposition == "cleared" else "#dc2626" if disposition == "fraud_confirmed" else "#6366f1"
            verdict_label = {"cleared": "Approved ✅", "fraud_confirmed": "Rejected ❌"}.get(disposition, disposition.replace("_", " ").title())
            st.markdown(
                f'<div style="font-size:1rem;font-weight:700;color:{verdict_color};margin-bottom:8px;">'
                f'{verdict_label}</div>',
                unsafe_allow_html=True,
            )

        # ── Case info ─────────────────────────────────────────────────────
        st.markdown(
            f"**Risk:** {_badge(risk)}  &nbsp;&nbsp;  "
            f"**Priority:** {case.get('priority', 'normal').title()}  &nbsp;&nbsp;  "
            f"**Assignee:** {case.get('assignee') or '—'}",
            unsafe_allow_html=True,
        )

        # ── Linked documents ──────────────────────────────────────────────
        docs = case.get("documents", [])
        if docs:
            st.markdown("**Documents:**")
            for doc in docs:
                src    = doc.get("source_name", "unknown")
                dlvl   = str(doc.get("risk_level", "low"))
                dscore = doc.get("truth_score", "")
                st.markdown(
                    f"&nbsp;&nbsp;{_badge(dlvl)} {src}  —  Score: **{dscore if isinstance(dscore, int) else '—'}**",
                    unsafe_allow_html=True,
                )

        # ── Advanced workflow (lazy-loaded via session-state toggle) ──────
        adv_key = f"_adv_open_{case_key}"
        if not st.session_state.get(adv_key):
            if st.button("⚙️ Advanced options", key=f"adv_btn_{case_key}", use_container_width=False):
                st.session_state[adv_key] = True
                st.rerun()
        else:
            if st.button("▲ Hide advanced", key=f"adv_hide_{case_key}", use_container_width=False):
                st.session_state[adv_key] = False
                st.rerun()

            # Load workflow data — from case dict (DB mode) OR file (legacy mode)
            if use_db:
                workflow = {
                    "status":      case.get("status", "new"),
                    "disposition": case.get("disposition", "open"),
                    "priority":    case.get("priority", "normal"),
                    "assignee":    case.get("assignee", ""),
                    "labels":      case.get("labels", []),
                    "notes":       case.get("notes", []),
                }
            else:
                try:
                    case_detail = service.get_case_detail(case_key)
                    workflow = case_detail["workflow"]
                except KeyError:
                    st.warning("Case detail not found.")
                    return

            statuses     = ["new", "triage", "investigating", "pending_client", "closed"]
            dispositions = ["open", "monitor", "escalate", "cleared", "fraud_confirmed"]
            priorities   = ["low", "normal", "high", "critical"]
            with st.form(f"adv_form_{case_key}"):
                wf1, wf2, wf3 = st.columns(3)
                cur_s = str(workflow.get("status", "new"))
                cur_d = str(workflow.get("disposition", "open"))
                cur_p = str(workflow.get("priority", "normal"))
                status_sel  = wf1.selectbox("Status", statuses,
                    index=statuses.index(cur_s) if cur_s in statuses else 0, key=f"s_{case_key}")
                disp_sel    = wf2.selectbox("Disposition", dispositions,
                    index=dispositions.index(cur_d) if cur_d in dispositions else 0, key=f"d_{case_key}")
                prio_sel    = wf3.selectbox("Priority", priorities,
                    index=priorities.index(cur_p) if cur_p in priorities else 1, key=f"p_{case_key}")
                assignee_val = st.text_input("Assignee", value=str(workflow.get("assignee", "")), key=f"a_{case_key}")
                labels_val   = st.text_input("Labels (comma-separated)", value=", ".join(workflow.get("labels", [])), key=f"l_{case_key}")
                note_author  = st.text_input("Note author", value="analyst", key=f"na_{case_key}")
                note_text    = st.text_area("Add a note", placeholder="Observations, evidence, next steps…", key=f"nt_{case_key}")
                if st.form_submit_button("Save", type="primary"):
                    service.update_case(case_key, status=status_sel, disposition=disp_sel,
                                        priority=prio_sel, assignee=assignee_val,
                                        labels=[i.strip() for i in labels_val.split(",") if i.strip()],
                                        note_text=note_text, note_author=note_author)
                    st.success("Updated.")
                    st.rerun()
            notes = workflow.get("notes", [])
            if notes:
                st.markdown(f"**Notes ({len(notes)}):**")
                for note in reversed(notes):
                    ts     = str(note.get("created_at", ""))[:19].replace("T", " ")
                    author = note.get("author", "")
                    st.markdown(
                        f'<div style="background:var(--bt-note-bg);border-left:3px solid var(--bt-note-accent);'
                        f'padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:8px;">'
                        f'<span style="font-size:11px;color:var(--bt-text-muted);">{ts} · {author}</span><br>'
                        f'{note.get("text", "")}</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Reports page
# ---------------------------------------------------------------------------

def _page_reports(service: BaseTruthService) -> None:
    st.markdown("# Reports")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Reports are grouped **by applicant** — one section per person, listing every document
scanned for them with individual download buttons.

- Each applicant card shows their name / reference, a summary table of all their documents,
  and a **Download PDF Report** button for each scan.
- To view raw JSON or run a new scan, use the **Scan** page.
- To search across all applicants, use the **Records** page.
"""
        )

    if _DB_IMPORTS_OK and db_available():
        # ── DB-driven: one section per entity ─────────────────────────────
        entities = get_all_entities_with_scans()
        if not entities:
            st.info("No scans in the database yet. Use **Scan** or **Bulk Scan** to process documents.")
            return

        # ── Case Bundle Reports (generated by Bulk Scan) ──────────────────
        _case_report_tab, _scan_report_tab = st.tabs(["📄 Case Reports", "📋 Individual Scan Reports"])

        with _case_report_tab:
            st.caption("Case reports are generated automatically when you run a **Bulk Scan**. "
                       "Each report covers all documents in the batch, income reconciliation, and an overall verdict.")
            _case_reports_found: list[tuple] = []  # (label, bytes, filename, key)

            # 1. Try MinIO — look for case_reports under each entity prefix
            try:
                _all_minio_objs = minio_list_objects(limit=500)
                for obj in _all_minio_objs:
                    if "case_reports/" in obj["key"] and obj["key"].endswith(".pdf"):
                        try:
                            _pdf_data = minio_get_object(obj["key"])
                            if _pdf_data:
                                _parts = obj["key"].split("/")
                                _entity_ref = _parts[0] if len(_parts) > 2 else "—"
                                _fname = _parts[-1]
                                _size_kb = round(obj.get("size_bytes", 0) / 1024, 1)
                                _modified = obj.get("last_modified", "")[:19].replace("T", " ")
                                _case_reports_found.append((
                                    _entity_ref,
                                    f"📄 **{_fname}**  ·  {_size_kb} KB  ·  {_modified}",
                                    _pdf_data,
                                    _fname,
                                    f"cr_minio_{_fname}",
                                ))
                        except Exception:
                            pass
            except Exception:
                pass

            # 2. Fallback — check artifacts/case_reports/ on disk
            if not _case_reports_found:
                _cr_dir = service.artifact_root / "case_reports"
                if _cr_dir.exists():
                    import datetime as _dt
                    for _pdf_path in sorted(_cr_dir.glob("*.pdf"), reverse=True):
                        _size_kb = round(_pdf_path.stat().st_size / 1024, 1)
                        _ts_str = _dt.datetime.fromtimestamp(_pdf_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                        _case_reports_found.append((
                            "System Reports (Disk)",
                            f"📄 **{_pdf_path.name}**  ·  {_size_kb} KB  ·  {_ts_str}",
                            _pdf_path.read_bytes(),
                            _pdf_path.name,
                            f"cr_disk_{_pdf_path.stem}",
                        ))

            if _case_reports_found:
                search_cr = st.text_input(
                    "🔍 Filter case reports",
                    placeholder="Search by BT-reference or filename...",
                    key="cr_search"
                ).strip().lower()

                filtered_cr = [
                    rpt for rpt in _case_reports_found
                    if not search_cr or search_cr in rpt[0].lower() or search_cr in rpt[1].lower() or search_cr in rpt[3].lower()
                ]

                if filtered_cr:
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for _eref, _label, _data, _fname, _key in filtered_cr:
                        grouped[_eref].append((_label, _data, _fname, _key))
                    
                    for _eref, items in grouped.items():
                        _ent_name = ""
                        if _eref != "System Reports (Disk)":
                            _ent = next((e for e in entities if e.get("entity_ref") == _eref), None)
                            if _ent:
                                _ent_name = f" — {_ent.get('first_name','')} {_ent.get('last_name','')}".strip()
                        
                        with st.expander(f"👤 {_eref}{_ent_name}  ({len(items)} docs)", expanded=True):
                            for _label, _data, _fname, _key in items:
                                c1, c2 = st.columns([5, 1])
                                c1.markdown(_label, unsafe_allow_html=True)
                                c2.download_button(
                                    "⬇ Download",
                                    data=_data,
                                    file_name=_fname,
                                    mime="application/pdf",
                                    key=_key,
                                    use_container_width=True,
                                )
                else:
                    st.info("No matching case reports found.")
            else:
                st.info("No case reports yet. Run a **Bulk Scan** to generate one.")

        with _scan_report_tab:
            # ── Search / filter ──────────────────────────────────────────────
            search_rpt = st.text_input(
                "🔍 Filter applicants",
                placeholder="Name, PAN, email, or BT-reference…",
                key="rpt_search",
            ).strip().lower()
            if search_rpt:
                filtered = [
                    e for e in entities
                    if search_rpt in (e.get("name") or "").lower()
                    or search_rpt in (e.get("pan_number") or "").lower()
                    or search_rpt in (e.get("email") or "").lower()
                    or search_rpt in (e.get("entity_ref") or "").lower()
                ]
            else:
                filtered = entities

            st.caption(f"{len(filtered)} applicant(s) shown — {sum(len(e['scans']) for e in filtered)} document(s)")

            import pandas as pd  # noqa: PLC0415

            for ent in filtered:
                ref = ent["entity_ref"]
                name = ent["name"] or ref
                scans = ent["scans"]
                pan = ent["pan_number"]
                email = ent["email"]
                sub = "  ·  ".join(filter(None, [pan, email]))
                hdr = f"👤 **{name}** — {ref}" + (f"  ·  {sub}" if sub else "")

                with st.expander(hdr + f"  ({len(scans)} doc{'s' if len(scans) != 1 else ''})", expanded=False):
                    if not scans:
                        st.info("No scans linked to this entity.")
                        continue

                    # Summary table
                    rows = [
                        {
                            "Document": s["source_name"],
                            "Type": s["document_type"].replace("_", " ").title(),
                            "Risk": s["risk_level"].title(),
                            "Score": _display_truth_score(s["truth_score"]),
                            "Scanned": s["generated_at"][:19].replace("T", " ") if s["generated_at"] else "—",
                            "PDF": "✅" if s["has_pdf"] else "—",
                        }
                        for s in scans
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                    # Per-scan PDF download buttons
                    st.markdown("**Download individual scan reports:**")
                    all_pdf_candidates = scans  # try all scans — fall back to MinIO
                    pdf_buttons: list[tuple] = []  # (label, data, filename, key)
                    for s in all_pdf_candidates:
                        stem = Path(s["source_name"]).stem
                        pdf_data: bytes | None = None
                        if s["has_pdf"]:
                            pdf_data = get_scan_pdf(s["id"])
                        # MinIO fallback if PDF not in DB
                        if not pdf_data:
                            try:
                                pdf_data = minio_get_object(f"{ref}/{stem}_report.pdf")
                            except Exception:  # noqa: BLE001
                                pass
                        if pdf_data:
                            pdf_buttons.append((
                                f"⬇ {stem[:20]}",
                                pdf_data,
                                f"{ref}_{stem}_report.pdf",
                                f"rpt_pdf_{s['id']}",
                            ))
                    if pdf_buttons:
                        btn_cols = st.columns(min(len(pdf_buttons), 3))
                        for idx, (label, data, fname, key) in enumerate(pdf_buttons):
                            btn_cols[idx % 3].download_button(
                                label,
                                data=data,
                                file_name=fname,
                                mime="application/pdf",
                                key=key,
                                use_container_width=True,
                            )
                    else:
                        st.caption("No PDF reports available for this applicant. Re-scan to generate them.")
    else:
        # ── DB offline fallback (file-based, flat list) ───────────────────
        st.info("📴 Database offline — showing file-based reports. Connect PostgreSQL for entity-grouped view.")
        artifact_root = service.artifact_root
        bundle_dir = artifact_root / "case_reports"
        bundle_pdfs = sorted(bundle_dir.glob("*.pdf"), reverse=True) if bundle_dir.exists() else []
        if bundle_pdfs:
            st.subheader("📄 Case Bundle PDFs")
            import datetime as _dt
            for pdf_path in bundle_pdfs:
                size_kb = round(pdf_path.stat().st_size / 1024, 1)
                ts_str = _dt.datetime.fromtimestamp(pdf_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{pdf_path.name}**  ·  <span style='color:#94a3b8;font-size:11px;'>{ts_str} · {size_kb} KB</span>", unsafe_allow_html=True)
                c2.download_button("⬇", data=pdf_path.read_bytes(), file_name=pdf_path.name,
                                   mime="application/pdf", key=f"bundle_{pdf_path.stem}", use_container_width=True)
        else:
            st.info("No reports found. Run a Bulk Scan to generate them.")


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
    st.markdown("# Datasources")
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
    st.markdown("# Settings")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Settings control how BaseTruth stores data and how the service is run.

- **Artifact root** — the local folder where all reports, structured summaries, and case records are written. Change it in the sidebar to point BaseTruth at a different workspace or network share.
- **Product information** — version numbers, runtime info, and API endpoint details.
- **Quick start commands** — copy-paste commands to start/stop the service, run tests, or rebuild the Docker containers.

Most settings are controlled via environment variables in `.env`. See the README for a full reference.
"""
        )

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
# Log Analyzer page
# ---------------------------------------------------------------------------


def _page_logs() -> None:
    import pandas as pd  # noqa: PLC0415
    from datetime import datetime as _dt  # noqa: PLC0415

    # ── Custom CSS for the modern log analyzer ──────────────────────────────
    st.markdown(
        """
        <style>
        .log-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
        .log-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
        .metric-row { display:flex; gap:14px; margin:16px 0 20px 0; }
        .metric-card {
            flex:1; padding:18px 20px; border-radius:14px;
            display:flex; flex-direction:column; gap:4px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .metric-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.12); }
        .metric-card .mc-value { font-size:1.9rem; font-weight:800; line-height:1.1; }
        .metric-card .mc-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.75; font-weight:600; }
        .mc-total   { background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); color:#3730a3; }
        .mc-error   { background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%); color:#991b1b; }
        .mc-warn    { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color:#92400e; }
        .mc-info    { background: linear-gradient(135deg, #d1fae5 0%, #6ee7b7 100%); color:#065f46; }
        .quick-filters { display:flex; gap:8px; flex-wrap:wrap; margin:2px 0 14px 0; }
        .log-detail-card {
            background: #f8fafc; border:1px solid #e2e8f0; border-radius:12px;
            padding:16px 20px; margin-top:10px;
        }
        .log-tail-row {
            padding:7px 14px; border-radius:8px; margin-bottom:4px; font-size:.82rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            border-left:4px solid transparent;
        }
        .log-tail-error   { background:#fff1f2; border-left-color:#ef4444; color:#991b1b; }
        .log-tail-warning { background:#fffbeb; border-left-color:#f59e0b; color:#92400e; }
        .log-tail-info    { background:#f0fdf4; border-left-color:#22c55e; color:#166534; }
        .log-tail-debug   { background:#f8fafc; border-left-color:#94a3b8; color:#64748b; }
        .module-chip {
            display:inline-block; padding:3px 10px; border-radius:20px; font-size:.72rem;
            font-weight:600; margin:2px; background:#e2e8f0; color:#334155;
        }
        .module-chip-hot { background:#fecaca; color:#991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header row ──────────────────────────────────────────────────────────
    _hdr_l, _hdr_r = st.columns([6, 1])
    _hdr_l.markdown('<div class="log-header"><h1>📊 Log Analyzer</h1></div>', unsafe_allow_html=True)
    if _hdr_r.button("🔄 Refresh", use_container_width=True, key="log_refresh"):
        st.rerun()

    # ── Load log file ───────────────────────────────────────────────────────
    lp = _log_path() if _LOGGER_OK else None
    if lp is None or not lp.exists():
        st.info(
            "No log file found yet. Run some scans and the log file will appear here.\n\n"
            f"Expected location: `{lp}`"
        )
        return

    raw_lines: list[str] = []
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            for _line in fh:
                _line = _line.strip()
                if _line:
                    raw_lines.append(_line)
    except Exception as exc:
        st.error(f"Could not read log file: {exc}")
        return

    records: list[dict] = []
    for _line in raw_lines:
        try:
            records.append(json.loads(_line))
        except json.JSONDecodeError:
            records.append({"ts": "", "level": "RAW", "msg": _line, "module": "", "func": "", "logger": ""})

    if not records:
        st.info("Log file exists but is empty. Run some scans first.")
        return

    df = pd.DataFrame(records)
    for col in ["ts", "level", "logger", "module", "func", "line", "msg"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")

    _n_total = len(df)
    _n_err = int((df["level"] == "ERROR").sum())
    _n_warn = int((df["level"] == "WARNING").sum())
    _n_info = int(((df["level"] == "INFO") | (df["level"] == "DEBUG")).sum())

    # ── Hero metric cards ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card mc-total">
            <span class="mc-value">{_n_total:,}</span>
            <span class="mc-label">Total Entries</span>
          </div>
          <div class="metric-card mc-error">
            <span class="mc-value">{_n_err:,}</span>
            <span class="mc-label">Errors</span>
          </div>
          <div class="metric-card mc-warn">
            <span class="mc-value">{_n_warn:,}</span>
            <span class="mc-label">Warnings</span>
          </div>
          <div class="metric-card mc-info">
            <span class="mc-value">{_n_info:,}</span>
            <span class="mc-label">Info + Debug</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Charts row: timeline + level distribution ────────────────────────────
    _chart_l, _chart_r = st.columns([2, 1])

    with _chart_l:
        st.markdown("##### 📈 Log Volume Timeline")
        if df["ts"].str.len().max() > 10:
            try:
                _ts_series = pd.to_datetime(df["ts"], errors="coerce")
                _ts_valid = _ts_series.dropna()
                if len(_ts_valid) > 2:
                    _timeline_df = _ts_valid.dt.floor("min").value_counts().sort_index().rename("count").reset_index()
                    _timeline_df.columns = ["time", "count"]
                    st.area_chart(_timeline_df.set_index("time"), height=220, use_container_width=True)
                else:
                    st.caption("Not enough timestamped entries for a timeline chart.")
            except Exception:
                st.caption("Could not parse timestamps for timeline.")
        else:
            st.caption("No timestamps available.")

    with _chart_r:
        st.markdown("##### 🎯 Level Distribution")
        _level_counts = df["level"].value_counts()
        # Build a small horizontal bar chart styled with Streamlit
        if len(_level_counts) > 0:
            _lc_df = _level_counts.reset_index()
            _lc_df.columns = ["Level", "Count"]
            st.bar_chart(_lc_df.set_index("Level"), height=220, use_container_width=True)

    # ── Module heatmap ──────────────────────────────────────────────────────
    st.markdown("##### 🧩 Module Activity")
    _module_stats = df.groupby("module")["level"].value_counts().unstack(fill_value=0)
    _top_modules = df["module"].value_counts().head(10)
    _chips_html = ""
    for _mod, _cnt in _top_modules.items():
        if not _mod:
            continue
        _mod_errors = int((df[df["module"] == _mod]["level"] == "ERROR").sum())
        _cls = "module-chip-hot" if _mod_errors > 0 else ""
        _err_tag = f" · {_mod_errors} err" if _mod_errors else ""
        _chips_html += f'<span class="module-chip {_cls}">{_mod} ({_cnt}{_err_tag})</span>'
    if _chips_html:
        st.markdown(f'<div style="margin:8px 0 16px 0;">{_chips_html}</div>', unsafe_allow_html=True)
    else:
        st.caption("No module data available.")

    st.divider()

    # ── Quick filter buttons + filters ──────────────────────────────────────
    st.markdown("##### 🔍 Filter Logs")
    _qf_cols = st.columns(7)
    _preset = "ALL"
    if _qf_cols[0].button("🔴 Errors Only", use_container_width=True, key="qf_err"):
        st.session_state["log_level_filter_v2"] = "ERROR"
    if _qf_cols[1].button("🟡 Warnings", use_container_width=True, key="qf_warn"):
        st.session_state["log_level_filter_v2"] = "WARNING"
    if _qf_cols[2].button("🟢 Info", use_container_width=True, key="qf_info"):
        st.session_state["log_level_filter_v2"] = "INFO"
    if _qf_cols[3].button("⚪ Debug", use_container_width=True, key="qf_debug"):
        st.session_state["log_level_filter_v2"] = "DEBUG"
    if _qf_cols[4].button("📋 All Levels", use_container_width=True, key="qf_all"):
        st.session_state["log_level_filter_v2"] = "ALL"

    # Persistent filter state
    _active_level = st.session_state.get("log_level_filter_v2", "ALL")

    _f1, _f2, _f3 = st.columns([1.5, 2, 3])
    level_opts = ["ALL"] + sorted(df["level"].unique().tolist())
    _default_idx = level_opts.index(_active_level) if _active_level in level_opts else 0
    module_opts = ["ALL"] + sorted([m for m in df["module"].unique() if m])
    chosen_level = _f1.selectbox("Level", level_opts, index=_default_idx, key="log_level_sel_v2")
    chosen_module = _f2.selectbox("Module", module_opts, key="log_module_sel_v2")
    search_text = _f3.text_input("Search messages", placeholder="keyword…", key="log_search_v2")

    view = df.copy()
    if chosen_level != "ALL":
        view = view[view["level"] == chosen_level]
    if chosen_module != "ALL":
        view = view[view["module"] == chosen_module]
    if search_text:
        view = view[view["msg"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Showing **{len(view):,}** of {len(df):,} entries")

    # ── Styled log table ─────────────────────────────────────────────────────
    _LEVEL_COLORS: dict[str, str] = {
        "ERROR":   "background-color:#fee2e2;color:#991b1b;font-weight:700",
        "WARNING": "background-color:#fef9c3;color:#854d0e;font-weight:600",
        "INFO":    "background-color:#f0fdf4;color:#166534",
        "DEBUG":   "background-color:#f1f5f9;color:#64748b",
    }

    def _style_level(val: str) -> str:
        return _LEVEL_COLORS.get(val, "")

    display_cols = [c for c in ["ts", "level", "logger", "func", "msg"] if c in view.columns]
    rename_map = {"ts": "Timestamp", "level": "Level", "logger": "Source", "func": "Function", "msg": "Message"}

    if len(view) > 0:
        styled = (
            view[display_cols]
            .rename(columns=rename_map)
            .style.map(_style_level, subset=["Level"])
        )
        st.dataframe(styled, hide_index=True, use_container_width=True, height=480)
    else:
        st.info("No log entries match your filters.")

    # ── Live tail — last 15 entries ──────────────────────────────────────────
    st.divider()
    st.markdown("##### 🔴 Live Tail — Latest Entries")
    _tail = view.tail(15).iloc[::-1]  # most recent first
    _tail_html = ""
    for _, row in _tail.iterrows():
        _lvl = str(row.get("level", "")).upper()
        _cls = {
            "ERROR": "log-tail-error",
            "WARNING": "log-tail-warning",
            "INFO": "log-tail-info",
            "DEBUG": "log-tail-debug",
        }.get(_lvl, "log-tail-debug")
        _ts_short = str(row.get("ts", ""))[-8:] if len(str(row.get("ts", ""))) > 8 else str(row.get("ts", ""))
        _mod = row.get("logger", "") or row.get("module", "")
        _msg = str(row.get("msg", ""))[:200]
        _tail_html += (
            f'<div class="log-tail-row {_cls}">'
            f'<strong>[{_lvl}]</strong> '
            f'<span style="opacity:.6">{_ts_short}</span> '
            f'<span style="color:#6366f1;font-weight:600">{_mod}</span> '
            f'— {_msg}'
            f'</div>'
        )
    if _tail_html:
        st.markdown(_tail_html, unsafe_allow_html=True)
    else:
        st.caption("No entries to display.")

    # ── JSON Inspector ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 🔎 JSON Inspector")
    st.caption("Select a log entry to view its full structured payload.")
    if len(view) > 0:
        _max_idx = len(view) - 1
        _sel = st.slider(
            "Entry (most recent = 0)",
            min_value=0,
            max_value=_max_idx,
            value=min(0, _max_idx),
            key="log_json_slider",
        )
        _record_idx = view.index[len(view) - 1 - _sel] if _sel <= _max_idx else view.index[0]
        _chosen_record = records[_record_idx]
        _c1, _c2 = st.columns([1, 3])
        with _c1:
            st.markdown(
                f"""
                **Level:** `{_chosen_record.get('level', '?')}`  
                **Module:** `{_chosen_record.get('logger', _chosen_record.get('module', '?'))}`  
                **Function:** `{_chosen_record.get('func', '?')}`  
                **Line:** `{_chosen_record.get('line', '?')}`  
                """,
            )
        with _c2:
            st.json(_chosen_record, expanded=True)

    # ── Download ─────────────────────────────────────────────────────────────
    st.divider()
    _dl1, _dl2, _ = st.columns([1, 1, 4])
    with open(lp, "rb") as fh:
        _dl1.download_button(
            "⬇ Download JSONL",
            data=fh.read(),
            file_name="basetruth.jsonl",
            mime="application/x-ndjson",
            key="log_dl_jsonl",
            use_container_width=True,
        )
    # Offer filtered CSV export
    if len(view) > 0:
        _csv_data = view[display_cols].to_csv(index=False)
        _dl2.download_button(
            "⬇ Download CSV",
            data=_csv_data,
            file_name="basetruth_logs_filtered.csv",
            mime="text/csv",
            key="log_dl_csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# DB Viewer page
# ---------------------------------------------------------------------------

_DB_TABLE_LABELS: dict[str, str] = {
    "entities":   "Entities",
    "scans":      "Scans",
    "document_information": "Document Extractions",
    "cases":      "Cases",
    "case_notes": "Case Notes",
}


def _page_database() -> None:
    st.markdown("# Database Viewer")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
This screen gives you direct visibility into what is stored in the system.

- **PostgreSQL tables** — browse entities, scans, cases, and notes row-by-row.
- **MinIO object storage** — list PDF reports and source documents stored in the S3-compatible bucket. Files are automatically uploaded here after each scan, organised by applicant reference (e.g. `BT-000001/payslip_report.pdf`).
- **Danger Zone** — reset (empty) both stores; useful during testing.
  Type `RESET` to confirm before anything is deleted.
"""
        )

    pg_tab, minio_tab, danger_tab = st.tabs(["🐘  PostgreSQL", "🪣  MinIO Storage", "⚠️  Danger Zone"])

    # ── PostgreSQL tab ───────────────────────────────────────────────────────
    with pg_tab:
        if not _DB_IMPORTS_OK or not db_available():
            st.warning(
                "PostgreSQL is not available.  Start the `db` Docker service and ensure "
                "`DATABASE_URL` is set correctly."
            )
        else:
            counts = db_table_counts()
            cc = st.columns(len(_DB_TABLE_LABELS))
            for i, (tbl, lbl) in enumerate(_DB_TABLE_LABELS.items()):
                cc[i].metric(lbl, f"{counts.get(tbl, 0):,}")

            st.divider()

            sel_col, _ = st.columns([2, 6])
            with sel_col:
                chosen_table = st.selectbox(
                    "Browse table",
                    list(_DB_TABLE_LABELS.keys()),
                    format_func=lambda t: _DB_TABLE_LABELS.get(t) or t,
                    key="db_viewer_table",
                )

            rows, total = db_table_rows(chosen_table, limit=500)
            cap = f"**{_DB_TABLE_LABELS[chosen_table]}** — {total:,} rows total"
            if total > 500:
                cap += "  ·  showing most-recent 500"
            st.subheader(cap)

            if rows:
                import pandas as pd  # noqa: PLC0415

                df = pd.DataFrame(rows)
                for col in df.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]", "object"]).columns:
                    try:
                        df[col] = df[col].astype(str)
                    except Exception:
                        pass
                st.dataframe(df, hide_index=True, use_container_width=True, height=480)
            else:
                st.info(f"No rows in **{_DB_TABLE_LABELS[chosen_table]}** yet.")

    # ── MinIO tab ────────────────────────────────────────────────────────────
    with minio_tab:
        if not _DB_IMPORTS_OK:
            st.warning("Store module not loaded.")
        else:
            minio_up = minio_available()
            if not minio_up:
                st.warning(
                    "MinIO is not reachable. Check that the `minio` Docker service is running "
                    "and that `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` are set."
                )
            else:
                stats = minio_bucket_stats()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Bucket", stats.get("bucket", "—"))
                mc2.metric("Objects", f"{stats.get('object_count', 0):,}")
                mc3.metric("Total size", f"{stats.get('total_mb', 0):.1f} MB")

                st.divider()
                objs = minio_list_objects(limit=500)
                if objs:
                    import pandas as pd  # noqa: PLC0415

                    st.subheader(f"{len(objs)} objects (most-recent first)")
                    obj_df = pd.DataFrame([
                        {
                            "Key": o["key"],
                            "Size (KB)": o["size_kb"],
                            "Last Modified": o["last_modified"][:19].replace("T", " "),
                        }
                        for o in objs
                    ])
                    st.dataframe(obj_df, hide_index=True, use_container_width=True, height=400)
                else:
                    st.info("The bucket is empty.")

    # ── Danger Zone tab ──────────────────────────────────────────────────────
    with danger_tab:
        st.markdown("### ⚠️ Irreversible Operations")
        st.error(
            "Actions below **permanently delete data** with no undo. "
            "Type the exact confirmation word shown before pressing the button."
        )

        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("#### 🗄️ Reset PostgreSQL")
            st.caption("Deletes all entities, scans, cases, and notes.")
            
            if "db_reset_success" in st.session_state:
                st.success("✅ Database reset — all tables cleared.")
                del st.session_state["db_reset_success"]

            db_confirm = st.text_input("Type RESET to confirm", key="db_reset_confirm_input", placeholder="RESET")
            if st.button("💀 Empty Database", type="primary", key="db_reset_execute_btn"):
                if db_confirm.strip() == "RESET":
                    ok = reset_db()
                    if ok:
                        st.session_state["db_reset_success"] = True
                        st.rerun()
                    else:
                        st.error("Reset failed — check the Logs page for details.")
                else:
                    st.error("Type exactly `RESET` (all caps) to confirm.")

        with dc2:
            st.markdown("#### 🪣 Reset MinIO Bucket")
            st.caption("Deletes all PDF/image objects in the storage bucket.")
            
            if "minio_reset_success" in st.session_state:
                st.success("✅ MinIO bucket cleared — all objects deleted.")
                del st.session_state["minio_reset_success"]

            minio_confirm = st.text_input("Type RESET to confirm", key="minio_truncate_confirm", placeholder="RESET")
            if st.button("🗑️ Empty MinIO Bucket", type="primary", key="minio_truncate_btn"):
                if minio_confirm.strip() == "RESET":
                    ok = minio_truncate_bucket()
                    if ok:
                        st.session_state["minio_reset_success"] = True
                        st.rerun()
                    else:
                        st.error("Reset failed — MinIO may be offline or misconfigured. Check the Logs page.")
                else:
                    st.error("Type exactly `RESET` (all caps) to confirm.")


# ---------------------------------------------------------------------------
# Index metrics (sidebar summary)
# ---------------------------------------------------------------------------
# Records page  (PostgreSQL entity search + scan history)
# ---------------------------------------------------------------------------

def _page_records() -> None:
    st.markdown("# Records")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Records shows every **applicant (entity)** in the database and all the documents linked to them.

- **Search** — type a name, PAN, Aadhaar, email, phone number, or BaseTruth reference (BT-XXXXXX) to find the right person.
- **Entity card** — click a result to expand the full entity card: all identity fields, risk summary, and individual scan results.
- **Download PDF** — each entity card has a “Download entity report” button that produces a single PDF covering every document scanned for that person.
- **Linking tip** — if the same person appears multiple times, use the Scan or Bulk Scan page with the entity-linking widget to attach future documents to the correct existing record.
"""
        )

    if not _DB_IMPORTS_OK or not db_available():
        st.warning(
            "PostgreSQL is not available. Connect the database to use the Records feature.\n\n"
            "Ensure `DATABASE_URL` is set and the `db` Docker service is healthy."
        )
        return

    # ── Search bar ----------------------------------------------------------
    sc1, sc2, sc3 = st.columns([4, 1.5, 1])
    with sc1:
        search_query = st.text_input(
            "Search",
            placeholder="Name, PAN, Aadhaar, email, phone, BT-XXXXXX…",
            label_visibility="collapsed",
            key="rec_search_query",
        )
    field_opts = {
        "All fields": "all",
        "Name": "name",
        "PAN": "pan",
        "Aadhaar": "aadhar",
        "Email": "email",
        "Phone": "phone",
    }
    with sc2:
        search_field_label = st.selectbox(
            "Field", list(field_opts.keys()), label_visibility="collapsed", key="rec_search_field"
        )
    search_field = field_opts[search_field_label]
    with sc3:
        do_search = st.button("Search →", use_container_width=True, type="primary", key="rec_do_search")

    # Trigger on button click or when query changes
    if do_search or search_query:
        results = search_entities(search_query, search_field, limit=100)
    else:
        results = search_entities("", "all", limit=50)

    # ── Entity table --------------------------------------------------------
    if not results:
        st.info("No records found. Scan some documents first — they will be stored automatically.")
        return

    st.subheader(f"{len(results)} record{'s' if len(results) != 1 else ''} found")
    try:
        import pandas as pd

        tbl_rows = [
            {
                "Ref #": r["entity_ref"],
                "First Name": r["first_name"],
                "Last Name": r["last_name"],
                "PAN": r["pan_number"],
                "Aadhaar": r["aadhar_number"],
                "Email": r["email"],
                "Phone": r["phone"],
                "Scans": r["scan_count"],
                "Latest Risk": str(r["latest_risk"]).title() if r["latest_risk"] else "—",
                "Score": r["latest_score"] if r["latest_score"] is not None else "—",
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, use_container_width=True)
    except ImportError:
        for r in results:
            st.write(f"{r['entity_ref']} — {r['first_name']} {r['last_name']}")

    st.divider()

    # ── Entity detail panel -------------------------------------------------
    ref_options = [r["entity_ref"] for r in results]
    selected_ref = st.selectbox(
        "Open entity record",
        options=ref_options,
        format_func=lambda ref: next(
            (
                f"{ref}  •  {r['first_name']} {r['last_name']}"
                for r in results if r["entity_ref"] == ref
            ),
            ref,
        ),
        key="rec_selected_ref",
    )

    selected_entity = next((r for r in results if r["entity_ref"] == selected_ref), None)
    if not selected_entity:
        return

    # ---- Identity card -----------------------------------------------------
    # Identity card — uses Streamlit's injected var(--text-color) +
    # var(--secondary-background-color), which auto-adapt to dark/light mode.
    _pan   = selected_entity['pan_number'] or '—'
    _aadh  = selected_entity['aadhar_number'] or '—'
    _email = selected_entity['email'] or '—'
    _phone = selected_entity['phone'] or '—'
    _scans = selected_entity['scan_count']
    _since = str(selected_entity['created_at'])[:10]
    _fname = selected_entity['first_name']
    _lname = selected_entity['last_name']
    st.markdown(
        f"""
        <div class="bt-entity-card" style="
          background:var(--secondary-background-color,#ffffff);
          border:1px solid rgba(99,102,241,0.20);
          border-left:4px solid #6366f1;
          box-shadow:0 2px 12px rgba(99,102,241,0.08);">
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="width:44px;height:44px;border-radius:12px;
              background:linear-gradient(135deg,#6366f1,#8b5cf6);
              display:flex;align-items:center;justify-content:center;
              font-size:18px;font-weight:800;color:#fff;flex-shrink:0;">
              {_fname[0].upper() if _fname else '?'}
            </div>
            <div>
              <div class="bt-entity-name">
                {_fname} {_lname}
              </div>
              <div style="margin-top:4px;">
                <span style="font-size:11px;font-weight:700;
                  background:rgba(99,102,241,0.12);color:#6366f1;
                  border:1px solid rgba(99,102,241,0.28);
                  padding:2px 10px;border-radius:6px;">{selected_ref}</span>
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px 24px;">
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">PAN</div>
              <div class="bt-entity-field-value">{_pan}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Aadhaar</div>
              <div class="bt-entity-field-value">{_aadh}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Email</div>
              <div class="bt-entity-field-value">{_email}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Phone</div>
              <div class="bt-entity-field-value">{_phone}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Documents Scanned</div>
              <div class="bt-entity-field-value">{_scans}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Member Since</div>
              <div class="bt-entity-field-value">{_since}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Entity-level PDF download ----------------------------------------
    _entity_pdf, _entity_pdf_src = get_entity_latest_pdf(selected_ref)
    if _entity_pdf:
        _pdf_label = f"{Path(_entity_pdf_src).stem}_report.pdf" if _entity_pdf_src else f"{selected_ref}_report.pdf"
        st.download_button(
            "📥  Download Latest PDF Report",
            data=_entity_pdf,
            file_name=_pdf_label,
            mime="application/pdf",
            key=f"entity_pdf_{selected_ref}",
            type="primary",
        )
    else:
        st.caption("No PDF report available yet for this entity.")

    # ---- Edit identity fields inline --------------------------------------
    with st.expander("✏️  Edit identity details", expanded=False):
        with st.form(f"edit_entity_{selected_ref}"):
            e1, e2 = st.columns(2)
            f_first = e1.text_input("First name", value=selected_entity["first_name"])
            f_last = e2.text_input("Last name", value=selected_entity["last_name"])
            e3, e4 = st.columns(2)
            f_email = e3.text_input("Email", value=selected_entity["email"])
            f_phone = e4.text_input("Phone", value=selected_entity["phone"])
            e5, e6 = st.columns(2)
            f_pan = e5.text_input("PAN number", value=selected_entity["pan_number"])
            f_aadhar = e6.text_input("Aadhaar number", value=selected_entity["aadhar_number"])
            if st.form_submit_button("Save changes", type="primary"):
                result = update_entity(
                    selected_ref,
                    {
                        "first_name": f_first,
                        "last_name": f_last,
                        "email": f_email,
                        "phone": f_phone,
                        "pan_number": f_pan,
                        "aadhar_number": f_aadhar,
                    },
                )
                if result:
                    st.success("Record updated.")
                    st.rerun()
                else:
                    st.error("Update failed — check DB connection.")

    # ---- Scan history for this entity ------------------------------------
    scans = get_entity_scans(selected_ref)
    st.subheader(f"Document history  ({len(scans)} scan{'s' if len(scans) != 1 else ''})")

    if not scans:
        st.info("No documents scanned for this entity yet.")
        return

    # ── Flat table with inline JSON download — expand row for forensic details
    st.caption("Expand a row to view forensic details. Download the entity PDF report above.")
    for sc in scans:
        risk      = str(sc.get("risk_level", "low")).lower()
        score     = sc.get("truth_score", "")
        doc_type  = str(sc.get("document_type", "generic")).replace("_", " ").title()
        fname     = sc.get("source_name", "unknown")
        ts        = str(sc.get("generated_at", ""))[:19].replace("T", " ")
        risk_icon = {"high": "🚨", "medium": "⚠️", "review": "🔷"}.get(risk, "✅")
        score_str = f"{score}/100" if isinstance(score, int) else "—"

        # Row: icon | name | type | score | date | JSON btn
        row_c1, row_c2, row_c3, row_c4, row_c5, row_c6 = st.columns(
            [0.4, 3.2, 2, 1, 2, 1.4]
        )
        row_c1.markdown(risk_icon)
        row_c2.markdown(f"**{fname}**")
        row_c3.markdown(doc_type)
        row_c4.markdown(f"**{score_str}**")
        row_c4.markdown(_badge(risk), unsafe_allow_html=True)
        row_c5.caption(ts)

        row_c6.download_button(
            "⬇ JSON",
            data=json.dumps(sc["report_json"], indent=2, ensure_ascii=False),
            file_name=f"{Path(fname).stem}_verification.json",
            mime="application/json",
            key=f"rec_json_{sc['id']}",
            use_container_width=True,
        )

        # Forensic details still available on demand
        with st.expander(f"🔬 Forensic details — {fname}", expanded=False):
            _render_report_summary(sc["report_json"])


# ---------------------------------------------------------------------------
# Index metrics strip
# ---------------------------------------------------------------------------

def _render_index_metrics() -> None:
    """Render the slim top-of-page stats bar — uses DB when available."""
    if _DB_IMPORTS_OK and db_available():
        stats = db_stats()
        cols = st.columns(4)
        cols[0].metric("Entities in DB", stats.get("entities", 0),
                       help="Unique individuals / organisations stored in PostgreSQL.")
        cols[1].metric("Scans in DB", stats.get("scans", 0),
                       help="Total document scans persisted to the database.")
        cols[2].metric("High-Risk Scans", stats.get("high_risk", 0),
                       help="Scans flagged high-risk (truth score < 60).")
        try:
            from basetruth.db import db_session
            from basetruth.db import Case as _Case
            from sqlalchemy import func as _func
            with db_session() as _s:
                pending = (
                    _s.query(_func.count(_Case.id))
                    .filter(_Case.disposition == "open")
                    .scalar() or 0
                )
            cols[3].metric("Pending Review", pending,
                           help="Cases still open — go to Cases to approve or reject.")
        except Exception:  # noqa: BLE001
            cols[3].metric("Pending Review", "—")
    else:
        # DB offline: show a minimal "offline" notice instead of stale file-system counts
        st.info(
            "📴 **Database offline** — connect PostgreSQL to see live statistics. "
            "Document scans still work and are saved to disk.",
            icon=None,
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

    if page == "dashboard":
        _page_dashboard(service)
    elif page == "scan":
        _page_scan(service)
    elif page == "bulk":
        _page_bulk(service)
    elif page == "cases":
        _page_cases(service)
    elif page == "records":
        _page_records()
    elif page == "reports":
        _page_reports(service)
    elif page == "datasources":
        _page_datasources(service)
    elif page == "logs":
        _page_logs()
    elif page == "database":
        _page_database()
    elif page == "settings":
        _page_settings()
    else:
        st.warning(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
