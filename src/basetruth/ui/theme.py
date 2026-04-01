"""BaseTruth UI — theme constants, CSS, and pure-Python badge/card helpers.

No Streamlit imports here — this module can be used safely in server-side
rendering or test environments that do not have a running Streamlit session.
"""
from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Risk / status colour palettes
# ---------------------------------------------------------------------------

_RISK_COLORS: Dict[str, tuple] = {
    "high":   ("rgba(220,38,38,0.10)",  "#dc2626", "rgba(220,38,38,0.30)"),
    "medium": ("rgba(217,119,6,0.10)",  "#d97706", "rgba(217,119,6,0.30)"),
    "low":    ("rgba(21,128,61,0.10)",  "#16a34a", "rgba(21,128,61,0.30)"),
    "review": ("rgba(29,78,216,0.10)",  "#2563eb", "rgba(29,78,216,0.30)"),
}

_STATUS_COLORS: Dict[str, str] = {
    "new": "#64748b",
    "triage": "#7c3aed",
    "investigating": "#2563eb",
    "pending_client": "#d97706",
    "closed": "#16a34a",
}

_DISPOSITION_ICONS: Dict[str, str] = {
    "open": "🔓",
    "monitor": "👁",
    "escalate": "⚠️",
    "cleared": "✅",
    "fraud_confirmed": "🚨",
}

# ---------------------------------------------------------------------------
# Full-page CSS (injected once in app.py via st.markdown(_CSS, unsafe_allow_html=True))
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ============================================================
   BASETRUTH UI — Modern Elegant Theme v2
   ============================================================ */

/* ---- Typography ------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                 'Noto Color Emoji', sans-serif !important;
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
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                 'Noto Color Emoji', sans-serif !important;
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

/* Sidebar secondary/default button */
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
    color: var(--text-color, #64748b) !important;
    opacity: 0.65 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    color: var(--text-color, #0f172a) !important;
}

/* ---- Metric text dark mode overrides ---- */
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
   ADDITIONAL POLISH
   ============================================================ */
:root { --text-color: #1e293b; }
@media (prefers-color-scheme: dark) { :root { --text-color: #f1f5f9; } }
[data-testid="stApp"][data-theme="dark"] { --text-color: #f1f5f9; }

.bt-search-card [data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
    gap: 0.75rem !important;
}
.bt-search-card [data-testid="column"]:last-child > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

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

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.28);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-size: 0.875rem !important;
    line-height: 1.55 !important;
}

[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #6366f1, #8b5cf6) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 999px !important;
    overflow: hidden !important;
}

[data-testid="stFileUploaderDropzone"] {
    border-color: rgba(99,102,241,0.35) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1 !important;
    background: rgba(99,102,241,0.04) !important;
}

[data-testid="stExpander"] summary svg { color: #6366f1 !important; }
[data-testid="stExpander"] summary:hover {
    background: rgba(99,102,241,0.06) !important;
}

[data-testid="stCaptionContainer"] {
    font-size: 11.5px !important;
    line-height: 1.55 !important;
    opacity: 0.75 !important;
}

[data-testid="stSpinner"] > div {
    border-top-color: #6366f1 !important;
}

[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#4f46e5 0%,#6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.30) !important;
}

.main h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.15 !important;
}

.main [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    border-bottom: 1px solid var(--bt-divider, #e2e8f0) !important;
    padding-bottom: 0.35rem !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stSidebar"] .stCaption:last-of-type {
    position: absolute;
    bottom: 1.25rem;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 10px !important;
    color: #334155 !important;
}

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

.bt-entity-card {
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
.bt-entity-card:hover {
    border-left-color: #8b5cf6 !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.18), 0 2px 8px rgba(0,0,0,0.08) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--bt-divider) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stDataFrame"] > div {
    border-radius: 12px !important;
    overflow: hidden !important;
}

.main .block-container > div {
    animation: bt-fadein 0.22s ease forwards;
}
@keyframes bt-fadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

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

.bt-score-card {
    transition: box-shadow 0.25s ease !important;
}
.bt-score-card:hover {
    box-shadow: 0 6px 24px rgba(99,102,241,0.15) !important;
}

.bt-pdf-banner {
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: box-shadow 0.2s ease;
}
.bt-pdf-banner:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.18);
}

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
.bt-search-card [data-testid="column"]:last-child .stButton {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.bt-search-card [data-testid="column"]:last-child .stButton > button {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
}

.main h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] hr {
    margin: 0.5rem 0 !important;
}

[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

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

[data-testid="stApp"][data-theme="dark"] .bt-pdf-banner div[style*="color:var(--text-color"] {
    color: #f1f5f9 !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Badge / card helpers (pure Python — return HTML strings)
# ---------------------------------------------------------------------------

def _badge(level: str, label: str = "") -> str:
    """Return an HTML risk-level badge string."""
    bg, fg, border = _RISK_COLORS.get(str(level).lower(), _RISK_COLORS["low"])
    text = label or str(level).upper()
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {border};'
        f'padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700;'
        f'letter-spacing:0.04em;white-space:nowrap;">{text}</span>'
    )


def _status_badge(status: str) -> str:
    """Return an HTML status badge string."""
    color = _STATUS_COLORS.get(str(status).lower(), "#64748b")
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600;">'
        f'{status.upper().replace("_", " ")}</span>'
    )


def _score_card(score: Any, risk_level: str) -> str:
    """Return an HTML truth-score card string."""
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
    """Return an emoji icon for a forensic signal."""
    if sig.get("passed") is True:
        return "✅"
    sev = str(sig.get("severity", "info")).lower()
    return {"high": "🚨", "medium": "⚠️", "low": "🔷"}.get(sev, "ℹ️")
