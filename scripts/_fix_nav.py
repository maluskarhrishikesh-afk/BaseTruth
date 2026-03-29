"""Fix broken emoji bytes in navigation and dark mode CSS."""
import re

path = 'src/basetruth/ui/app.py'
content = open(path, encoding='utf-8').read()

# ── 1. Fix navigation emojis ──────────────────────────────────────────────
# The broken chars: \ufffd is a UTF-8 replacement character

# Replace broken Records label: \ufffd  Records  →  🗂️  Records
content = re.sub(
    r'"\ufffd\s+Records"',
    '"🗂️  Records"',
    content,
)

# Replace broken Reports label: \ufffd📊  Reports  →  📊  Reports
content = re.sub(
    r'"\ufffd📊\s+Reports"',
    '"📊  Reports"',
    content,
)

# ── 2. Fix dark mode metric text visibility ────────────────────────────────
# The problem: CSS variables for dark mode may not apply because Streamlit 
# doesn't always add data-theme="dark" to stApp.  Explicitly set text colors
# for metric values and labels in dark mode using multiple robust selectors.

old_metric_css = '''/* ---- Metric cards ----------------------------------------- */
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
}'''

new_metric_css = '''/* ---- Metric cards ----------------------------------------- */
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
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    color: #0f172a !important;
}

/* ---- Metric text — dark mode overrides -------------------- */
/* Multiple selectors for cross-version Streamlit compatibility */
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetricValue"],
[data-theme="dark"] [data-testid="stMetricValue"],
body[data-theme="dark"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}
[data-testid="stApp"][data-theme="dark"] [data-testid="stMetricLabel"],
[data-theme="dark"] [data-testid="stMetricLabel"],
body[data-theme="dark"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
}'''

if old_metric_css in content:
    content = content.replace(old_metric_css, new_metric_css)
    print('Metric CSS patched.')
else:
    print('WARNING: Could not find old metric CSS block — skipping metric patch.')

open(path, 'w', encoding='utf-8').write(content)
print('Done. File saved.')
