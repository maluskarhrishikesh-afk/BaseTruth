"""Log Analyzer page."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from basetruth.ui.components import _LOGGER_OK, _log_path, _page_title


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts_str: str) -> str:
    """Extract HH:MM:SS from an ISO-8601 timestamp written by the JSON formatter.

    The logger writes timestamps like '2026-04-15T14:32:45.123456+00:00'.
    We take characters 11-19 to get 'HH:MM:SS'.  Any shorter/malformed
    strings fall back gracefully so the UI never crashes on bad data.
    """
    s = str(ts_str or "")
    if len(s) >= 19 and "T" in s:
        return s[11:19]
    if len(s) >= 8:
        return s[-8:]
    return s or "—"


def _build_display_msg(row: dict) -> str:
    """Build a self-contained one-liner from a log record.

    The raw 'msg' field might be short (e.g. 'Bulk scan worker failed').
    We merge in the 'file', 'path', and 'error' extra fields — which carry
    the actual failure details — so every table row is useful on its own
    without needing to open the JSON inspector.
    """
    msg = str(row.get("msg", ""))

    # Prepend the affected filename so the operator knows which document failed.
    file_ctx = str(row.get("file", "") or "")
    path_ctx = str(row.get("path", "") or "")
    name_ctx = file_ctx or (Path(path_ctx).name if path_ctx else "")
    if name_ctx and name_ctx not in msg:
        msg = f"[{name_ctx}]  {msg}"

    # Append the error string if it adds information not already in the message.
    error_ctx = str(row.get("error", "") or "")
    if error_ctx and error_ctx not in msg:
        msg = f"{msg}  ·  {error_ctx[:400]}"

    return msg


def _level_css(level: str) -> str:
    """Return a CSS class suffix for the given log level."""
    return {"ERROR": "error", "WARNING": "warning", "INFO": "info", "DEBUG": "debug"}.get(
        level.upper(), "debug"
    )


def _level_icon(level: str) -> str:
    return {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🟢", "DEBUG": "⚪"}.get(
        level.upper(), "⚫"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _page_logs() -> None:
    import pandas as pd  # noqa: PLC0415

    st.markdown(
        """
        <style>
        /* ── Log feed rows ── */
        .bt-log-row {
            padding: 8px 14px;
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: .84rem;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            border-left: 4px solid transparent;
            line-height: 1.5;
        }
        .bt-log-error   { background:#fff1f2; border-left-color:#ef4444; color:#7f1d1d; }
        .bt-log-warning { background:#fffbeb; border-left-color:#f59e0b; color:#78350f; }
        .bt-log-info    { background:#f0fdf4; border-left-color:#22c55e; color:#14532d; }
        .bt-log-debug   { background:#f8fafc; border-left-color:#94a3b8; color:#475569; }
        .bt-log-raw     { background:#fafafa; border-left-color:#d1d5db; color:#374151; }
        /* ── Metric cards ── */
        .bt-metric-row  { display:flex; gap:14px; margin:14px 0; flex-wrap:wrap; }
        .bt-metric-card {
            flex:1; min-width:120px; padding:16px 18px; border-radius:12px;
            display:flex; flex-direction:column; gap:2px;
        }
        .bt-metric-card .val { font-size:2rem; font-weight:800; line-height:1.1; }
        .bt-metric-card .lbl { font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; opacity:.8; font-weight:600; }
        .btm-total   { background:linear-gradient(135deg,#e0e7ff,#c7d2fe); color:#3730a3; }
        .btm-error   { background:linear-gradient(135deg,#fee2e2,#fca5a5); color:#991b1b; }
        .btm-warn    { background:linear-gradient(135deg,#fef3c7,#fde68a); color:#92400e; }
        .btm-info    { background:linear-gradient(135deg,#d1fae5,#6ee7b7); color:#065f46; }
        /* ── Traceback block ── */
        .bt-traceback {
            background:#1e1e2e; color:#cdd6f4; border-radius:8px;
            padding:12px 16px; font-family:monospace; font-size:.78rem;
            white-space:pre-wrap; overflow-x:auto; margin-top:8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ──────────────────────────────────────────────────────────────
    _h1, _h2, _h3 = st.columns([5, 1, 1])
    _h1.markdown(_page_title("📋", "Log Analyzer"), unsafe_allow_html=True)
    if _h2.button("🔄 Refresh", use_container_width=True, key="log_refresh"):
        st.rerun()

    lp = _log_path() if _LOGGER_OK else None
    if lp is None or not lp.exists():
        st.info(
            "No log file found yet. Run some scans and the log file will appear here.\n\n"
            f"Expected location: `{lp}`"
        )
        return

    if _h3.button("🗑 Clear", use_container_width=True, key="log_clear"):
        try:
            lp.write_text("", encoding="utf-8")
            st.success("Log file cleared.")
            st.rerun()
        except Exception as _clr_exc:  # noqa: BLE001
            st.error(f"Could not clear log file: {_clr_exc}")

    # ── Read & parse JSONL ───────────────────────────────────────────────────
    raw_lines: list = []
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            for _line in fh:
                _line = _line.strip()
                if _line:
                    raw_lines.append(_line)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read log file: {exc}")
        return

    records: list = []
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

    n_total = len(df)
    n_err   = int((df["level"] == "ERROR").sum())
    n_warn  = int((df["level"] == "WARNING").sum())
    n_info  = int(df["level"].isin(["INFO", "DEBUG"]).sum())

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="bt-metric-row">
          <div class="bt-metric-card btm-total"><span class="val">{n_total:,}</span><span class="lbl">Total Entries</span></div>
          <div class="bt-metric-card btm-error"><span class="val">{n_err:,}</span><span class="lbl">🔴 Errors</span></div>
          <div class="bt-metric-card btm-warn"><span class="val">{n_warn:,}</span><span class="lbl">🟡 Warnings</span></div>
          <div class="bt-metric-card btm-info"><span class="val">{n_info:,}</span><span class="lbl">🟢 Info + Debug</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters ──────────────────────────────────────────────────────────────
    _f1, _f2, _f3 = st.columns([1.5, 2, 3])
    level_opts = ["ALL"] + sorted(df["level"].unique().tolist())
    module_opts = ["ALL"] + sorted([m for m in df["logger"].unique() if m])
    
    if "log_level_sel_v3" not in st.session_state:
        st.session_state["log_level_sel_v3"] = "ALL"
    if "log_module_sel_v3" not in st.session_state:
        st.session_state["log_module_sel_v3"] = "ALL"
        
    chosen_level  = _f1.selectbox("Level", level_opts, key="log_level_sel_v3")
    chosen_module = _f2.selectbox("Module", module_opts, key="log_module_sel_v3")
    search_text   = _f3.text_input("Search messages", placeholder="keyword or error text…", key="log_search_v3")

    def _set_log_level(val: str) -> None:
        st.session_state["log_level_sel_v3"] = val if val in level_opts else "ALL"

    # Quick-filter buttons — clicking sets the level filter.
    _qf = st.columns(5)
    _qf_labels = [("🔴 Errors", "ERROR"), ("🟡 Warnings", "WARNING"), ("🟢 Info", "INFO"), ("⚪ Debug", "DEBUG"), ("📋 All", "ALL")]
    for _col, (_lbl, _val) in zip(_qf, _qf_labels):
        _col.button(_lbl, use_container_width=True, key=f"qf_{_val}", on_click=_set_log_level, args=(_val,))

    # ── Apply filters ─────────────────────────────────────────────────────────
    view = df.copy()
    if chosen_level != "ALL":
        view = view[view["level"] == chosen_level]
    if chosen_module != "ALL":
        view = view[view["logger"] == chosen_module]
    if search_text:
        # Search across the message, error extra, and file extra fields.
        _mask = (
            view["msg"].str.contains(search_text, case=False, na=False)
            | view.get("error", pd.Series(dtype=str)).fillna("").str.contains(search_text, case=False, na=False)
            | view.get("file",  pd.Series(dtype=str)).fillna("").str.contains(search_text, case=False, na=False)
        )
        view = view[_mask]

    # Ascending order — oldest entries at the top so the user can scroll
    # down naturally to reach the most recent log lines (same reading
    # direction as a terminal / tail -f output).
    view = view.reset_index(drop=True)

    st.caption(f"Showing **{len(view):,}** of {n_total:,} entries — oldest first")

    # ── Main log table (CloudWatch Style View) ────────────────────────────────
    if len(view) > 0:
        log_lines = []
        for _, row in view.iterrows():
            ts = _fmt_ts(row.get("ts", ""))
            lvl = str(row.get("level", "")).ljust(7)
            src = ".".join(str(row.get("logger", "")).split(".")[-2:]) if "." in str(row.get("logger", "")) else str(row.get("logger", ""))
            msg = _build_display_msg(row.to_dict())
            
            # Simple color coding for level
            color = "#d4d4d4"
            if "ERROR" in lvl: color = "#ef4444"
            elif "WARN" in lvl: color = "#f59e0b"
            elif "INFO" in lvl: color = "#22c55e"
            
            log_lines.append(f"<span style='color:#888'>{ts}</span> <span style='color:{color}; font-weight:bold'>{lvl}</span> <span style='color:#a78bfa'>[{src}]</span> {msg}")
            
        cloudwatch_html = "<br/>".join(log_lines)
        
        st.markdown(
            f"""
            <div style="background-color: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 6px; 
                        font-family: 'Consolas', 'Courier New', monospace; height: 80vh; 
                        overflow-y: auto; white-space: pre-wrap; font-size: 0.85rem; line-height: 1.4;
                        border: 1px solid #30363d;">
{cloudwatch_html}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No log entries match your filters.")

    # ── Errors & Tracebacks ───────────────────────────────────────────────────
    # Show the most recent errors with their full traceback so the developer
    # gets all the context needed to diagnose a failure without digging through
    # the raw JSONL file.
    _err_view = view[view["level"] == "ERROR"].head(20)
    if len(_err_view) > 0:
        st.divider()
        st.markdown("#### 🔴 Recent Errors & Tracebacks")
        st.caption(f"Showing the {len(_err_view)} most recent error(s) from the current filter.")
        for _, _erow in _err_view.iterrows():
            _edict   = _erow.to_dict()
            _emsg    = _build_display_msg(_edict)
            _ets     = _fmt_ts(_edict.get("ts", ""))
            _esrc    = ".".join(str(_edict.get("logger", "")).split(".")[-2:]) or "?"
            _exc_txt = str(_edict.get("exc", "") or "")
            # Extra context fields (doc, scan, path etc.) — skip internal keys.
            _skip = {"ts", "level", "logger", "module", "func", "line", "msg", "exc"}
            _extras = {k: v for k, v in _edict.items() if k not in _skip and v not in (None, "", [], {})}

            with st.expander(f"🔴 {_ets}  ·  {_esrc}  —  {_emsg[:120]}", expanded=False):
                # Full message in a styled box.
                st.markdown(
                    f'<div class="bt-log-row bt-log-error"><strong>[ERROR]</strong> {_ets} — {_emsg}</div>',
                    unsafe_allow_html=True,
                )
                # Traceback block if present.
                if _exc_txt:
                    st.markdown("**Traceback:**")
                    st.markdown(
                        f'<div class="bt-traceback">{_exc_txt}</div>',
                        unsafe_allow_html=True,
                    )
                # Context extras (file, path, error string, etc.).
                if _extras:
                    st.markdown("**Context:**")
                    st.json(_extras)




